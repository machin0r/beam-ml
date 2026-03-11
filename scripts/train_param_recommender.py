"""
Train the parameter recommender model.

For a given material + machine + target density, predicts 80% parameter ranges
(q=0.10 lower bound, q=0.90 upper bound) for each of the 4 key process parameters
using quantile XGBoost regression.

Saves 8 .pkl files and a schema.json to models/param_recommender/.
"""

import json
import pickle
from pathlib import Path

import mlflow
import pandas as pd
from xgboost import XGBRegressor

from src.config import PROJECT_ROOT, MLFLOW_TRACKING_URI
from src.features.engineering import prepare_features_for_training
from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "lpbf_enriched_v1.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "param_recommender"
EXPERIMENT_NAME = "lpbf_param_recommender"

# Input features (numeric columns used as model inputs)
RECOMMENDER_NUMERIC_FEATURES = [
    "Melting Point (K)",
    "Thermal Conductivity (W/mK)",
    "Density (g/cm^3)",
    "Specific Heat Capacity (J/kgK)",
    "D50 μm",
    "RD (%)",
]

# Categorical input features (one-hot encoded)
RECOMMENDER_CATEGORICAL_FEATURES = [
    "Material",
    "Atmosphere",
    "Printer Model",
]

# Target process parameters to predict
TARGET_COLUMNS = [
    "Laser Power (W)",
    "Scan Speed (mm/s)",
    "Hatch space (mm)",
    "Layer thickness (mm)",
]

# Clean names for file system (used in pkl filenames)
TARGET_CLEAN_NAMES = {
    "Laser Power (W)": "laser_power_w",
    "Scan Speed (mm/s)": "scan_speed_mm_s",
    "Hatch space (mm)": "hatch_space_mm",
    "Layer thickness (mm)": "layer_thickness_mm",
}

QUANTILES = [0.10, 0.90]

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": 0,
}


def load_and_prepare_data():
    """Load training data and prepare input/output matrices."""
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Drop rows missing any required column
    required_cols = RECOMMENDER_NUMERIC_FEATURES + RECOMMENDER_CATEGORICAL_FEATURES + TARGET_COLUMNS
    df = df.dropna(subset=required_cols)
    logger.info(f"Rows after dropping NAs: {len(df)}")

    # Build input feature matrix
    input_df = df[RECOMMENDER_NUMERIC_FEATURES + RECOMMENDER_CATEGORICAL_FEATURES].copy()
    X, expected_columns = prepare_features_for_training(
        input_df,
        numeric_features=RECOMMENDER_NUMERIC_FEATURES,
        categorical_features=RECOMMENDER_CATEGORICAL_FEATURES,
    )

    return X, df[TARGET_COLUMNS], expected_columns


def train_recommender():
    """Train 8 quantile models and save artifacts."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y_all, expected_columns = load_and_prepare_data()
    logger.info(f"Input shape: {X.shape}, {len(TARGET_COLUMNS)} targets, 2 quantiles = 8 models")

    with mlflow.start_run(run_name="quantile_xgb_recommender"):
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("targets", TARGET_COLUMNS)
        mlflow.log_param("quantiles", QUANTILES)
        mlflow.log_params(XGB_PARAMS)

        for target_col in TARGET_COLUMNS:
            y = y_all[target_col]
            clean_name = TARGET_CLEAN_NAMES[target_col]

            for q in QUANTILES:
                q_label = f"q{int(q * 100):02d}"
                model_filename = f"{clean_name}_{q_label}.pkl"

                logger.info(f"Training {clean_name} @ q={q} ...")

                model = XGBRegressor(
                    objective="reg:quantileerror",
                    quantile_alpha=q,
                    **XGB_PARAMS,
                )
                model.fit(X, y)

                # Save model
                model_path = OUTPUT_DIR / model_filename
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

                mlflow.log_artifact(str(model_path))
                logger.info(f"  Saved: {model_path}")

        # Save schema
        schema = {
            "expected_columns": expected_columns,
            "numeric_features": RECOMMENDER_NUMERIC_FEATURES,
            "categorical_features": RECOMMENDER_CATEGORICAL_FEATURES,
            "targets": TARGET_COLUMNS,
            "target_clean_names": TARGET_CLEAN_NAMES,
            "quantiles": QUANTILES,
            "n_features": len(expected_columns),
        }
        schema_path = OUTPUT_DIR / "schema.json"
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2)

        mlflow.log_artifact(str(schema_path))
        logger.info(f"Schema saved: {schema_path}")

    logger.info("=" * 70)
    logger.info("Parameter recommender training complete!")
    logger.info(f"Artifacts saved to: {OUTPUT_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    train_recommender()
