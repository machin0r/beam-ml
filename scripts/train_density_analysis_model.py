#!/usr/bin/env python3
"""
Train baseline models with MLflow tracking
"""

import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import matplotlib

from src.data.load import load_training_data
from src.models.train import train_and_log_model
from src.config import MLFLOW_TRACKING_URI, DENSITY_EXPERIMENT_NAME
from src.logging_config import get_logger

matplotlib.use("Agg")

logger = get_logger(__name__)


def main() -> None:
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(DENSITY_EXPERIMENT_NAME)

    logger.info("Loading data...")
    X_train, X_test, Y_train, Y_test, expected_columns = load_training_data()

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Total features: {len(expected_columns)}")

    logger.info("=" * 70)
    logger.info("Training Models".center(70))
    logger.info("=" * 70)

    # Define models to train
    models = {
        "Linear": Ridge(alpha=1.0),
        "Random_Forest": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }

    results = []
    for name, model in models.items():
        _, metrics = train_and_log_model(model, name, X_train, X_test, Y_train, Y_test)
        results.append((name, metrics.rmse, metrics.r2))

    logger.info("=" * 70)
    logger.info("Summary".center(70))
    logger.info("=" * 70)
    best_model = min(results, key=lambda x: x[1])
    logger.info(
        f"Best model: {best_model[0]} (RMSE: {best_model[1]:.3f}, R²: {best_model[2]:.3f})"
    )
    logger.info("All experiments logged to MLflow!")
    logger.info("Run 'mlflow ui' to view results")


if __name__ == "__main__":
    main()
