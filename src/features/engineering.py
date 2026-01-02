"""
Feature engineering for LPBF density prediction
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

from src.logging_config import get_logger

logger = get_logger(__name__)


# Feature definitions
NUMERIC_FEATURES = [
    "Laser Power (W)",
    "Scan Speed (mm/s)",
    "Hatch space (mm)",
    "Layer thickness (mm)",
    "Spot size (mm)",
    "D50 μm",
    "Melting Point (K)",
    "Thermal Conductivity (W/mK)",
    "Density (g/cm^3)",
    "Specific Heat Capacity (J/kgK)",
]

CATEGORICAL_FEATURES = [
    "Material",
    "Density Measurement Method",
    "Atmosphere",
    "Printer Model",
]


def prepare_features_for_training(
    df: pd.DataFrame,
    numeric_features: List[str] = NUMERIC_FEATURES,
    categorical_features: List[str] = CATEGORICAL_FEATURES,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare features for training with one-hot encoding

    Args:
        df: Raw DataFrame with all features
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names to encode

    Returns:
        Tuple of (encoded_dataframe, expected_column_names)
    """
    logger.info("Preparing features for training...")

    all_features = numeric_features + categorical_features
    X = df[all_features].copy()

    # One-hot encode categorical features
    for cat_col in categorical_features:
        logger.debug(f"Encoding {cat_col}: {X[cat_col].nunique()} unique values")

    X = pd.get_dummies(
        X,
        columns=categorical_features,
        drop_first=False,
        dtype=int,
    )

    expected_columns = X.columns.tolist()

    logger.info(f"Training features prepared: {len(expected_columns)} total columns")
    logger.info(
        f"  - {len(numeric_features)} numeric features\n"
        f"  - {len(categorical_features)} categorical features "
        f"({len(expected_columns) - len(numeric_features)} encoded columns)"
    )

    return X, expected_columns


def prepare_features_for_inference(
    df: pd.DataFrame,
    expected_columns: List[str],
    numeric_features: List[str] = NUMERIC_FEATURES,
    categorical_features: List[str] = CATEGORICAL_FEATURES,
) -> pd.DataFrame:
    """
    Prepare features for inference

    Args:
        df: Raw DataFrame with all features
        expected_columns: Column names from training
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names to encode

    Returns:
        DataFrame with columns matching training exactly
    """
    logger.debug("Preparing features for inference...")

    all_features = numeric_features + categorical_features
    X = df[all_features].copy()

    # One-hot encode categorical features
    X = pd.get_dummies(
        X,
        columns=categorical_features,
        drop_first=False,
        dtype=int,
    )

    # Add missing columns with zeros (handles new categories in training)
    for col in expected_columns:
        if col not in X.columns:
            logger.debug(f"Adding missing column: {col}")
            X[col] = 0

    # Remove extra columns (handles new categories at inference)
    extra_cols = set(X.columns) - set(expected_columns)
    if extra_cols:
        logger.warning(
            f"Found {len(extra_cols)} unexpected columns at inference time. "
            f"These will be dropped: {extra_cols}"
        )
        X = X.drop(columns=list(extra_cols))

    # Reorder to match training exactly
    X = X[expected_columns]

    logger.debug(f"Inference features prepared: {len(X.columns)} columns")

    return X


def save_feature_schema(
    expected_columns: List[str],
    numeric_features: List[str],
    categorical_features: List[str],
    output_path: Path,
) -> None:
    """
    Save feature schema to JSON for inference

    Args:
        expected_columns: List of expected column names from training
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        output_path: Path to save schema JSON
    """
    schema = {
        "expected_columns": expected_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "n_features": len(expected_columns),
        "n_numeric": len(numeric_features),
        "n_categorical": len(categorical_features),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)

    logger.info(f"Feature schema saved to {output_path}")
    logger.info(f"  - Total features: {schema['n_features']}")


def load_feature_schema(schema_path: Path) -> Dict:
    """
    Load feature schema from JSON

    Args:
        schema_path: Path to schema JSON file

    Returns:
        Dictionary with schema information
    """
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Feature schema not found at {schema_path}. "
            "Train a model first to generate the schema."
        )

    with open(schema_path) as f:
        schema = json.load(f)

    logger.info(f"Feature schema loaded from {schema_path}")
    logger.info(f"  - Total features: {schema['n_features']}")

    return schema
