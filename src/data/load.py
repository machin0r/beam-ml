"""
Data loading utilities for training
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, List

from src.features.engineering import (
    prepare_features_for_training,
    save_feature_schema,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.config import TARGET_COLUMN, PROJECT_ROOT
from src.logging_config import get_logger

logger = get_logger(__name__)


def load_training_data(
    test_size: float = 0.2,
    random_state: int = 42,
    target_column: str = TARGET_COLUMN,
    save_schema: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Load and prepare training data

    Args:
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        target_column: Name of target column
        save_schema: Whether to save feature schema for inference

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, expected_columns)
    """
    logger.info("Loading training data...")

    # Load raw data
    data_path = PROJECT_ROOT / "data" / "processed" / "lpbf_enriched_v1.csv"
    df = pd.read_csv(data_path)

    logger.info(f"Loaded {len(df)} samples from {data_path}")

    # Prepare features using new engineering functions
    X, expected_columns = prepare_features_for_training(
        df,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

    # Get target
    y = df[target_column]

    # Save feature schema for inference
    if save_schema:
        schema_path = PROJECT_ROOT / "models" / "feature_schema.json"
        save_feature_schema(
            expected_columns=expected_columns,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            output_path=schema_path,
        )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test, expected_columns
