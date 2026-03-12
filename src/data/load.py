"""
Data loading utilities for training
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple, List

from src.features.engineering import (
    compute_derived_features,
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
    min_density: float = 90.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Load and prepare training data

    Args:
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        target_column: Name of target column
        save_schema: Whether to save feature schema for inference
        min_density: Minimum relative density (%) to include. Filters out
            failed/extreme-parameter samples outside the optimisation use case.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, expected_columns)
    """
    logger.info("Loading training data...")

    # Load raw data
    data_path = PROJECT_ROOT / "data" / "processed" / "lpbf_enriched_v1.csv"
    df = pd.read_csv(data_path)

    logger.info(f"Loaded {len(df)} samples from {data_path}")

    # Filter to practical operating range
    if min_density is not None:
        before = len(df)
        df = df[df[target_column] >= min_density].copy()
        logger.info(
            f"Filtered to >= {min_density}% density: {len(df)} samples "
            f"({before - len(df)} removed)"
        )

    # Add derived features
    df = compute_derived_features(df)

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

    # DOI-aware split: keep all rows from the same study in the same split
    doi_groups = df["DOI"]
    n_dois = doi_groups.nunique()
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=doi_groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    n_train_dois = doi_groups.iloc[train_idx].nunique()
    n_test_dois = doi_groups.iloc[test_idx].nunique()
    logger.info(f"DOI-aware split: {n_dois} studies total")
    logger.info(f"Training set: {len(X_train)} samples ({n_train_dois} studies)")
    logger.info(f"Test set: {len(X_test)} samples ({n_test_dois} studies)")

    return X_train, X_test, y_train, y_test, expected_columns
