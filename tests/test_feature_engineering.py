"""
Test feature engineering module

Verifies that training and inference use identical feature preprocessing.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.features.engineering import (
    prepare_features_for_training,
    prepare_features_for_inference,
    save_feature_schema,
    load_feature_schema,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.config import PROJECT_ROOT


@pytest.fixture
def sample_data():
    """Load sample data for testing"""
    data_path = PROJECT_ROOT / "data" / "processed" / "lpbf_enriched_v1.csv"
    return pd.read_csv(data_path)


@pytest.fixture
def training_features(sample_data):
    """Prepare training features from sample data"""
    X_train, expected_columns = prepare_features_for_training(
        sample_data.head(100),
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )
    return X_train, expected_columns


class TestFeatureEngineering:
    """Test suite for feature engineering"""

    def test_training_preparation_shape(self, training_features):
        """Test that training preparation returns correct shape"""
        X_train, expected_columns = training_features

        assert X_train.shape[0] == 100, "Should have 100 samples"
        assert len(expected_columns) > 0, "Should have feature columns"
        assert X_train.shape[1] == len(
            expected_columns
        ), "Columns should match expected"

    def test_training_preparation_columns(self, training_features):
        """Test that training preparation includes all expected columns"""
        X_train, expected_columns = training_features

        # Check numeric features are present
        for feature in NUMERIC_FEATURES:
            assert feature in X_train.columns, f"Missing numeric feature: {feature}"

        # Check one-hot encoded categorical features present
        assert any(
            col.startswith("Material_") for col in X_train.columns
        ), "Missing Material encoding"

    def test_inference_matches_training(self, sample_data, training_features):
        """Test that inference features match training exactly"""
        _, expected_columns = training_features

        # Prepare inference features
        X_inference = prepare_features_for_inference(
            sample_data.tail(10),
            expected_columns=expected_columns,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )

        # Verify exact column match
        assert (
            list(X_inference.columns) == expected_columns
        ), "Column order must match training"
        assert X_inference.shape[1] == len(expected_columns), "Feature count must match"

    def test_inference_handles_missing_columns(self, sample_data, training_features):
        """Test that inference adds missing one-hot encoded columns as zeros"""
        _, expected_columns = training_features

        # Create test sample with one category
        test_sample = sample_data.iloc[0:1].copy()

        X_test = prepare_features_for_inference(
            test_sample,
            expected_columns=expected_columns,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )

        # Should have all expected columns even if some categories are missing
        assert X_test.shape[1] == len(expected_columns), "Should add missing columns"
        assert list(X_test.columns) == expected_columns, "Column order should match"

    def test_no_missing_values(self, training_features):
        """Test that prepared features have no missing values"""
        X_train, _ = training_features

        assert (
            not X_train.isnull().any().any()
        ), "Training features should have no missing values"

    def test_numeric_features_are_numeric(self, training_features):
        """Test that numeric features have correct dtype"""
        X_train, _ = training_features

        for feature in NUMERIC_FEATURES:
            assert pd.api.types.is_numeric_dtype(
                X_train[feature]
            ), f"{feature} should be numeric"

    def test_categorical_one_hot_encoding(self, training_features):
        """Test that categorical features are properly one-hot encoded"""
        X_train, _ = training_features

        # Check that categorical features are one-hot encoded (binary)
        for feature in CATEGORICAL_FEATURES:
            encoded_cols = [
                col for col in X_train.columns if col.startswith(f"{feature}_")
            ]
            assert len(encoded_cols) > 0, f"No one-hot encoding found for {feature}"

            # Each encoded column should be binary (0 or 1)
            for col in encoded_cols:
                assert set(X_train[col].unique()).issubset(
                    {0, 1}
                ), f"{col} should be binary"


class TestFeatureSchema:
    """Test suite for feature schema saving/loading"""

    def test_save_and_load_schema(self, tmp_path, training_features):
        """Test saving and loading feature schema"""
        _, expected_columns = training_features

        schema_path = tmp_path / "test_schema.json"

        # Save schema
        save_feature_schema(
            expected_columns=expected_columns,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            output_path=schema_path,
        )

        assert schema_path.exists(), "Schema file should be created"

        # Load schema
        loaded_schema = load_feature_schema(schema_path)

        assert (
            loaded_schema["expected_columns"] == expected_columns
        ), "Columns should match"
        assert (
            loaded_schema["numeric_features"] == NUMERIC_FEATURES
        ), "Numeric features should match"
        assert (
            loaded_schema["categorical_features"] == CATEGORICAL_FEATURES
        ), "Categorical features should match"

    def test_schema_contains_correct_counts(self, tmp_path, training_features):
        """Test that schema contains correct feature counts"""
        _, expected_columns = training_features

        schema_path = tmp_path / "test_schema.json"

        save_feature_schema(
            expected_columns=expected_columns,
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
            output_path=schema_path,
        )

        schema = load_feature_schema(schema_path)

        assert schema["n_features"] == len(
            expected_columns
        ), "Total feature count should match"
        assert schema["n_numeric"] == len(
            NUMERIC_FEATURES
        ), "Numeric count should match"
        assert schema["n_categorical"] == len(
            CATEGORICAL_FEATURES
        ), "Categorical count should match"

    def test_production_schema_exists(self):
        """Test that production schema file exists (after training)"""
        schema_path = PROJECT_ROOT / "models" / "feature_schema.json"

        if schema_path.exists():
            schema = load_feature_schema(schema_path)

            assert "expected_columns" in schema, "Schema should have expected_columns"
            assert "n_features" in schema, "Schema should have n_features"
            assert len(schema["expected_columns"]) > 0, "Should have features"
        else:
            pytest.skip("Production schema not found - run training first")
