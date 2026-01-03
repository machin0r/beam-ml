"""
Test inference module

Tests model loading, feature preprocessing, and predictions.
"""

import pytest

from src.models.inference import (
    load_production_model,
    load_schema,
    predict_density,
    validate_input,
    get_model_info,
    batch_predict,
)


@pytest.fixture
def valid_input():
    """Valid input data for predictions"""
    return {
        "Laser Power (W)": 200.0,
        "Scan Speed (mm/s)": 800.0,
        "Hatch space (mm)": 0.1,
        "Layer thickness (mm)": 0.03,
        "Spot size (mm)": 0.08,
        "D50 μm": 45.0,
        "Melting Point (K)": 1673.0,
        "Thermal Conductivity (W/mK)": 16.3,
        "Density (g/cm^3)": 7.99,
        "Specific Heat Capacity (J/kgK)": 500.0,
        "Material": "316L",
        "Density Measurement Method": "Archimedes",
        "Atmosphere": "Argon",
        "Printer Model": "EOS M290",
    }


@pytest.fixture
def aluminium_input():
    """Valid input data for AlSi10Mg material"""
    return {
        "Laser Power (W)": 370.0,
        "Scan Speed (mm/s)": 1300.0,
        "Hatch space (mm)": 0.19,
        "Layer thickness (mm)": 0.03,
        "Spot size (mm)": 0.08,
        "D50 μm": 45.0,
        "Melting Point (K)": 848.0,
        "Thermal Conductivity (W/mK)": 173.0,
        "Density (g/cm^3)": 2.67,
        "Specific Heat Capacity (J/kgK)": 897.0,
        "Material": "AlSi10Mg",
        "Density Measurement Method": "Archimedes",
        "Atmosphere": "Argon",
        "Printer Model": "EOS M290",
    }


class TestModelLoading:
    """Test suite for model and schema loading"""

    def test_load_production_model(self):
        """Test loading production model from MLflow registry"""
        model = load_production_model()

        assert model is not None, "Model should be loaded"
        # Model should be cached on second call
        model2 = load_production_model()
        assert model is model2, "Model should be cached"

    def test_load_schema(self):
        """Test loading feature schema"""
        schema = load_schema()

        assert "expected_columns" in schema, "Schema should have expected_columns"
        assert "n_features" in schema, "Schema should have n_features"
        assert schema["n_features"] > 0, "Should have features"

        # Schema should be cached
        schema2 = load_schema()
        assert schema is schema2, "Schema should be cached"

    def test_get_model_info(self):
        """Test retrieving model information"""
        info = get_model_info()

        assert "model_name" in info, "Should have model name"
        assert "n_features" in info, "Should have feature count"
        assert info["n_features"] > 0, "Should have features"


class TestInputValidation:
    """Test suite for input validation"""

    def test_validate_valid_input(self, valid_input):
        """Test validation passes for valid input"""
        is_valid, errors = validate_input(valid_input)

        assert is_valid, "Valid input should pass validation"
        assert len(errors) == 0, "Should have no errors"

    def test_validate_missing_numeric_feature(self, valid_input):
        """Test validation fails when numeric feature is missing"""
        invalid_input = valid_input.copy()
        del invalid_input["Laser Power (W)"]

        is_valid, errors = validate_input(invalid_input)

        assert not is_valid, "Should fail validation"
        assert len(errors) > 0, "Should have errors"
        assert any(
            "Laser Power" in error for error in errors
        ), "Should mention missing feature"

    def test_validate_missing_categorical_feature(self, valid_input):
        """Test validation fails when categorical feature is missing"""
        invalid_input = valid_input.copy()
        del invalid_input["Material"]

        is_valid, errors = validate_input(invalid_input)

        assert not is_valid, "Should fail validation"
        assert len(errors) > 0, "Should have errors"
        assert any(
            "Material" in error for error in errors
        ), "Should mention missing feature"

    def test_validate_wrong_numeric_type(self, valid_input):
        """Test validation fails when numeric feature has wrong type"""
        invalid_input = valid_input.copy()
        invalid_input["Laser Power (W)"] = "not a number"

        is_valid, errors = validate_input(invalid_input)

        assert not is_valid, "Should fail validation"
        assert len(errors) > 0, "Should have errors"

    def test_validate_wrong_categorical_type(self, valid_input):
        """Test validation fails when categorical feature has wrong type"""
        invalid_input = valid_input.copy()
        invalid_input["Material"] = 123  # Should be string

        is_valid, errors = validate_input(invalid_input)

        assert not is_valid, "Should fail validation"
        assert len(errors) > 0, "Should have errors"


class TestPrediction:
    """Test suite for making predictions"""

    def test_predict_density_valid_input(self, valid_input):
        """Test prediction with valid input"""
        prediction, metadata = predict_density(valid_input)

        assert isinstance(prediction, float), "Prediction should be float"
        assert 0 <= prediction <= 100, "Prediction should be in [0, 100]% range"
        assert "model_name" in metadata, "Metadata should have model name"
        assert "material" in metadata, "Metadata should have material"
        assert metadata["material"] == "316L", "Material should match input"

    def test_predict_density_aluminium(self, aluminium_input):
        """Test prediction with different material"""
        prediction, metadata = predict_density(aluminium_input)

        assert isinstance(prediction, float), "Prediction should be float"
        assert 0 <= prediction <= 100, "Prediction should be in [0, 100]% range"
        assert metadata["material"] == "AlSi10Mg", "Material should match input"

    def test_predict_density_missing_feature_raises_error(self, valid_input):
        """Test that prediction fails with missing feature"""
        invalid_input = valid_input.copy()
        del invalid_input["Laser Power (W)"]

        with pytest.raises((ValueError, KeyError)):
            predict_density(invalid_input)

    def test_prediction_consistency(self, valid_input):
        """Test that same input produces same prediction"""
        prediction1, _ = predict_density(valid_input)
        prediction2, _ = predict_density(valid_input)

        assert prediction1 == prediction2, "Same input should give same prediction"

    def test_prediction_metadata_completeness(self, valid_input):
        """Test that prediction metadata is complete"""
        _, metadata = predict_density(valid_input)

        required_keys = ["model_name", "model_stage", "n_features", "material"]
        for key in required_keys:
            assert key in metadata, f"Metadata should have {key}"


class TestBatchPrediction:
    """Test suite for batch predictions"""

    def test_batch_predict_single(self, valid_input):
        """Test batch prediction with single input"""
        results = batch_predict([valid_input])

        assert len(results) == 1, "Should have one result"
        prediction, metadata = results[0]
        assert isinstance(prediction, float), "Prediction should be float"
        assert 0 <= prediction <= 100, "Prediction should be in range"

    def test_batch_predict_multiple(self, valid_input, aluminium_input):
        """Test batch prediction with multiple inputs"""
        results = batch_predict([valid_input, aluminium_input])

        assert len(results) == 2, "Should have two results"

        for prediction, metadata in results:
            assert isinstance(prediction, float), "Prediction should be float"
            assert 0 <= prediction <= 100, "Prediction should be in range"

    def test_batch_predict_with_invalid_input(self, valid_input):
        """Test that batch prediction handles invalid input gracefully"""
        invalid_input = valid_input.copy()
        del invalid_input["Laser Power (W)"]

        results = batch_predict([valid_input, invalid_input])

        assert len(results) == 2, "Should have two results"

        # First should succeed
        prediction1, metadata1 = results[0]
        assert prediction1 is not None, "First prediction should succeed"

        # Second should fail gracefully
        prediction2, metadata2 = results[1]
        assert prediction2 is None, "Second prediction should be None"
        assert "error" in metadata2, "Should have error in metadata"


class TestPredictionQuality:
    """Test suite for prediction quality and ranges"""

    def test_high_quality_parameters_prediction(self):
        """Test prediction with known high-quality parameters"""
        # Parameters that should produce high density
        high_quality_input = {
            "Laser Power (W)": 195.0,
            "Scan Speed (mm/s)": 1083.0,
            "Hatch space (mm)": 0.12,
            "Layer thickness (mm)": 0.03,
            "Spot size (mm)": 0.08,
            "D50 μm": 45.0,
            "Melting Point (K)": 1933.0,
            "Thermal Conductivity (W/mK)": 21.9,
            "Density (g/cm^3)": 8.19,
            "Specific Heat Capacity (J/kgK)": 444.0,
            "Material": "316L",
            "Density Measurement Method": "Archimedes",
            "Atmosphere": "Argon",
            "Printer Model": "EOS M290",
        }

        prediction, _ = predict_density(high_quality_input)

        # Should predict reasonable density (not testing exact value as model may vary)
        assert (
            prediction > 90
        ), f"High quality params should predict >90%, got {prediction}%"

    def test_predictions_differ_by_material(self, valid_input, aluminium_input):
        """Test that different materials produce different predictions"""
        pred_steel, _ = predict_density(valid_input)
        pred_aluminium, _ = predict_density(aluminium_input)

        # Predictions should differ (materials have different properties)
        assert (
            pred_steel != pred_aluminium
        ), "Different materials should give different predictions"
