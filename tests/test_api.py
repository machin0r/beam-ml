"""
Test FastAPI application

Tests all API endpoints using FastAPI TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    """Create a test client"""
    return TestClient(app)


@pytest.fixture
def valid_prediction_request():
    """Valid prediction request payload"""
    return {
        "laser_power_w": 200.0,
        "scan_speed_mm_s": 800.0,
        "hatch_space_mm": 0.1,
        "layer_thickness_mm": 0.03,
        "spot_size_mm": 0.08,
        "d50_um": 45.0,
        "melting_point_k": 1673.0,
        "thermal_conductivity_w_mk": 16.3,
        "density_g_cm3": 7.99,
        "specific_heat_j_kgk": 500.0,
        "material": "316L",
        "measurement_method": "Archimedes",
        "atmosphere": "Argon",
        "printer_model": "EOS M290",
    }


class TestRootEndpoint:
    """Test suite for root endpoint"""

    def test_root_endpoint(self, client):
        """Test root endpoint returns welcome message"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data


class TestHealthEndpoint:
    """Test suite for health check endpoint"""

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert data["model_loaded"] is True

    def test_health_check_includes_model_info(self, client):
        """Test health check includes model metadata"""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()

        assert "model_name" in data
        assert "model_version" in data


class TestModelInfoEndpoint:
    """Test suite for model info endpoint"""

    def test_model_info(self, client):
        """Test model info endpoint"""
        response = client.get("/api/v1/model-info")

        assert response.status_code == 200
        data = response.json()

        required_fields = ["model_name", "stage", "n_features"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_model_info_feature_counts(self, client):
        """Test model info includes feature counts"""
        response = client.get("/api/v1/model-info")

        assert response.status_code == 200
        data = response.json()

        assert data["n_features"] > 0, "Should have features"
        if "numeric_features" in data and "categorical_features" in data:
            assert data["numeric_features"] > 0, "Should have numeric features"
            assert data["categorical_features"] > 0, "Should have categorical features"


class TestPredictionEndpoint:
    """Test suite for prediction endpoint"""

    def test_predict_success(self, client, valid_prediction_request):
        """Test successful prediction"""
        response = client.post("/api/v1/predict", json=valid_prediction_request)

        assert response.status_code == 200
        data = response.json()

        assert "predicted_density" in data
        assert "quality_assessment" in data
        assert "material" in data
        assert "model_version" in data

    def test_predict_density_in_range(self, client, valid_prediction_request):
        """Test prediction is in valid range"""
        response = client.post("/api/v1/predict", json=valid_prediction_request)

        assert response.status_code == 200
        data = response.json()

        density = data["predicted_density"]
        assert 0 <= density <= 100, f"Density {density}% should be in [0, 100]"

    def test_predict_material_matches_request(self, client, valid_prediction_request):
        """Test response material matches request"""
        response = client.post("/api/v1/predict", json=valid_prediction_request)

        assert response.status_code == 200
        data = response.json()

        assert data["material"] == valid_prediction_request["material"]

    def test_predict_missing_field_returns_422(self, client, valid_prediction_request):
        """Test prediction with missing field returns validation error"""
        invalid_request = valid_prediction_request.copy()
        del invalid_request["laser_power_w"]

        response = client.post("/api/v1/predict", json=invalid_request)

        assert response.status_code == 422  # Unprocessable Entity (Pydantic validation)
        data = response.json()
        assert "detail" in data

    def test_predict_invalid_value_returns_422(self, client, valid_prediction_request):
        """Test prediction with out-of-range value returns validation error"""
        invalid_request = valid_prediction_request.copy()
        invalid_request["laser_power_w"] = -100  # Negative power (invalid)

        response = client.post("/api/v1/predict", json=invalid_request)

        assert response.status_code == 422

    def test_predict_wrong_type_returns_422(self, client, valid_prediction_request):
        """Test prediction with wrong data type returns validation error"""
        invalid_request = valid_prediction_request.copy()
        invalid_request["laser_power_w"] = "not a number"

        response = client.post("/api/v1/predict", json=invalid_request)

        assert response.status_code == 422

    def test_predict_different_materials(self, client, valid_prediction_request):
        """Test predictions for different materials"""
        # Test 316L (steel)
        steel_request = valid_prediction_request.copy()
        steel_response = client.post("/api/v1/predict", json=steel_request)
        assert steel_response.status_code == 200
        steel_density = steel_response.json()["predicted_density"]

        # Test AlSi10Mg (aluminium)
        aluminium_request = valid_prediction_request.copy()
        aluminium_request.update(
            {
                "material": "AlSi10Mg",
                "melting_point_k": 848.0,
                "thermal_conductivity_w_mk": 173.0,
                "density_g_cm3": 2.67,
                "specific_heat_j_kgk": 897.0,
            }
        )
        aluminium_response = client.post("/api/v1/predict", json=aluminium_request)
        assert aluminium_response.status_code == 200
        aluminium_density = aluminium_response.json()["predicted_density"]

        # Predictions should differ
        assert (
            steel_density != aluminium_density
        ), "Different materials should produce different predictions"


class TestFeatureRangesEndpoint:
    """Test suite for feature ranges endpoint"""

    def test_feature_ranges_endpoint_exists(self, client):
        """Test that feature ranges endpoint exists"""
        response = client.get("/api/v1/feature-ranges")

        # May return 200 with data or 404 if not available
        assert response.status_code in [200, 404]

    def test_feature_ranges_structure(self, client):
        """Test feature ranges response structure if available"""
        response = client.get("/api/v1/feature-ranges")

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list), "Should return list of feature ranges"

            if len(data) > 0:
                # Check first item structure
                first_range = data[0]
                assert "feature" in first_range
                assert "min" in first_range
                assert "max" in first_range
                assert "unit" in first_range


class TestAPIDocumentation:
    """Test suite for API documentation endpoints"""

    def test_openapi_schema(self, client):
        """Test OpenAPI schema is available"""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()

        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema

    def test_swagger_ui(self, client):
        """Test Swagger UI is available"""
        response = client.get("/docs")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_redoc_ui(self, client):
        """Test ReDoc UI is available"""
        response = client.get("/redoc")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestCORS:
    """Test suite for CORS configuration"""

    def test_cors_headers_present(self, client):
        """Test CORS headers are present in responses"""
        response = client.options(
            "/api/v1/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

        # Should allow CORS
        assert "access-control-allow-origin" in response.headers


class TestErrorHandling:
    """Test suite for error handling"""

    def test_404_on_invalid_endpoint(self, client):
        """Test 404 for non-existent endpoint"""
        response = client.get("/api/v1/nonexistent")

        assert response.status_code == 404

    def test_405_on_wrong_method(self, client):
        """Test 405 for wrong HTTP method"""
        response = client.get("/api/v1/predict")  # Should be POST

        assert response.status_code == 405  # Method Not Allowed


class TestPredictionQuality:
    """Test suite for prediction quality"""

    def test_high_quality_parameters(self, client):
        """Test prediction with known high-quality parameters"""
        high_quality_request = {
            "laser_power_w": 195.0,
            "scan_speed_mm_s": 1083.0,
            "hatch_space_mm": 0.12,
            "layer_thickness_mm": 0.03,
            "spot_size_mm": 0.08,
            "d50_um": 45.0,
            "melting_point_k": 1933.0,
            "thermal_conductivity_w_mk": 21.9,
            "density_g_cm3": 8.19,
            "specific_heat_j_kgk": 444.0,
            "material": "316L",
            "measurement_method": "Archimedes",
            "atmosphere": "Argon",
            "printer_model": "EOS M290",
        }

        response = client.post("/api/v1/predict", json=high_quality_request)

        assert response.status_code == 200
        data = response.json()

        assert (
            data["predicted_density"] > 90
        ), f"Expected >90%, got {data['predicted_density']}%"
        assert data["quality_assessment"] in [
            "Excellent (Near full density)",
            "Good (High density)",
            "Acceptable (Moderate density)",
        ], "Should have above acceptable quality assessment"

    def test_prediction_consistency(self, client, valid_prediction_request):
        """Test same input produces same prediction"""
        response1 = client.post("/api/v1/predict", json=valid_prediction_request)
        response2 = client.post("/api/v1/predict", json=valid_prediction_request)

        assert response1.status_code == 200
        assert response2.status_code == 200

        density1 = response1.json()["predicted_density"]
        density2 = response2.json()["predicted_density"]

        assert density1 == density2, "Same input should give same prediction"
