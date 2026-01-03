"""
Inference module for LPBF density prediction
"""

import mlflow
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

from src.features.engineering import (
    prepare_features_for_inference,
    load_feature_schema,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.config import MLFLOW_TRACKING_URI, PROJECT_ROOT
from src.logging_config import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_production_model():
    """
    Load the production model from MLflow Model Registry

    Uses LRU cache to load model only once per process.

    Returns:
        Loaded MLflow model
    """
    logger.info("Loading production model from MLflow...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        # Load model tagged as "Production"
        model_uri = "models:/lpbf_density_predictor/Production"
        model = mlflow.pyfunc.load_model(model_uri)

        logger.info(f"Production model loaded successfully")
        logger.info(f"  Model URI: {model_uri}")

        return model

    except Exception as e:
        logger.error(f"Failed to load production model: {e}")
        raise RuntimeError(
            f"Could not load production model. "
            f"Ensure a model is registered and promoted to Production stage. "
            f"Error: {e}"
        )


@lru_cache(maxsize=1)
def load_schema() -> Dict:
    """
    Load feature schema

    Uses LRU cache to load schema only once per process.

    Returns:
        Feature schema dictionary
    """
    schema_path = PROJECT_ROOT / "models" / "feature_schema.json"
    return load_feature_schema(schema_path)


def preprocess_input(
    input_data: Dict[str, float],
    expected_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Preprocess raw input for inference

    Args:
        input_data: Dictionary with feature names and values
        expected_columns: Expected column names (loaded from schema if not provided)

    Returns:
        Preprocessed DataFrame ready for model prediction
    """
    # Load schema if columns not provided
    if expected_columns is None:
        schema = load_schema()
        expected_columns = schema["expected_columns"]

    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    # Apply same preprocessing as training
    X = prepare_features_for_inference(
        df,
        expected_columns=expected_columns,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

    return X


def predict_density(input_data: Dict[str, float]) -> Tuple[float, Dict]:
    """
    Predict relative density for given input parameters

    Args:
        input_data: Dictionary with all required features:
            - Laser Power (W)
            - Scan Speed (mm/s)
            - Hatch space (mm)
            - Layer thickness (mm)
            - Spot size (mm)
            - D50 μm
            - Melting Point (K)
            - Thermal Conductivity (W/mK)
            - Density (g/cm^3)
            - Specific Heat Capacity (J/kgK)
            - Material
            - Density Measurement Method
            - Atmosphere
            - Printer Model

    Returns:
        Tuple of (prediction, metadata)
            - prediction: Predicted relative density (%)
            - metadata: Additional information about prediction
    """
    logger.debug(
        f"Making prediction for material: {input_data.get('Material', 'Unknown')}"
    )

    try:
        # Load model and schema
        model = load_production_model()
        schema = load_schema()

        # Preprocess input
        X = preprocess_input(input_data, expected_columns=schema["expected_columns"])

        # Make prediction
        prediction = model.predict(X)[0]

        # Create metadata
        metadata = {
            "model_name": "lpbf_density_predictor",
            "model_stage": "Production",
            "n_features": len(X.columns),
            "material": input_data.get("Material", "Unknown"),
        }

        logger.info(f"Prediction: {prediction:.2f}% for {metadata['material']}")

        return float(prediction), metadata

    except KeyError as e:
        logger.error(f"Missing required feature: {e}")
        raise ValueError(f"Missing required feature: {e}") from e

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise RuntimeError(f"Prediction failed: {e}") from e


def batch_predict(input_list: List[Dict[str, float]]) -> List[Tuple[float, Dict]]:
    """
    Make predictions for multiple inputs

    Args:
        input_list: List of input dictionaries

    Returns:
        List of (prediction, metadata) tuples
    """
    logger.info(f"Making batch predictions for {len(input_list)} samples")

    results = []
    for input_data in input_list:
        try:
            prediction, metadata = predict_density(input_data)
            results.append((prediction, metadata))
        except Exception as e:
            logger.warning(f"Batch prediction failed for one sample: {e}")
            results.append((None, {"error": str(e)}))

    successful = sum(1 for p, _ in results if p is not None)
    logger.info(f"Batch prediction complete: {successful}/{len(input_list)} successful")

    return results


def validate_input(input_data: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Validate input data for required features and types

    Args:
        input_data: Input dictionary to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check required numeric features
    for feature in NUMERIC_FEATURES:
        if feature not in input_data:
            errors.append(f"Missing required feature: {feature}")
        elif not isinstance(input_data[feature], (int, float)):
            errors.append(f"{feature} must be numeric, got {type(input_data[feature])}")

    # Check required categorical features
    for feature in CATEGORICAL_FEATURES:
        if feature not in input_data:
            errors.append(f"Missing required feature: {feature}")
        elif not isinstance(input_data[feature], str):
            errors.append(f"{feature} must be string, got {type(input_data[feature])}")

    is_valid = len(errors) == 0

    if is_valid:
        logger.debug("Input validation passed")
    else:
        logger.warning(f"Input validation failed: {len(errors)} errors")

    return is_valid, errors


def get_model_info() -> Dict:
    """
    Get information about the loaded model

    Returns:
        Dictionary with model metadata
    """
    try:
        model = load_production_model()
        schema = load_schema()

        # Get model version info from MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        versions = client.get_latest_versions(
            "lpbf_density_predictor", stages=["Production"]
        )

        if versions:
            version_info = versions[0]
            info = {
                "model_name": "lpbf_density_predictor",
                "version": version_info.version,
                "stage": version_info.current_stage,
                "description": version_info.description,
                "n_features": schema["n_features"],
                "numeric_features": len(schema["numeric_features"]),
                "categorical_features": len(schema["categorical_features"]),
            }
        else:
            info = {
                "model_name": "lpbf_density_predictor",
                "stage": "Production",
                "n_features": schema["n_features"],
            }

        return info

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return {"error": str(e)}
