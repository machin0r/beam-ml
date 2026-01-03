"""
API routes for LPBF density prediction
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import List

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    FeatureRangesResponse,
    ErrorResponse,
)
from api.dependencies import (
    get_model,
    get_feature_schema,
    get_feature_stats,
    get_model_info,
)
from src.models.inference import predict_density, validate_input
from src.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Predict relative density",
    description="Predict relative density for L-PBF parts based on process parameters and material properties",
)
async def predict(
    request: PredictionRequest,
    model=Depends(get_model),
    schema=Depends(get_feature_schema),
) -> PredictionResponse:
    """
    Make a density prediction based on input parameters
    """
    logger.info(f"Prediction request for material: {request.material}")

    try:
        input_data = {
            "Laser Power (W)": request.laser_power,
            "Scan Speed (mm/s)": request.scan_speed,
            "Hatch space (mm)": request.hatch_space,
            "Layer thickness (mm)": request.layer_thickness,
            "Spot size (mm)": request.spot_size,
            "D50 μm": request.d50,
            "Melting Point (K)": request.melting_point,
            "Thermal Conductivity (W/mK)": request.thermal_conductivity,
            "Density (g/cm^3)": request.density,
            "Specific Heat Capacity (J/kgK)": request.specific_heat,
            "Material": request.material,
            "Density Measurement Method": request.measurement_method,
            "Atmosphere": request.atmosphere,
            "Printer Model": request.printer_model,
        }

        is_valid, errors = validate_input(input_data)
        if not is_valid:
            logger.warning(f"Invalid input: {errors}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Invalid input", "details": errors},
            )

        prediction, metadata = predict_density(input_data)

        if prediction >= 99.5:
            quality = "Excellent (Near full density)"
        elif prediction >= 97:
            quality = "Good (High density)"
        elif prediction >= 95:
            quality = "Acceptable (Moderate density)"
        else:
            quality = "Poor (Low density)"

        warnings = None

        response = PredictionResponse(
            predicted_density=round(prediction, 2),
            quality_assessment=quality,
            material=request.material,
            model_version=metadata.get("model_version", "1"),
            warnings=warnings,
        )

        logger.info(f"Prediction successful: {prediction:.2f}% for {request.material}")

        return response

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Validation error", "details": str(e)},
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Prediction failed", "details": str(e)},
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API and model are working correctly",
)
async def health_check(model_info=Depends(get_model_info)) -> HealthResponse:
    """
    Health check endpoint
    """
    try:
        model_loaded = "error" not in model_info

        if model_loaded:
            return HealthResponse(
                status="healthy",
                model_loaded=True,
                model_name=model_info.get("model_name"),
                model_version=str(model_info.get("version")),
            )
        else:
            return HealthResponse(
                status="unhealthy",
                model_loaded=False,
                model_name=None,
                model_version=None,
            )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_name=None,
            model_version=None,
        )


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Get model information",
    description="Get information about the currently loaded model",
)
async def model_info(info=Depends(get_model_info)) -> ModelInfoResponse:
    """
    Get information about the loaded model
    """
    try:
        if "error" in info:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Failed to get model info", "details": info["error"]},
            )

        return ModelInfoResponse(
            model_name=info.get("model_name", "unknown"),
            version=str(info.get("version", "unknown")),
            stage=info.get("stage", "unknown"),
            description=info.get("description"),
            n_features=info.get("n_features", 0),
            numeric_features=info.get("numeric_features", 0),
            categorical_features=info.get("categorical_features", 0),
        )

    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to get model info", "details": str(e)},
        )


@router.get(
    "/feature-ranges",
    response_model=List[FeatureRangesResponse],
    summary="Get valid feature ranges",
    description="Get the valid ranges for input features based on training data",
)
async def feature_ranges(
    stats=Depends(get_feature_stats),
) -> List[FeatureRangesResponse]:
    """
    Get valid feature ranges from training data
    """
    try:
        if stats is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "Feature statistics not available",
                    "details": "Run feature analysis script to generate statistics",
                },
            )

        ranges = []
        for feature, feature_stats in stats["numeric"].items():
            # Extract unit from feature name if present
            unit = None
            if "(" in feature and ")" in feature:
                unit = feature.split("(")[1].split(")")[0]

            ranges.append(
                FeatureRangesResponse(
                    feature=feature,
                    min=feature_stats["min"],
                    max=feature_stats["max"],
                    mean=feature_stats["mean"],
                    p5=feature_stats["p5"],
                    p95=feature_stats["p95"],
                    unit=unit,
                )
            )

        logger.info(f"Returned feature ranges for {len(ranges)} features")

        return ranges

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get feature ranges: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to get feature ranges", "details": str(e)},
        )
