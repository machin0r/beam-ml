"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class PredictionRequest(BaseModel):
    """Request schema for density prediction"""

    # Process parameters
    laser_power: float = Field(
        ...,
        ge=50.0,
        le=500.0,
        description="Laser power in Watts",
        alias="laser_power_w",
    )
    scan_speed: float = Field(
        ...,
        ge=100.0,
        le=3000.0,
        description="Scan speed in mm/s",
        alias="scan_speed_mm_s",
    )
    hatch_space: float = Field(
        ...,
        ge=0.03,
        le=0.25,
        description="Hatch spacing in mm",
        alias="hatch_space_mm",
    )
    layer_thickness: float = Field(
        ...,
        ge=0.01,
        le=0.15,
        description="Layer thickness in mm",
        alias="layer_thickness_mm",
    )
    spot_size: float = Field(
        ...,
        ge=0.01,
        le=0.5,
        description="Spot size in mm",
        alias="spot_size_mm",
    )
    d50: float = Field(
        ...,
        ge=10.0,
        le=100.0,
        description="Powder particle size D50 in μm",
        alias="d50_um",
    )

    # Material properties
    melting_point: float = Field(
        ...,
        ge=273.0,
        le=4000.0,
        description="Melting point in Kelvin",
        alias="melting_point_k",
    )
    thermal_conductivity: float = Field(
        ...,
        ge=1.0,
        le=500.0,
        description="Thermal conductivity in W/mK",
        alias="thermal_conductivity_w_mk",
    )
    density: float = Field(
        ...,
        ge=1.0,
        le=25.0,
        description="Material density in g/cm³",
        alias="density_g_cm3",
    )
    specific_heat: float = Field(
        ...,
        ge=100.0,
        le=2000.0,
        description="Specific heat capacity in J/kgK",
        alias="specific_heat_j_kgk",
    )

    # Categorical features
    material: str = Field(
        ...,
        description="Material name (e.g., 316L, AlSi10Mg)",
    )
    measurement_method: str = Field(
        ...,
        description="Density measurement method",
        alias="measurement_method",
    )
    atmosphere: str = Field(
        ...,
        description="Build atmosphere (e.g., Argon, Nitrogen)",
    )
    printer_model: str = Field(
        ...,
        description="Printer model name",
        alias="printer_model",
    )

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Response schema for density prediction"""

    predicted_density: float = Field(
        ..., description="Predicted relative density in percentage"
    )
    quality_assessment: str = Field(
        ..., description="Quality assessment of the prediction"
    )
    material: str = Field(..., description="Material used for prediction")
    model_version: str = Field(..., description="Model version used")
    warnings: Optional[List[str]] = Field(
        default=None, description="Any warnings about the prediction"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_density": 98.5,
                "quality_assessment": "Excellent (Near full density)",
                "material": "316L",
                "model_version": "1",
                "warnings": None,
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check"""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Model name")
    model_version: Optional[str] = Field(None, description="Model version")


class ModelInfoResponse(BaseModel):
    """Response schema for model information"""

    model_name: str
    version: str
    stage: str
    description: Optional[str] = None
    n_features: int
    numeric_features: int
    categorical_features: int


class FeatureRangesResponse(BaseModel):
    """Response schema for valid feature ranges"""

    feature: str
    min: float
    max: float
    mean: float
    p5: float
    p95: float
    unit: Optional[str] = None


class ErrorResponse(BaseModel):
    """Response schema for errors"""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
