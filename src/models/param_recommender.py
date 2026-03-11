"""
Inference module for the LPBF parameter recommender.

Loads 8 pre-trained quantile XGBoost models (4 targets × q0.10/q0.90)
and returns recommended process parameter ranges for a given material
and target density.
"""

import json
import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.features.engineering import prepare_features_for_inference
from src.config import PROJECT_ROOT
from src.logging_config import get_logger

logger = get_logger(__name__)

TARGET_CLEAN_NAMES = {
    "Laser Power (W)": "laser_power_w",
    "Scan Speed (mm/s)": "scan_speed_mm_s",
    "Hatch space (mm)": "hatch_space_mm",
    "Layer thickness (mm)": "layer_thickness_mm",
}


def _get_recommender_dir() -> Path:
    path = os.getenv("RECOMMENDER_MODEL_PATH")
    if path:
        return Path(path)
    return PROJECT_ROOT / "models" / "param_recommender"


@lru_cache(maxsize=1)
def load_recommender_models() -> Dict:
    """
    Load all 8 quantile models and the schema from disk.

    Returns a dict with keys:
        "schema": dict
        "models": { "laser_power_w_q10": XGBRegressor, ... }
    """
    recommender_dir = _get_recommender_dir()
    schema_path = recommender_dir / "schema.json"

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Recommender schema not found at {schema_path}. "
            "Run scripts/train_param_recommender.py first."
        )

    with open(schema_path) as f:
        schema = json.load(f)

    models: Dict = {}
    for target_col, clean_name in TARGET_CLEAN_NAMES.items():
        for q in schema["quantiles"]:
            q_label = f"q{int(q * 100):02d}"
            key = f"{clean_name}_{q_label}"
            model_path = recommender_dir / f"{clean_name}_{q_label}.pkl"

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Recommender model not found: {model_path}. "
                    "Run scripts/train_param_recommender.py first."
                )

            with open(model_path, "rb") as f:
                models[key] = pickle.load(f)

    logger.info(f"Loaded {len(models)} recommender models from {recommender_dir}")
    return {"schema": schema, "models": models}


def predict_parameter_ranges(input_data: Dict) -> Dict[str, Dict[str, float]]:
    """
    Predict recommended process parameter ranges.

    Args:
        input_data: dict with keys matching RECOMMENDER_NUMERIC_FEATURES +
                    RECOMMENDER_CATEGORICAL_FEATURES, e.g.:
                    {
                        "Melting Point (K)": 1673.0,
                        "Thermal Conductivity (W/mK)": 16.3,
                        "Density (g/cm^3)": 7.99,
                        "Specific Heat Capacity (J/kgK)": 500.0,
                        "D50 μm": 45.0,
                        "RD (%)": 99.0,
                        "Material": "316L",
                        "Atmosphere": "Argon",
                        "Printer Model": "EOS M290",
                    }

    Returns:
        dict with parameter ranges, e.g.:
        {
            "laser_power_w": {"min": 150.0, "max": 260.0},
            "scan_speed_mm_s": {"min": 600.0, "max": 1200.0},
            ...
        }
    """
    bundle = load_recommender_models()
    schema = bundle["schema"]
    models = bundle["models"]

    df = pd.DataFrame([input_data])

    X = prepare_features_for_inference(
        df,
        expected_columns=schema["expected_columns"],
        numeric_features=schema["numeric_features"],
        categorical_features=schema["categorical_features"],
    )

    result: Dict[str, Dict[str, float]] = {}

    for target_col, clean_name in TARGET_CLEAN_NAMES.items():
        quantile_values: List[float] = []
        for q in schema["quantiles"]:
            q_label = f"q{int(q * 100):02d}"
            key = f"{clean_name}_{q_label}"
            pred = float(models[key].predict(X)[0])
            quantile_values.append(pred)

        low, high = min(quantile_values), max(quantile_values)
        result[clean_name] = {"min": round(low, 4), "max": round(high, 4)}

    logger.info(f"Parameter ranges predicted for material={input_data.get('Material')}")
    return result
