"""
FastAPI dependencies for dependency injection
"""

from functools import lru_cache
from typing import Dict

from src.models.inference import (
    load_production_model,
    load_schema,
    get_model_info as get_inference_model_info,
)
from src.data.feature_analysis import load_feature_statistics
from src.config import PROJECT_ROOT
from src.logging_config import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_model():
    """
    Dependency to get the loaded model

    Cached to load only once per application lifecycle.
    """
    logger.info("Loading model for API...")
    return load_production_model()


@lru_cache(maxsize=1)
def get_feature_schema() -> Dict:
    """
    Dependency to get feature schema

    Cached to load only once per application lifecycle.
    """
    logger.info("Loading feature schema for API...")
    return load_schema()


@lru_cache(maxsize=1)
def get_feature_stats() -> Dict | None:
    """
    Dependency to get feature statistics

    Cached to load only once per application lifecycle.
    Returns None if stats file doesn't exist.
    """
    try:
        stats_path = PROJECT_ROOT / "reports" / "feature_space_stats.json"
        logger.info("Loading feature statistics for API...")
        return load_feature_statistics(stats_path)
    except FileNotFoundError:
        logger.warning("Feature statistics not found. Range validation disabled.")
        return None


@lru_cache(maxsize=1)
def get_model_info() -> Dict:
    """
    Dependency to get model information

    Cached to load only once per application lifecycle.
    """
    logger.info("Loading model info for API...")
    return get_inference_model_info()
