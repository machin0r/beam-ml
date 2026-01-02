#!/usr/bin/env python3
"""
Register the best model to MLflow Model Registry
"""

import mlflow
from mlflow.tracking import MlflowClient
from src.logging_config import get_logger


logger = get_logger(__name__)


def register_best_model():
    """Find and register the best performing model"""

    mlflow.set_tracking_uri("file:./mlruns")
    client = MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name("lpbf_density_prediction")

    # Get best run by R²
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.r2 DESC"],
        max_results=1,
    )

    best_run = runs[0]

    logger.info("=" * 70)
    logger.info("Best Model Found".center(70))
    logger.info("=" * 70)
    logger.info(f"Run Name: {best_run.data.tags.get('mlflow.runName')}")
    logger.info(f"RMSE: {best_run.data.metrics['rmse']:.3f}")
    logger.info(f"R²: {best_run.data.metrics['r2']:.3f}")
    logger.info(f"MAE: {best_run.data.metrics['mae']:.3f}")
    logger.info(f"Run ID: {best_run.info.run_id}")

    # Register model
    model_uri = f"runs:/{best_run.info.run_id}/model"

    model_version = mlflow.register_model(
        model_uri=model_uri, name="lpbf_density_predictor"
    )

    logger.info("\n" + "=" * 70)
    logger.info(
        f"Registered as 'lpbf_density_predictor' version {model_version.version}"
    )

    # Add description
    client.update_model_version(
        name="lpbf_density_predictor",
        version=model_version.version,
        description=f"""
        Model for L-PBF relative density prediction.

        Performance:
        - RMSE: {best_run.data.metrics['rmse']:.3f}%
        - R²: {best_run.data.metrics['r2']:.3f}
        - MAE: {best_run.data.metrics['mae']:.3f}%
        """,
    )

    logger.info("Added model description")

    # Promote to Production stage
    client.transition_model_version_stage(
        name="lpbf_density_predictor",
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True,  # Archive old production models
    )

    logger.info("Model promoted to Production stage!")
    logger.info("Inference will now use this model automatically")
    logger.info("=" * 70)


if __name__ == "__main__":
    register_best_model()
