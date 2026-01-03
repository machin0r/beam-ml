"""
Export the Production model from MLflow registry to a standalone directory.

This script extracts only the Production model artifacts
"""

import json
import mlflow
import shutil
from datetime import datetime
from pathlib import Path

from src.config import PROJECT_ROOT, MLFLOW_TRACKING_URI
from src.logging_config import get_logger
from src.features.engineering import load_feature_schema

logger = get_logger(__name__)


def export_production_model(
    model_name: str = "lpbf_density_predictor",
    export_dir: Path = PROJECT_ROOT / "models" / "production",
) -> None:
    """
    Export the Production stage model to a standalone directory.

    Args:
        model_name: Name of the registered model
        export_dir: Directory to export the model to
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    logger.info(f"Exporting Production model: {model_name}")

    client = mlflow.MlflowClient()

    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])

        if not versions:
            raise ValueError(f"No Production version found for model '{model_name}'")

        production_version = versions[0]

        logger.info(f"Found Production version: {production_version.version}")
        logger.info(f"Run ID: {production_version.run_id}")

        export_dir.mkdir(parents=True, exist_ok=True)

        if (export_dir / "model").exists():
            logger.info("Removing existing exported model...")
            shutil.rmtree(export_dir / "model")

        model_uri = f"models:/{model_name}/Production"
        logger.info(f"Downloading model from {model_uri}")

        model_path = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri, dst_path=str(export_dir)
        )

        logger.info(f"Model exported to: {model_path}")

        # Load feature schema to include in metadata
        schema_path = PROJECT_ROOT / "models" / "feature_schema.json"
        schema = load_feature_schema(schema_path)

        metadata = {
            "model_name": model_name,
            "version": production_version.version,
            "run_id": production_version.run_id,
            "exported_at": datetime.now().isoformat(),
            "n_features": schema["n_features"],
            "n_numeric": schema["n_numeric"],
            "n_categorical": schema["n_categorical"],
        }

        metadata_path = export_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to: {metadata_path}")
        logger.info("=" * 70)
        logger.info("Production model exported successfully!")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Failed to export model: {e}")
        raise


if __name__ == "__main__":
    export_production_model()
