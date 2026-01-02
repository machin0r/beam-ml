from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# MLflow
MLFLOW_TRACKING_URI = "file:./mlruns"
DENSITY_EXPERIMENT_NAME = "lpbf_density_prediction"

# Target
TARGET_COLUMN = "RD (%)"
