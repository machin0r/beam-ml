from dataclasses import dataclass, asdict
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import matplotlib
import matplotlib.pyplot as plt

from src.logging_config import get_logger

matplotlib.use("Agg")

logger = get_logger(__name__)


@dataclass
class Metrics:
    rmse: float
    r2: float
    mae: float


def train_and_log_model(
    model: BaseEstimator,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[BaseEstimator, Metrics]:
    """Train model and log everything to MLflow"""

    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_params(model.get_params())

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        metrics = Metrics(
            rmse=root_mean_squared_error(y_test, y_pred),
            r2=r2_score(y_test, y_pred),
            mae=mean_absolute_error(y_test, y_pred),
        )

        mlflow.log_metrics(asdict(metrics))
        mlflow.sklearn.log_model(model, name="model")

        # Log feature importances if available
        if hasattr(model, "feature_importances_"):

            fig, ax = plt.subplots(figsize=(10, 6))
            importances = pd.Series(
                model.feature_importances_, index=X_train.columns
            ).sort_values(ascending=True)

            importances.plot(kind="barh", ax=ax)
            ax.set_xlabel("Importance")
            ax.set_title(f"Feature Importance - {model_name}")
            plt.tight_layout()

            mlflow.log_figure(fig, "feature_importance.png")
            plt.close(fig)

        logger.info(
            f"{model_name:20s} | RMSE: {metrics.rmse:.3f} | R²: {metrics.r2:.3f} | MAE: {metrics.mae:.3f}"
        )

        return model, metrics
