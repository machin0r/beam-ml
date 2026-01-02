#!/usr/bin/env python3
"""
Analyse the best model to understand what's happening
"""

import mlflow
import pandas as pd
import matplotlib.pyplot as plt

from src.logging_config import get_logger
from src.data.load import load_training_data


logger = get_logger(__name__)


def main():
    # Load the best run
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("lpbf_density_prediction")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1,
    )

    best_run = runs[0]
    logger.info(f"Best model: {best_run.data.tags.get('mlflow.runName')}")
    logger.info(f"RMSE: {best_run.data.metrics['rmse']:.3f}")
    logger.info(f"R²: {best_run.data.metrics['r2']:.3f}")

    # Load the model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    X_train, X_test, Y_train, Y_test, expected_columns = load_training_data()

    importances = pd.DataFrame(
        {"feature": X_train.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    logger.info("=== TOP 10 MOST IMPORTANT FEATURES ===")
    logger.info(importances.head(10).to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    importances.head(15).plot(x="feature", y="importance", kind="barh", ax=ax)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("reports/figures/feature_importance_best.png", dpi=150)
    logger.info("Saved plot to reports/figures/feature_importance_best.png")


if __name__ == "__main__":
    main()
