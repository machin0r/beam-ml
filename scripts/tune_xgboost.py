#!/usr/bin/env python3
"""
Hyperparameter tuning for XGBoost
"""

import mlflow
from xgboost import XGBRegressor
from src.data.load import load_training_data
from src.models.train import train_and_log_model

from src.logging_config import get_logger


logger = get_logger(__name__)


def main():
    mlflow.set_experiment("lpbf_density_prediction")
    X_train, X_test, Y_train, Y_test, expected_columns = load_training_data()
    logger.info(
        f"Loaded {len(X_train)} training samples with {len(expected_columns)} features"
    )

    # Grid of parameters to try
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.3],
        "n_estimators": [50, 100, 200],
    }

    logger.info("Testing parameter combinations...")

    for max_depth in param_grid["max_depth"]:
        for lr in param_grid["learning_rate"]:
            for n_est in param_grid["n_estimators"]:

                model_name = f"XGB_d{max_depth}_lr{lr}_n{n_est}"

                model = XGBRegressor(
                    max_depth=max_depth,
                    learning_rate=lr,
                    n_estimators=n_est,
                    random_state=42,
                )

                train_and_log_model(model, model_name, X_train, X_test, Y_train, Y_test)

    logger.info("Hyperparameter tuning complete!")


if __name__ == "__main__":
    main()
