#!/usr/bin/env python3
"""
Hyperparameter tuning for Random Forest
"""

import mlflow
from sklearn.ensemble import RandomForestRegressor
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
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
    }

    logger.info("Testing parameter combinations...")

    for n_est in param_grid["n_estimators"]:
        for max_depth in param_grid["max_depth"]:
            for min_split in param_grid["min_samples_split"]:

                model_name = f"RF_n{n_est}_d{max_depth}_ms{min_split}"

                model = RandomForestRegressor(
                    n_estimators=n_est,
                    max_depth=max_depth,
                    min_samples_split=min_split,
                    random_state=42,
                    n_jobs=-1,
                )

                train_and_log_model(
                    model,
                    model_name,
                    X_train,
                    X_test,
                    Y_train,
                    Y_test,
                )

    logger.info("Hyperparameter tuning complete!")


if __name__ == "__main__":
    main()
