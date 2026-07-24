# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: Train Model, QML or ML
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
import os

from utils.path import (
    FRESH_RETAIL_EVAL_PATH_FEATURE,
    FRESH_RETAIL_TRAIN_PATH_FEATURE,
    VQR_TRAIN_RESULT,
)

import pandas as pd
import numpy as np
from typing import Tuple, List

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from evaluation.metrics import mae, mape, mse, r2, rmse
from algorithms.xgboost import xgboost
from algorithms.random_forest import random_forest
from algorithms.k_nearest_n import knn

# ==========================================================================
# PARAMETERS
# ==========================================================================
DEMAND = "sale_amount"


def _read_csv_if_exists(path: str):
    """Avoid crashing at import time before the featured CSVs have been generated."""
    return pd.read_csv(path) if os.path.exists(path) else None


food_train_df = _read_csv_if_exists(FRESH_RETAIL_TRAIN_PATH_FEATURE)
food_eval_df = _read_csv_if_exists(FRESH_RETAIL_EVAL_PATH_FEATURE)

# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================


# Create Sample
def create_food_sample(
    n_sample: int = 3000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    train_sample = food_train_df.sample(
        n=n_sample,
        random_state=random_state,
    )

    test_sample = food_eval_df.sample(
        n=n_sample,
        random_state=random_state,
    )

    F_train_s = train_sample.drop(columns=[DEMAND]).to_numpy()
    f_train_s = train_sample[DEMAND].to_numpy()

    F_test_s = test_sample.drop(columns=[DEMAND]).to_numpy()
    f_test_s = test_sample[DEMAND].to_numpy()

    return (
        F_train_s,
        f_train_s,
        F_test_s,
        f_test_s,
    )


# ----------------------------------------------------------
# Feature Scaler
def feature_scaler(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


# ----------------------------------------------------------
# Write CSV
def write_csv(path: str, content: List = []) -> None:
    if os.path.exists(path):
        with open(path, "a", newline="") as vqr:
            vqr.write("model_name_2")
    else:
        print(f'[ERROR]path: "{path}" does not exist')
        os.mkdir(path)
        return


# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================
def main():

    # ----------------------------------------------------------
    # Create Sample
    n = 250000
    x_train_raw, y_train, x_test_raw, y_test = create_food_sample(n_sample=n)
    # x_train_scaled, x_test_scaled = feature_scaler(x_train_raw, x_test_raw)

    # ----------------------------------------------------------
    # Select number of feature
    k = 17
    selection = SelectKBest(score_func=mutual_info_regression, k=k)
    X_train_new = selection.fit_transform(x_train_raw, y_train)
    X_test_new = selection.transform(x_test_raw)

    # ----------------------------------------------------------
    # Train model
    # model = xgboost(X_train_new, y_train)
    # model = vqr_train(X_train=X_train_new, Y_train=y_train, k=4)
    # model = random_forest(X_train_new, y_train)
    model = knn(X_train_new, y_train)
    if model is None:
        print("not found model")
        return
    # ----------------------------------------------------------
    # Prediction and Evalation
    if isinstance(X_test_new, np.ndarray):
        y_pred = model.predict(X_test_new)
        print("Model: XGBoost")
        print("=" * 50)
        print(f"n = {n}, k = {k}")
        print(f"R^2: {r2(y_test, y_pred)}")
        print(f"MAE: {mae(y_test, y_pred)}")
        print(f"MSE: {mse(y_test, y_pred)}")
        print(f"RMSE: {rmse(y_test, y_pred)}")
        print(f"MAPE: {mape(y_test, y_pred)}")
        print(f"y_pred_1: {y_pred[0]}")
        print(f"y_test_1: {y_test[0]}")
        print(f"y_pred_2: {y_pred[10]}")
        print(f"y_test_2: {y_test[10]}")
        print(f"y_pred_3: {y_pred[20]}")
        print(f"y_test_3: {y_test[20]}")

    # ----------------------------------------------------------
    # Write Data to CSV file
    # write_csv(
    #     VQR_TRAIN_RESULT,
    # )


if __name__ == "__main__":
    main()
