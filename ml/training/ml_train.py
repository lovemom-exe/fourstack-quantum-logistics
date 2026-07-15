#==========================================================================
# Author: Hoang Anh Quan
# Purpose: Train Model, QML or ML
#==========================================================================
# IMPORTS & MODULE LOADING
#==========================================================================

from utils.path import FRESH_RETAIL_EVAL_PATH_FEATURE, FRESH_RETAIL_TRAIN_PATH_FEATURE

import pandas as pd
import numpy as np
from typing import Tuple

from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import StatevectorEstimator

from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
from qiskit_machine_learning.algorithms import VQR
from xgboost import XGBRegressor
#==========================================================================
# PARAMETERS
#==========================================================================
DEMAND = "sale_amount"
food_train_df = pd.read_csv(FRESH_RETAIL_TRAIN_PATH_FEATURE)
food_eval_df = pd.read_csv(FRESH_RETAIL_EVAL_PATH_FEATURE)

# ----------------------------------------------------------
# FREST RETAIL: TRAIN TEST SPLIT, PREDICT FEATURE SPLIT
# ----------------------------------------------------------
# f_train = food_train_df[DEMAND]
# F_train = food_train_df.drop(DEMAND)
# f_test = food_eval_df[DEMAND]
# F_test = food_eval_df.drop(DEMAND)

#==========================================================================
# CORE LOGIC & FUNCTIONS
#==========================================================================

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


# Feature Scaler
def feature_scaler(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test
# VQR
def vqr_train(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    k: int,
) -> VQR | None:
    assert X_train.shape[1] == k, (
        "Dataset's features don't equal to k!"
    )
    # Feature Map
    print("Feature Map: ", end="")
    featuremap = zz_feature_map(feature_dimension=k, reps = 2)
    print('Done')

    # Ansatz
    print("Ansatz: ", end="")
    ansatz = real_amplitudes(num_qubits=k, reps=2)
    print('Done')

    # Optimizer
    print("Optimizer: ", end="")
    optimizer = COBYLA(maxiter=100)
    print('Done')

    # Estimator
    print("Estimator: ", end="")
    estimator = StatevectorEstimator()
    print('Done')

    # gradient = ParamShiftEstimatorGradient(estimator)

    # Build Model
    model = VQR(
        feature_map=featuremap,
        ansatz= ansatz,
        optimizer=optimizer,
        estimator=estimator,
    )

    # Fit Data
    print("Fit Data: ", end="")
    model.fit(X_train, Y_train)
    print("Done")
    return model


# XGBoost
def xgb_train(
    X_train: np.ndarray,
    Y_train: np.ndarray,
) -> XGBRegressor | None:

    # Model
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        objective="reg:squarederror",
        tree_method='hist',
        n_jobs=3
    )

    #
    model.fit(X_train, Y_train)
    return model

#==========================================================================
# MAIN EXECUTION ENTRYPOINT
#==========================================================================
def main():

    # import qiskit
    # import qiskit_machine_learning

    # featuremap = zz_feature_map(feature_dimension=2, reps = 2)
    # ansatz = real_amplitudes(num_qubits=2, reps=2)
    # estimator = StatevectorEstimator()
    # print("qiskit:", qiskit.__version__)
    # print("qml:", qiskit_machine_learning.__version__)
    # print(type(featuremap))
    # print(type(ansatz))
    # print(type(estimator))
    x_train_raw, y_train, x_test_raw, y_test = create_food_sample(n_sample=3000)
    x_train_scaled, x_test_scaled = feature_scaler(x_train_raw, x_test_raw)
    selection = SelectKBest(score_func=mutual_info_regression, k=4)
    X_train_new = selection.fit_transform(x_train_scaled, y_train)
    X_test_new = selection.transform(x_test_scaled)

    featuremap = zz_feature_map(4)
    ansatz = real_amplitudes(4)
    estimator = StatevectorEstimator()

    print("Before constructor")
    model = VQR(
        feature_map=featuremap,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=100),
        estimator=estimator,
    )
    print("Before fit")
    model.fit(X_train_new[:50], y_train[:50])
    print("After fit")
    print("After constructor")
    y_pred = model.predict(X_test_new)

    print(f"r2_score: {r2_score(y_test, y_pred)}")
    print(f"mae: {mean_absolute_error(y_test, y_pred)}")
    print(f"mse: {mean_squared_error(y_test, y_pred)}")
    print(f"rmse: {root_mean_squared_error(y_test, y_pred)}")

    # model = vqr_train(
    #     X_train=X_train_new,
    #     Y_train=y_train,
    #     k = 4
    # )
    # if model is None:
    #     print("not found model")
    #     return
    # y_pred = model.predict(X_test_new)


if __name__ == "__main__":
    main()
