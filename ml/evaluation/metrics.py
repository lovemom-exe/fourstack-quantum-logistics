# ==========================================================================
# Author: Nguyen Minh Hoang
# Purpose: Shared regression metrics for model evaluation
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
import numpy as np

# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
# Hand-rolled instead of calling sklearn.metrics at each site, so every training
# script (ml_train, train_xgboost, eval_ksweep, train_vqr_local/ksweep, ...)
# reports metrics from ONE shared definition that cannot drift between call
# sites. Each function coerces its inputs to float arrays first, so Python lists,
# pandas Series and object dtypes all score identically.


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R^2 coefficient of determination (standard 1 - SS_res / SS_tot form)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # Constant y_true -> zero total variance, so R^2 is undefined. Return 0.0
    # (a model explains none of a spread that does not exist) rather than
    # dividing by zero and yielding inf/nan.
    if ss_tot == 0:
        return 0.0

    return float(1 - ss_res / ss_tot)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error, in the target's own units."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error (sqrt of mse; penalizes large errors more)."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error, as a percentage.
    Rows where y_true == 0 are masked out to avoid division blow-up.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true != 0
    if not np.any(mask):
        return 0.0

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
