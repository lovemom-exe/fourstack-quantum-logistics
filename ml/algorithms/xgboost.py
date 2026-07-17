# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: XGBoost Regression
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================

import numpy as np
from xgboost import XGBRegressor

# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================


def xgboost(
    X_train: np.ndarray,
    Y_train: np.ndarray,
) -> XGBRegressor | None:

    if not isinstance(X_train, np.ndarray):
        return
    if not isinstance(Y_train, np.ndarray):
        return

    # Model
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=3,
    )

    model.fit(X_train, Y_train)
    return model
