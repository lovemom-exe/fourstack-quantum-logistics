# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: K-Nearest Neighbors Regressor
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================

import numpy as np
from sklearn.neighbors import KNeighborsRegressor


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
def knn(
    X_train: np.ndarray,
    Y_train: np.ndarray,
) -> KNeighborsRegressor | None:

    if not isinstance(X_train, np.ndarray):
        return None

    if not isinstance(Y_train, np.ndarray):
        return None

    # Model
    model = KNeighborsRegressor(
        n_neighbors=5,
        weights="distance",
        algorithm="auto",
        p=2,
        n_jobs=-1,
    )

    model.fit(X_train, Y_train)

    return model
