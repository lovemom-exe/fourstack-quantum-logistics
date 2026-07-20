# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: Random forest
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================

import numpy as np
from sklearn.ensemble import RandomForestRegressor


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
def random_forest(
    X_train: np.ndarray,
    Y_train: np.ndarray,
) -> RandomForestRegressor | None:

    if not isinstance(X_train, np.ndarray):
        return
    if not isinstance(Y_train, np.ndarray):
        return

    # Model
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, Y_train)
    return model
