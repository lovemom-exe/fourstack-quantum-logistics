# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: This module is responsible for splitting the dataset into training, validation, and test sets
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from typing import Tuple

# ==========================================================================
# PARAMETERS
# ==========================================================================


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
def split(
    data: pd.DataFrame, y_location: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = data.iloc[:, y_location]
    X = data.drop(data.columns[y_location], axis=1)

    X = X.to_numpy()
    y = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=42
    )

    return X_train, X_test, y_train, y_test


# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================
