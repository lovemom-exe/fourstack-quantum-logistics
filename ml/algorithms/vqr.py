# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: VQR
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================

import numpy as np
import inspect

from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQR
from qiskit.primitives import StatevectorEstimator

# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================


def vqr(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    k: int,
) -> VQR | None:
    """Build Variational Quantum Regressor Model

    Args:
        X_train (np.ndarray)
        Y_train (np.ndarray)
        k (int): Number of feature

    Returns:
        VQR | None
    """
    assert X_train.shape[1] == k, "Dataset's features don't equal to k!"
    # Feature Map
    featuremap = zz_feature_map(feature_dimension=k, reps=2)

    # Ansatz
    ansatz = real_amplitudes(num_qubits=k, reps=2)

    # Optimizer
    optimizer = COBYLA(maxiter=100)

    # Estimator
    estimator = StatevectorEstimator()

    # Build Model
    model = VQR(
        feature_map=featuremap,
        ansatz=ansatz,
        optimizer=optimizer,
        estimator=estimator,
    )

    # Fit Data
    print("Fit Data: ", end="")
    model.fit(X_train, Y_train)
    print("Done")
    return model


# def main():
#     estimator = StatevectorEstimator()
#     print(inspect.signature(estimator.run))


# if __name__ == "__main__":
#     main()
