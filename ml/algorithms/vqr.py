# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: VQR
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================

import numpy as np

from algorithms.quapp_estimator import QuappEstimator

from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQR
from qiskit.primitives import StatevectorEstimator
from qiskit.qasm2 import dumps

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

    # Estimar
    # estimator = QuappEstimator()
    estimator = StatevectorEstimator()

    # Build Model
    model = VQR(
        feature_map=featuremap,
        ansatz=ansatz,
        optimizer=optimizer,
        estimator=estimator,
    )

    # Fit Data
    model.fit(X_train, Y_train)
    return model


# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================


def main():
    pass
    # X = np.array(
    #     [
    #         [0.1, 0.2],
    #         [0.2, 0.3],
    #         [0.3, 0.4],
    #         [0.4, 0.5],
    #     ]
    # )

    # y = np.array([1.0, 2.0, 3.0, 4.0])
    # model = vqr(X, y, k=2)
    # if model is None:
    #     return

    # X_test = np.array([[0.7, 0.9]])
    # y_test = model.predict(X_test)
    # print(y_test)


if __name__ == "__main__":
    main()
