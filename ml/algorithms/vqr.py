# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: VQR
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================

from typing import Callable

import numpy as np

from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQR

# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================


def vqr(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    k: int,
    estimator=None,
    feature_map_reps: int = 2,
    ansatz_reps: int = 2,
    maxiter: int = 100,
    initial_point: np.ndarray | None = None,
    callback: Callable[[np.ndarray, float], None] | None = None,
) -> VQR | None:
    """Build Variational Quantum Regressor Model

    Args:
        X_train (np.ndarray)
        Y_train (np.ndarray)
        k (int): Number of feature
        estimator: A qiskit ``BaseEstimatorV2`` primitive. When ``None`` the
            cloud ``QuappEstimator`` is used (spends QuApp credit). Pass a local
            ``qiskit.primitives.StatevectorEstimator()`` to run entirely offline
            with no cloud calls. The ``QuappEstimator`` is only imported when it
            is actually needed, so the local path never touches it.
        feature_map_reps (int): reps for the ``zz_feature_map``.
        ansatz_reps (int): reps for the ``real_amplitudes`` ansatz.
        maxiter (int): COBYLA max iterations.
        initial_point: Optional starting weights for the ansatz. Used to run
            multiple random restarts and pick the best by validation loss.
        callback: Optional ``(weights, objective_value)`` callback forwarded to
            VQR so training progress can be observed.

    Returns:
        VQR | None
    """
    assert X_train.shape[1] == k, "Dataset's features don't equal to k!"
    # Feature Map
    featuremap = zz_feature_map(feature_dimension=k, reps=feature_map_reps)

    # Ansatz
    ansatz = real_amplitudes(num_qubits=k, reps=ansatz_reps)

    # Optimizer
    optimizer = COBYLA(maxiter=maxiter)

    # Estimator
    if estimator is None:
        # Lazy import so the local StatevectorEstimator path never imports or
        # instantiates the cloud estimator (and never needs an access token).
        from algorithms.quapp_estimator import QuappEstimator

        estimator = QuappEstimator()

    # Build Model
    model = VQR(
        feature_map=featuremap,
        ansatz=ansatz,
        optimizer=optimizer,
        estimator=estimator,
        initial_point=initial_point,
        callback=callback,
    )

    # Fit Data
    model.fit(X_train, Y_train)
    return model


# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================


def main():
    from qiskit.primitives import StatevectorEstimator

    X = np.array(
        [
            [0.1, 0.2],
            [0.2, 0.3],
            [0.3, 0.4],
            [0.4, 0.5],
        ]
    )

    y = np.array([1.0, 2.0, 3.0, 4.0])
    # Local, credit-free smoke test.
    model = vqr(X, y, k=2, estimator=StatevectorEstimator(), maxiter=20)


if __name__ == "__main__":
    main()
