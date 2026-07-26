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
    precision: float | None = None,
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
        precision: Expectation-value precision for the underlying
            ``EstimatorQNN``. Leave ``None`` to keep qiskit-machine-learning's
            default of ``0.015625``.

            IMPORTANT: that default is NOT exact. ``EstimatorQNN`` calls
            ``estimator.run(..., precision=0.015625)``, which overrides a
            ``StatevectorEstimator``'s own ``default_precision=0.0`` and makes it
            inject sampled noise (~4096 shots). With an unseeded estimator the
            objective is then random on every evaluation, so two identical runs
            diverge and results are not reproducible. Pass ``precision=0.0`` for
            an exact, deterministic statevector objective.

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

    # VQR builds its EstimatorQNN internally and never exposes the precision,
    # and `default_precision` is a read-only property - so the backing field is
    # the only way in. Asserted so a library change can't silently drop this.
    if precision is not None:
        model.neural_network._default_precision = precision
        assert model.neural_network.default_precision == precision, (
            "Could not set EstimatorQNN precision; qiskit-machine-learning may "
            "have changed its internals."
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
