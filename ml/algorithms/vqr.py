"""Configurable Variational Quantum Regressor construction and training."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes, z_feature_map, zz_feature_map
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.algorithms import VQR
from qiskit_machine_learning.optimizers import COBYLA, SLSQP, Optimizer

VQRCallback = Callable[[NDArray[np.float64], float], None]


def _build_feature_map(
    feature_count: int,
    feature_map_name: str,
    reps: int,
    entanglement: str,
) -> QuantumCircuit:
    """Create a supported Qiskit feature map."""
    normalized_name = feature_map_name.strip().lower()
    if normalized_name == "zz":
        if feature_count < 2:
            raise ValueError("The ZZ feature map requires at least two features.")
        return zz_feature_map(
            feature_dimension=feature_count,
            reps=reps,
            entanglement=entanglement,
        )
    if normalized_name == "z":
        return z_feature_map(feature_dimension=feature_count, reps=reps)
    raise ValueError(
        f"Unsupported feature_map_name={feature_map_name!r}; expected 'z' or 'zz'."
    )


def _build_optimizer(optimizer_name: str, maxiter: int) -> Optimizer:
    """Create a supported Qiskit Machine Learning optimizer."""
    normalized_name = optimizer_name.strip().upper()
    if normalized_name == "COBYLA":
        return COBYLA(maxiter=maxiter)
    if normalized_name == "SLSQP":
        return SLSQP(maxiter=maxiter)
    raise ValueError(
        f"Unsupported optimizer_name={optimizer_name!r}; "
        "expected 'COBYLA' or 'SLSQP'."
    )


def build_vqr(
    *,
    feature_count: int,
    feature_map_name: str = "zz",
    feature_map_reps: int = 1,
    ansatz_name: str = "real_amplitudes",
    ansatz_reps: int = 1,
    entanglement: str = "linear",
    optimizer_name: str = "COBYLA",
    maxiter: int = 20,
    random_state: int = 42,
    callback: VQRCallback | None = None,
    initial_point: NDArray[np.float64] | None = None,
) -> VQR:
    """Build an unfitted VQR configured for small simulator experiments.

    The function uses Qiskit's ``StatevectorEstimator`` and functional circuit
    constructors, which are compatible with the project's locked Qiskit 2.5.0
    and Qiskit Machine Learning 0.9.0 versions.
    """
    if feature_count < 1:
        raise ValueError("feature_count must be at least 1.")
    if feature_map_reps < 1 or ansatz_reps < 1:
        raise ValueError("Circuit repetition counts must be at least 1.")
    if maxiter < 1:
        raise ValueError("maxiter must be at least 1.")

    feature_map = _build_feature_map(
        feature_count=feature_count,
        feature_map_name=feature_map_name,
        reps=feature_map_reps,
        entanglement=entanglement,
    )

    if ansatz_name.strip().lower() != "real_amplitudes":
        raise ValueError(
            f"Unsupported ansatz_name={ansatz_name!r}; "
            "expected 'real_amplitudes'."
        )
    ansatz = real_amplitudes(
        num_qubits=feature_count,
        reps=ansatz_reps,
        entanglement=entanglement,
    )

    if initial_point is None:
        rng = np.random.default_rng(random_state)
        initial_point = rng.uniform(-0.1, 0.1, size=ansatz.num_parameters)
    else:
        initial_point = np.asarray(initial_point, dtype=float)
        if initial_point.shape != (ansatz.num_parameters,):
            raise ValueError(
                "initial_point must contain exactly "
                f"{ansatz.num_parameters} values; got shape {initial_point.shape}."
            )

    return VQR(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=_build_optimizer(optimizer_name, maxiter),
        estimator=StatevectorEstimator(seed=random_state),
        callback=callback,
        initial_point=initial_point,
    )


def train_vqr(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    *,
    feature_count: int,
    feature_map_name: str = "zz",
    feature_map_reps: int = 1,
    ansatz_name: str = "real_amplitudes",
    ansatz_reps: int = 1,
    entanglement: str = "linear",
    optimizer_name: str = "COBYLA",
    maxiter: int = 20,
    random_state: int = 42,
    callback: VQRCallback | None = None,
    initial_point: NDArray[np.float64] | None = None,
) -> VQR:
    """Build and fit a configurable VQR."""
    X_array = np.asarray(X_train, dtype=float)
    y_array = np.asarray(y_train, dtype=float).reshape(-1)
    if X_array.ndim != 2:
        raise ValueError(f"X_train must be two-dimensional; got shape {X_array.shape}.")
    if X_array.shape[1] != feature_count:
        raise ValueError(
            f"X_train has {X_array.shape[1]} features but "
            f"feature_count={feature_count} was requested."
        )
    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError("X_train and Y_train must contain the same number of rows.")

    model = build_vqr(
        feature_count=feature_count,
        feature_map_name=feature_map_name,
        feature_map_reps=feature_map_reps,
        ansatz_name=ansatz_name,
        ansatz_reps=ansatz_reps,
        entanglement=entanglement,
        optimizer_name=optimizer_name,
        maxiter=maxiter,
        random_state=random_state,
        callback=callback,
        initial_point=initial_point,
    )
    model.fit(X_array, y_array)
    return model


def vqr(
    X_train: NDArray[np.float64],
    Y_train: NDArray[np.float64],
    k: int,
    *,
    feature_map_name: str = "zz",
    feature_map_reps: int = 1,
    ansatz_name: str = "real_amplitudes",
    ansatz_reps: int = 1,
    entanglement: str = "linear",
    optimizer_name: str = "COBYLA",
    maxiter: int = 20,
    random_state: int = 42,
    callback: VQRCallback | None = None,
    initial_point: NDArray[np.float64] | None = None,
) -> VQR:
    """Preserve the project's original ``vqr(X, Y, k)`` entry point."""
    return train_vqr(
        X_train,
        Y_train,
        feature_count=k,
        feature_map_name=feature_map_name,
        feature_map_reps=feature_map_reps,
        ansatz_name=ansatz_name,
        ansatz_reps=ansatz_reps,
        entanglement=entanglement,
        optimizer_name=optimizer_name,
        maxiter=maxiter,
        random_state=random_state,
        callback=callback,
        initial_point=initial_point,
    )


def main() -> None:
    """Keep module execution intentionally side-effect free."""


if __name__ == "__main__":
    main()
