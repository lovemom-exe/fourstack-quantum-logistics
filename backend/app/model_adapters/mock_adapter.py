"""Clearly labeled deterministic fallback predictions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class MockModelAdapter:
    @property
    def is_ready(self) -> bool:
        return True

    @property
    def is_mock(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "deterministic-mock"

    @property
    def model_version(self) -> str:
        return "0.0.0"

    @property
    def target(self) -> str:
        return "units_sold"

    def load(self) -> None:
        return None

    def predict(
        self, features: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("Mock prediction features must be two-dimensional.")
        weighted = matrix @ np.arange(1, matrix.shape[1] + 1, dtype=float)
        return np.maximum(0.0, 10.0 + np.mod(np.abs(weighted), 490.0))
