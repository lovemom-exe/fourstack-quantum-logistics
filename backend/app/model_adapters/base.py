"""Common model adapter contract."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class PredictionModelAdapter(Protocol):
    @property
    def is_ready(self) -> bool: ...

    @property
    def is_mock(self) -> bool: ...

    @property
    def model_name(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    @property
    def target(self) -> str: ...

    def load(self) -> None: ...

    def predict(
        self, features: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...
