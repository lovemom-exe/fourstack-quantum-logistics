"""Optional sklearn artifact adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from typing import Protocol, cast

import joblib
import numpy as np
from numpy.typing import NDArray

from app.core.exceptions import ModelArtifactError


class SklearnPredictorProtocol(Protocol):
    def predict(self, values: object) -> object: ...


class SklearnModelAdapter:
    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self._lock = RLock()
        self._model: SklearnPredictorProtocol | None = None
        self._metadata: dict[str, object] = {}

    @property
    def is_ready(self) -> bool:
        return (self.artifact_dir / "model.joblib").is_file()

    @property
    def is_mock(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return str(self._metadata.get("model_name", "sklearn-model"))

    @property
    def model_version(self) -> str:
        return str(self._metadata.get("model_version", "unknown"))

    @property
    def target(self) -> str:
        return str(self._metadata.get("target", "units_sold"))

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                self._model = cast(
                    SklearnPredictorProtocol,
                    joblib.load(self.artifact_dir / "model.joblib"),
                )
                metadata_path = self.artifact_dir / "model_metadata.json"
                if metadata_path.is_file():
                    value = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if isinstance(value, Mapping):
                        self._metadata = dict(value)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ModelArtifactError("Sklearn model artifacts could not be loaded.") from exc

    def predict(
        self, features: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        self.load()
        if self._model is None:
            raise ModelArtifactError("Sklearn model is not loaded.")
        predictions = np.asarray(self._model.predict(features), dtype=float).reshape(-1)
        return np.clip(predictions, 0.0, None)
