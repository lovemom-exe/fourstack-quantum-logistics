"""Lazy, cached Qiskit VQR inference adapter."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Protocol, cast

import joblib
import numpy as np
from numpy.typing import NDArray

from app.core.exceptions import ModelArtifactError, ModelFeatureMismatchError
from app.services.model_artifact_service import ModelArtifactService


class TransformerProtocol(Protocol):
    def transform(self, values: object) -> object: ...


class PredictorProtocol(Protocol):
    def predict(self, values: object) -> object: ...


class VQRModelAdapter:
    """Load training artifacts once; never fit preprocessing at inference."""

    def __init__(
        self,
        artifact_dir: Path,
        *,
        artifact_service: ModelArtifactService | None = None,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.artifact_service = artifact_service or ModelArtifactService(
            artifact_dir, str(artifact_dir)
        )
        self._lock = RLock()
        self._loaded = False
        self._model: PredictorProtocol | None = None
        self._selector: TransformerProtocol | None = None
        self._x_scaler: TransformerProtocol | None = None
        self._y_scaler: TransformerProtocol | None = None
        self._schema: dict[str, object] = {}
        self._metadata: dict[str, object] = {}

    @property
    def is_ready(self) -> bool:
        return not self.artifact_service.missing_files

    @property
    def is_mock(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return str(self._metadata.get("model_name", "perishable-demand-vqr"))

    @property
    def model_version(self) -> str:
        return str(self._metadata.get("model_version", "unknown"))

    @property
    def target(self) -> str:
        return str(self._metadata.get("target", self._schema.get("target", "units_sold")))

    @property
    def schema(self) -> dict[str, object]:
        if not self._loaded:
            self.load()
        return dict(self._schema)

    def _load_vqr_model(self, path: Path) -> PredictorProtocol:
        try:
            from qiskit_machine_learning.algorithms import VQR
        except ImportError as exc:
            raise ModelArtifactError(
                "Qiskit Machine Learning is not installed."
            ) from exc
        return cast(PredictorProtocol, VQR.from_dill(str(path)))

    def load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            status = self.artifact_service.status()
            if not status.ready:
                raise ModelArtifactError(
                    "Required VQR artifacts are missing.",
                    details={"missing_files": status.missing_files},
                )
            self._schema = self.artifact_service.feature_schema()
            self._metadata = self.artifact_service.read_json("model_metadata.json")
            try:
                self._selector = cast(
                    TransformerProtocol,
                    joblib.load(self.artifact_dir / "feature_selector.joblib"),
                )
                self._x_scaler = cast(
                    TransformerProtocol,
                    joblib.load(self.artifact_dir / "x_scaler.joblib"),
                )
                self._y_scaler = cast(
                    TransformerProtocol,
                    joblib.load(self.artifact_dir / "y_scaler.joblib"),
                )
                self._model = self._load_vqr_model(
                    self.artifact_dir / "vqr_model.dill"
                )
            except (OSError, ValueError, TypeError) as exc:
                raise ModelArtifactError(
                    "One or more VQR artifacts could not be loaded."
                ) from exc
            self._loaded = True

    def predict(
        self, features: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        self.load()
        if (
            self._selector is None
            or self._x_scaler is None
            or self._y_scaler is None
            or self._model is None
        ):
            raise ModelArtifactError("VQR adapter did not initialize completely.")
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2:
            raise ModelFeatureMismatchError("Prediction features must be two-dimensional.")
        expected_inputs = self._schema.get("candidate_feature_order")
        if not isinstance(expected_inputs, list):
            expected_inputs = self._schema.get("feature_order")
        if not isinstance(expected_inputs, list) or matrix.shape[1] != len(expected_inputs):
            raise ModelFeatureMismatchError(
                "Input feature count does not match the saved schema.",
                details={
                    "received": matrix.shape[1],
                    "expected": len(expected_inputs)
                    if isinstance(expected_inputs, list)
                    else None,
                },
            )
        selected = self._selector.transform(matrix)
        selected_array = np.asarray(selected, dtype=float)
        expected_selected = self._schema.get("feature_count")
        if (
            not isinstance(expected_selected, int)
            or selected_array.ndim != 2
            or selected_array.shape[1] != expected_selected
        ):
            raise ModelFeatureMismatchError(
                "Feature selector output does not match the saved feature_count.",
                details={
                    "received": selected_array.shape[1]
                    if selected_array.ndim == 2
                    else None,
                    "expected": expected_selected,
                },
            )
        support_method = getattr(self._selector, "get_support", None)
        saved_selected = self._schema.get("feature_order")
        if callable(support_method) and isinstance(saved_selected, list):
            support = np.asarray(support_method(), dtype=bool)
            if support.shape == (len(expected_inputs),):
                actual_selected = [
                    str(name)
                    for name, included in zip(
                        expected_inputs, support, strict=True
                    )
                    if included
                ]
                if actual_selected != saved_selected:
                    raise ModelFeatureMismatchError(
                        "Saved selector order does not match feature_order."
                    )
        scaled_features = self._x_scaler.transform(selected_array)
        scaled_predictions = self._model.predict(scaled_features)
        inverse = getattr(self._y_scaler, "inverse_transform", None)
        if not callable(inverse):
            raise ModelArtifactError("Saved y_scaler has no inverse_transform method.")
        predictions = np.asarray(
            inverse(np.asarray(scaled_predictions, dtype=float).reshape(-1, 1)),
            dtype=float,
        ).reshape(-1)
        predictions = np.clip(predictions, 0.0, None)
        if not np.isfinite(predictions).all():
            raise ModelArtifactError("VQR inference produced a non-finite value.")
        return predictions
