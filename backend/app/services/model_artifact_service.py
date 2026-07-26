"""Lightweight model artifact contract inspection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from app.core.exceptions import ModelArtifactError
from app.schemas.model import ModelStatusResponse


REQUIRED_VQR_FILES = (
    "vqr_model.dill",
    "x_scaler.joblib",
    "y_scaler.joblib",
    "feature_selector.joblib",
    "feature_schema.json",
    "model_metadata.json",
)


class ModelArtifactService:
    def __init__(self, artifact_dir: Path, public_directory: str) -> None:
        self.artifact_dir = artifact_dir
        self.public_directory = public_directory

    @property
    def missing_files(self) -> list[str]:
        return [
            filename
            for filename in REQUIRED_VQR_FILES
            if not (self.artifact_dir / filename).is_file()
        ]

    def read_json(self, filename: str) -> dict[str, object]:
        path = self.artifact_dir / filename
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelArtifactError(
                f"{filename} could not be read.",
                details={"filename": filename},
            ) from exc
        if not isinstance(value, Mapping):
            raise ModelArtifactError(f"{filename} must contain a JSON object.")
        return dict(value)

    def feature_schema(self) -> dict[str, object]:
        if "feature_schema.json" in self.missing_files:
            raise ModelArtifactError("feature_schema.json is missing.")
        schema = self.read_json("feature_schema.json")
        feature_order = schema.get("feature_order") or schema.get("selected_features")
        if not isinstance(feature_order, list) or not all(
            isinstance(item, str) for item in feature_order
        ):
            raise ModelArtifactError("feature_schema.json has no valid feature_order.")
        if len(feature_order) != len(set(feature_order)):
            raise ModelArtifactError("feature_order contains duplicate features.")
        feature_count = schema.get("feature_count")
        if not isinstance(feature_count, int) or feature_count != len(feature_order):
            raise ModelArtifactError(
                "feature_count does not match feature_order.",
                details={
                    "feature_count": feature_count,
                    "ordered_feature_count": len(feature_order),
                },
            )
        if "units_sold" in feature_order:
            raise ModelArtifactError("The target units_sold cannot be a model feature.")
        selected_features = schema.get("selected_features")
        if (
            isinstance(selected_features, list)
            and selected_features
            and selected_features != feature_order
        ):
            raise ModelArtifactError(
                "selected_features and feature_order do not match."
            )
        candidate_order = schema.get("candidate_feature_order")
        if candidate_order is not None:
            if not isinstance(candidate_order, list) or not all(
                isinstance(item, str) for item in candidate_order
            ):
                raise ModelArtifactError(
                    "candidate_feature_order must be a list of strings."
                )
            if len(candidate_order) != len(set(candidate_order)):
                raise ModelArtifactError(
                    "candidate_feature_order contains duplicate features."
                )
            missing_selected = sorted(set(feature_order) - set(candidate_order))
            if missing_selected:
                raise ModelArtifactError(
                    "Selected features are absent from candidate_feature_order.",
                    details={"features": missing_selected},
                )
            if "units_sold" in candidate_order:
                raise ModelArtifactError(
                    "The target units_sold cannot be a candidate feature."
                )
        return schema

    def status(self) -> ModelStatusResponse:
        missing = self.missing_files
        if missing:
            return ModelStatusResponse(
                ready=False,
                artifact_directory=self.public_directory,
                missing_files=missing,
            )
        schema = self.feature_schema()
        metadata = self.read_json("model_metadata.json")
        selected = schema.get("feature_order") or schema.get("selected_features") or []
        return ModelStatusResponse(
            ready=True,
            artifact_directory=self.public_directory,
            model_name=str(
                metadata.get("model_name", schema.get("model_name", "perishable-demand-vqr"))
            ),
            model_version=str(
                metadata.get("model_version", schema.get("model_version", "unknown"))
            ),
            target=str(metadata.get("target", schema.get("target", "units_sold"))),
            feature_count=int(schema["feature_count"]),
            selected_features=[str(item) for item in selected],
        )
