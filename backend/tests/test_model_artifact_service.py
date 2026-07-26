"""Artifact status and dynamic schema contract tests."""

import json

import pytest

from app.core.exceptions import ModelArtifactError, ModelFeatureMismatchError
from app.services.feature_service import FeatureService
from app.services.model_artifact_service import (
    ModelArtifactService,
    REQUIRED_VQR_FILES,
)


def test_model_status_reports_missing_artifacts(tmp_path) -> None:
    service = ModelArtifactService(tmp_path, "../ml/models/perishable_vqr")
    status = service.status()
    assert status.ready is False
    assert status.missing_files == list(REQUIRED_VQR_FILES)
    assert str(tmp_path) not in status.artifact_directory


def test_model_status_route_reports_training_incomplete(client) -> None:
    response = client.get("/api/v1/models/status")
    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert "vqr_model.dill" in response.json()["missing_files"]


def test_model_feature_order_comes_from_schema(tmp_path) -> None:
    for filename in REQUIRED_VQR_FILES:
        (tmp_path / filename).write_bytes(b"placeholder")
    (tmp_path / "feature_schema.json").write_text(
        json.dumps(
            {
                "target": "units_sold",
                "feature_count": 2,
                "feature_order": ["selling_price", "category_Dairy"],
                "candidate_feature_order": [
                    "selling_price",
                    "category_Dairy",
                    "category_Produce",
                ],
            }
        )
    )
    (tmp_path / "model_metadata.json").write_text(
        json.dumps(
            {
                "model_name": "perishable-demand-vqr",
                "model_version": "1.0.0",
                "target": "units_sold",
            }
        )
    )
    status = ModelArtifactService(tmp_path, "configured/model").status()
    assert status.ready is True
    assert status.feature_count == 2
    assert status.selected_features == ["selling_price", "category_Dairy"]
    frame = FeatureService().build_frame(
        [{"selling_price": 20, "category": "Dairy"}],
        {
            "feature_count": 2,
            "feature_order": ["selling_price", "category_Dairy"],
            "candidate_feature_order": [
                "selling_price",
                "category_Dairy",
                "category_Produce",
            ],
        },
    )
    assert list(frame.columns) == [
        "selling_price",
        "category_Dairy",
        "category_Produce",
    ]
    assert frame.iloc[0].tolist() == [20.0, 1.0, 0.0]


def test_feature_count_mismatch_and_target_leakage_are_rejected(tmp_path) -> None:
    schema_path = tmp_path / "feature_schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "feature_count": 3,
                "feature_order": ["selling_price", "category_Dairy"],
            }
        )
    )
    service = ModelArtifactService(tmp_path, "model")
    with pytest.raises(ModelArtifactError):
        service.feature_schema()
    with pytest.raises(ModelFeatureMismatchError):
        FeatureService().build_frame(
            [{"units_sold": 10}],
            {
                "feature_count": 1,
                "feature_order": ["units_sold"],
            },
        )
