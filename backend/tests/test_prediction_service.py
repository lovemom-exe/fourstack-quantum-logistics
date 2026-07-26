"""Mock/real adapter behavior and preprocessing safety tests."""

from uuid import UUID

import numpy as np
import pytest

from app.core.config import Settings
from app.core.exceptions import ModelNotReadyError
from app.api.dependencies import get_prediction_service
from app.model_adapters.mock_adapter import MockModelAdapter
from app.model_adapters.vqr_adapter import VQRModelAdapter
from app.services.model_artifact_service import ModelArtifactService
from app.services.prediction_service import PredictionService
from app.schemas.prediction import (
    ForecastPointResponse,
    PredictionResponse,
)


class IdentityTransformer:
    def __init__(self) -> None:
        self.transform_calls = 0

    def transform(self, values):
        self.transform_calls += 1
        return np.asarray(values, dtype=float)


class NegativePredictor:
    def predict(self, values):
        return np.array([0.25, 0.75])


class NegativeInverseScaler:
    def __init__(self) -> None:
        self.inputs = None

    def transform(self, values):
        raise AssertionError("Target scaler transform should not run at inference.")

    def inverse_transform(self, values):
        self.inputs = np.asarray(values, dtype=float)
        return np.array([[-5.0], [12.0]])


def test_vqr_adapter_never_fits_and_clips_after_inverse_transform(tmp_path) -> None:
    adapter = VQRModelAdapter(
        tmp_path, artifact_service=ModelArtifactService(tmp_path, "model")
    )
    selector = IdentityTransformer()
    x_scaler = IdentityTransformer()
    y_scaler = NegativeInverseScaler()
    adapter._loaded = True
    adapter._selector = selector
    adapter._x_scaler = x_scaler
    adapter._y_scaler = y_scaler
    adapter._model = NegativePredictor()
    adapter._schema = {
        "candidate_feature_order": ["a", "b"],
        "feature_order": ["a", "b"],
        "feature_count": 2,
    }
    result = adapter.predict(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert selector.transform_calls == 1
    assert x_scaler.transform_calls == 1
    assert y_scaler.inputs.tolist() == [[0.25], [0.75]]
    assert result.tolist() == [0.0, 12.0]
    assert not hasattr(selector, "fit")
    assert not hasattr(x_scaler, "fit_transform")


def test_mock_adapter_is_deterministic_and_clearly_marked() -> None:
    adapter = MockModelAdapter()
    features = np.array([[1.0, 2.0], [1.0, 2.0]])
    first = adapter.predict(features)
    second = adapter.predict(features)
    assert adapter.is_mock is True
    assert adapter.model_name == "deterministic-mock"
    assert first.tolist() == second.tolist()


def test_missing_real_model_raises_503_contract_when_mock_disabled(tmp_path) -> None:
    settings = Settings(
        app_env="test",
        model_artifact_dir=tmp_path,
        allow_mock_predictions=False,
    )
    service = PredictionService(
        settings=settings,
        artifacts=ModelArtifactService(tmp_path, "model"),
        features=object(),
        predictions=object(),
        datasets=object(),
        mappings=object(),
        products=object(),
        suppliers=object(),
        warehouses=object(),
        stores=object(),
        sales=object(),
        inventory=object(),
    )
    with pytest.raises(ModelNotReadyError) as captured:
        service._select_adapter()
    assert captured.value.status_code == 503
    assert captured.value.code == "MODEL_NOT_READY"


def test_mock_prediction_response_is_explicitly_marked(app, client) -> None:
    class FakePredictionService:
        def predict(self, request, user):
            return PredictionResponse(
                job_id=UUID("55555555-5555-5555-5555-555555555555"),
                status="completed",
                model_name="deterministic-mock",
                model_version="0.0.0",
                target="units_sold",
                forecast_horizon_days=request.forecast_horizon_days,
                is_mock=True,
                results=[
                    ForecastPointResponse(
                        product_id=request.product_ids[0],
                        warehouse_id=request.warehouse_id,
                        forecast_date=request.forecast_start_date,
                        predicted_units_sold=42,
                    )
                ],
            )

    app.dependency_overrides[get_prediction_service] = (
        lambda: FakePredictionService()
    )
    response = client.post(
        "/api/v1/predictions",
        json={
            "warehouse_id": "33333333-3333-3333-3333-333333333333",
            "product_ids": ["66666666-6666-6666-6666-666666666666"],
            "forecast_start_date": "2026-07-27",
            "forecast_horizon_days": 14,
            "scenario_overrides": {},
        },
    )
    assert response.status_code == 200
    assert response.json()["is_mock"] is True
    assert response.json()["model_name"] == "deterministic-mock"


def test_prediction_route_returns_structured_model_not_ready(app, client) -> None:
    class UnavailablePredictionService:
        def predict(self, request, user):
            raise ModelNotReadyError()

    app.dependency_overrides[get_prediction_service] = (
        lambda: UnavailablePredictionService()
    )
    response = client.post(
        "/api/v1/predictions",
        json={
            "warehouse_id": "33333333-3333-3333-3333-333333333333",
            "product_ids": ["66666666-6666-6666-6666-666666666666"],
            "forecast_start_date": "2026-07-27",
            "forecast_horizon_days": 14,
        },
    )
    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "MODEL_NOT_READY",
            "message": "The trained prediction model is not available yet.",
            "details": {},
        }
    }
