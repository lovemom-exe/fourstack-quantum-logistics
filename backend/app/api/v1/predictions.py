"""Prediction job routes."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends

from app.api.dependencies import (
    get_current_user,
    get_prediction_repository,
    get_prediction_service,
)
from app.core.exceptions import AuthorizationError
from app.repositories.prediction_repository import PredictionRepository
from app.schemas.common import CurrentUser
from app.schemas.prediction import (
    ForecastPointResponse,
    PredictionJobResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services.prediction_service import PredictionService


router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Generate and persist a synchronous forecast job",
)
def create_prediction(
    payload: PredictionRequest,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> PredictionResponse:
    return service.predict(payload, user)


@router.get(
    "/{job_id}",
    response_model=PredictionJobResponse,
    summary="Get an organization prediction job",
)
def get_prediction_job(
    job_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    repository: Annotated[
        PredictionRepository, Depends(get_prediction_repository)
    ],
) -> PredictionJobResponse:
    row = repository.get_for_organization(job_id, user.organization_id)
    if row is None:
        raise AuthorizationError("Prediction job was not found in your organization.")
    return PredictionJobResponse.model_validate(row)


@router.get(
    "/{job_id}/results",
    response_model=list[ForecastPointResponse],
    summary="Get persisted forecast points for a job",
)
def get_prediction_results(
    job_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    repository: Annotated[
        PredictionRepository, Depends(get_prediction_repository)
    ],
) -> list[ForecastPointResponse]:
    rows = repository.list_results(job_id, user.organization_id)
    if not rows:
        job = repository.get_for_organization(job_id, user.organization_id)
        if job is None:
            raise AuthorizationError(
                "Prediction job was not found in your organization."
            )
    return [
        ForecastPointResponse(
            product_id=row.get("product_id"),
            warehouse_id=row.get("warehouse_id"),
            store_id=row.get("store_id"),
            forecast_date=row["forecast_date"],
            predicted_units_sold=int(round(float(row["predicted_units_sold"]))),
            lower_bound=int(round(float(row["lower_bound"])))
            if row.get("lower_bound") is not None
            else None,
            upper_bound=int(round(float(row["upper_bound"])))
            if row.get("upper_bound") is not None
            else None,
        )
        for row in rows
    ]
