"""Confirmed dataset ingestion route."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends

from app.api.dependencies import get_current_user, get_ingestion_service
from app.schemas.common import CurrentUser
from app.schemas.ingestion import IngestionResponse
from app.services.ingestion_service import IngestionService


router = APIRouter(prefix="/datasets", tags=["Ingestion"])


@router.post(
    "/{dataset_id}/ingest",
    response_model=IngestionResponse,
    summary="Ingest a validated CSV in bounded database batches",
)
def ingest_dataset(
    dataset_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
) -> IngestionResponse:
    return service.ingest(dataset_id, user)
