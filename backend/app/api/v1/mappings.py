"""Column mapping routes."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends

from app.api.dependencies import get_current_user, get_mapping_orchestrator
from app.schemas.common import CurrentUser
from app.schemas.mapping import (
    ColumnMappingResponse,
    ColumnMappingsUpdate,
)
from app.services.dataset_mapping_service import DatasetMappingService


router = APIRouter(prefix="/datasets", tags=["Mappings"])


@router.post(
    "/{dataset_id}/auto-map",
    response_model=list[ColumnMappingResponse],
    summary="Regenerate unconfirmed mapping suggestions",
)
def auto_map(
    dataset_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[
        DatasetMappingService, Depends(get_mapping_orchestrator)
    ],
) -> list[ColumnMappingResponse]:
    return service.auto_map(dataset_id, user)


@router.get(
    "/{dataset_id}/mappings",
    response_model=list[ColumnMappingResponse],
    summary="List dataset column mappings",
)
def list_mappings(
    dataset_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[
        DatasetMappingService, Depends(get_mapping_orchestrator)
    ],
) -> list[ColumnMappingResponse]:
    return service.list(dataset_id, user)


@router.put(
    "/{dataset_id}/mappings",
    response_model=list[ColumnMappingResponse],
    summary="Replace dataset mappings with reviewed mappings",
)
def update_mappings(
    dataset_id: UUID,
    payload: ColumnMappingsUpdate,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[
        DatasetMappingService, Depends(get_mapping_orchestrator)
    ],
) -> list[ColumnMappingResponse]:
    return service.update(dataset_id, payload, user)
