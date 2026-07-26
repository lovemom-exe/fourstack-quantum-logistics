"""CSV upload, metadata, preview, deletion, validation, and readiness routes."""

from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Query,
    UploadFile,
    status,
)

from app.api.dependencies import (
    get_current_user,
    get_dataset_service,
    get_readiness_orchestrator,
    get_validation_orchestrator,
)
from app.schemas.common import CurrentUser, DeleteResponse
from app.schemas.dataset import (
    DatasetColumnResponse,
    DatasetPreviewResponse,
    DatasetReadinessResponse,
    DatasetResponse,
    DatasetType,
    DatasetUploadResponse,
)
from app.schemas.validation import DatasetValidationResponse
from app.services.dataset_service import DatasetService
from app.services.dataset_validation_service import (
    DatasetValidationOrchestrator,
)
from app.services.readiness_service import ReadinessOrchestrator


router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.post(
    "/upload",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and inspect a CSV without ingesting business records",
)
async def upload_dataset(
    file: Annotated[UploadFile, File(description="CSV source file")],
    dataset_name: Annotated[str, Form(min_length=1)],
    dataset_type: Annotated[DatasetType, Form()],
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[DatasetService, Depends(get_dataset_service)],
    warehouse_id: Annotated[UUID | None, Form()] = None,
) -> DatasetUploadResponse:
    content = await file.read()
    return service.upload(
        content=content,
        filename=file.filename,
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        warehouse_id=warehouse_id,
        user=user,
    )


@router.get("", response_model=list[DatasetResponse], summary="List datasets")
def list_datasets(
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> list[DatasetResponse]:
    return service.list(user)


@router.get(
    "/{dataset_id}", response_model=DatasetResponse, summary="Get dataset metadata"
)
def get_dataset(
    dataset_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> DatasetResponse:
    return service.get(dataset_id, user)


@router.delete(
    "/{dataset_id}", response_model=DeleteResponse, summary="Delete dataset and CSV"
)
def delete_dataset(
    dataset_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> DeleteResponse:
    return DeleteResponse(id=dataset_id, deleted=service.delete(dataset_id, user))


@router.get(
    "/{dataset_id}/preview",
    response_model=DatasetPreviewResponse,
    summary="Preview a bounded number of CSV rows",
)
def preview_dataset(
    dataset_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[DatasetService, Depends(get_dataset_service)],
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
) -> DatasetPreviewResponse:
    return service.preview(dataset_id, user, limit)


@router.get(
    "/{dataset_id}/columns",
    response_model=list[DatasetColumnResponse],
    summary="List inferred CSV column metadata",
)
def dataset_columns(
    dataset_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[DatasetService, Depends(get_dataset_service)],
) -> list[DatasetColumnResponse]:
    return service.columns(dataset_id, user)


@router.post(
    "/{dataset_id}/validate",
    response_model=DatasetValidationResponse,
    summary="Validate a stored CSV using confirmed mappings",
)
def validate_dataset(
    dataset_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[
        DatasetValidationOrchestrator, Depends(get_validation_orchestrator)
    ],
) -> DatasetValidationResponse:
    return service.validate(dataset_id, user)


@router.get(
    "/{dataset_id}/readiness",
    response_model=DatasetReadinessResponse,
    summary="Report dataset and forecasting readiness",
)
def dataset_readiness(
    dataset_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[
        ReadinessOrchestrator, Depends(get_readiness_orchestrator)
    ],
) -> DatasetReadinessResponse:
    return service.for_dataset(dataset_id, user)
