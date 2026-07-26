"""Model artifact status route."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_artifact_service
from app.schemas.model import ModelStatusResponse
from app.services.model_artifact_service import ModelArtifactService


router = APIRouter(prefix="/models", tags=["Models"])


@router.get(
    "/status",
    response_model=ModelStatusResponse,
    summary="Inspect lightweight model artifact readiness",
)
def model_status(
    service: Annotated[ModelArtifactService, Depends(get_artifact_service)],
) -> ModelStatusResponse:
    return service.status()
