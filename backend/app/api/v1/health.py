"""Health endpoints that do not require a trained model or authentication."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_artifact_service
from app.core.config import Settings, get_settings
from app.services.model_artifact_service import ModelArtifactService


router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Check backend dependency configuration",
)
def health(
    settings: Annotated[Settings, Depends(get_settings)],
    artifacts: Annotated[ModelArtifactService, Depends(get_artifact_service)],
) -> dict[str, object]:
    return {
        "status": "ok",
        "database_configured": settings.supabase_configured,
        "storage_configured": settings.storage_configured,
        "model_ready": artifacts.status().ready,
    }
