"""Model artifact status contracts."""

from pydantic import Field

from app.schemas.common import APIModel


class ModelStatusResponse(APIModel):
    ready: bool
    model_type: str = "vqr"
    artifact_directory: str
    missing_files: list[str] = Field(default_factory=list)
    model_name: str | None = None
    model_version: str | None = None
    target: str | None = None
    feature_count: int | None = None
    selected_features: list[str] = Field(default_factory=list)
