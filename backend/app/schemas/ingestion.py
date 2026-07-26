"""Dataset ingestion result contracts."""

from uuid import UUID

from pydantic import Field

from app.schemas.common import APIModel


class RowFailure(APIModel):
    row_number: int
    code: str
    message: str


class IngestionResponse(APIModel):
    dataset_id: UUID
    status: str
    inserted_count: int = 0
    updated_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    failures: list[RowFailure] = Field(default_factory=list)
