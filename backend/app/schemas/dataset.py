"""Dataset upload, preview, and metadata contracts."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from app.schemas.common import APIModel


class DatasetType(StrEnum):
    WAREHOUSE = "warehouse"
    PRODUCTS = "products"
    SUPPLIERS = "suppliers"
    SALES = "sales"
    INVENTORY = "inventory"
    CUSTOM = "custom"


class DatasetStatus(StrEnum):
    UPLOADED = "uploaded"
    MAPPING_REQUIRED = "mapping_required"
    VALIDATED = "validated"
    READY = "ready"
    INGESTING = "ingesting"
    INGESTED = "ingested"
    FAILED = "failed"


class DatasetColumnResponse(APIModel):
    id: UUID
    dataset_id: UUID
    source_column: str
    normalized_column: str
    detected_type: str
    sample_values: list[object] = Field(default_factory=list)
    missing_count: int
    unique_count: int
    created_at: datetime | None = None


class DatasetResponse(APIModel):
    id: UUID
    organization_id: UUID
    dataset_name: str
    dataset_type: DatasetType
    status: DatasetStatus
    original_filename: str
    storage_path: str
    file_size_bytes: int
    row_count: int
    column_count: int
    date_min: date | None = None
    date_max: date | None = None
    uploaded_by: UUID
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class DatasetUploadResponse(DatasetResponse):
    duplicate_row_count: int
    columns: list[DatasetColumnResponse]
    suggested_mapping_count: int


class DatasetPreviewResponse(APIModel):
    dataset_id: UUID
    columns: list[str]
    rows: list[dict[str, object]]
    returned_rows: int
    total_rows: int


class DatasetReadinessResponse(APIModel):
    dataset_id: UUID
    warehouse_ready: bool
    products_ready: bool
    suppliers_ready: bool
    sales_ready: bool
    inventory_ready: bool
    mappings_ready: bool
    validation_ready: bool
    ingestion_ready: bool
    model_ready: bool
    forecast_ready: bool
    warnings: list[str] = Field(default_factory=list)
