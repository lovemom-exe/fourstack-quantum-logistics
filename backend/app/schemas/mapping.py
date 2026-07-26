"""Column mapping contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from app.schemas.common import APIModel


class MappingType(StrEnum):
    EXACT = "exact"
    ALIAS = "alias"
    MANUAL = "manual"
    GENERATED = "generated"
    DEFAULT = "default"
    IGNORED = "ignored"
    IDENTIFIER = "identifier"
    CUSTOM_FEATURE = "custom_feature"


class MappingSuggestion(APIModel):
    source_column: str
    suggested_target: str | None = None
    confidence: float = Field(ge=0, le=1)
    mapping_type: MappingType
    is_confirmed: bool = False


class ColumnMappingUpdate(APIModel):
    source_column: str
    target_field: str | None = None
    mapping_type: MappingType
    is_confirmed: bool
    default_value: object | None = None


class ColumnMappingsUpdate(APIModel):
    mappings: list[ColumnMappingUpdate]


class ColumnMappingResponse(APIModel):
    id: UUID
    organization_id: UUID
    dataset_id: UUID
    source_column: str
    target_field: str | None = None
    mapping_type: MappingType
    confidence: float
    is_confirmed: bool
    default_value: object | None = None
    created_at: datetime
    updated_at: datetime
