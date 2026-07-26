"""Warehouse request and response contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator

from app.schemas.common import APIModel


class WarehouseType(StrEnum):
    AMBIENT = "ambient"
    CHILLED = "chilled"
    FROZEN = "frozen"
    MIXED = "mixed"


class WarehouseBase(APIModel):
    warehouse_code: str = Field(min_length=1, max_length=100)
    warehouse_name: str = Field(min_length=1, max_length=255)
    warehouse_type: WarehouseType
    country: str | None = Field(default=None, max_length=100)
    region: str | None = Field(default=None, max_length=100)
    city: str | None = Field(default=None, max_length=100)
    storage_capacity: float | None = Field(default=None, ge=0)
    capacity_unit: str | None = Field(default=None, max_length=50)
    timezone: str = "UTC"
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("warehouse_name", "warehouse_code")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Value cannot be blank.")
        return stripped

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("Timezone must be a valid IANA timezone name.") from exc
        return value


class WarehouseCreate(WarehouseBase):
    pass


class WarehouseUpdate(APIModel):
    warehouse_code: str | None = Field(default=None, min_length=1, max_length=100)
    warehouse_name: str | None = Field(default=None, min_length=1, max_length=255)
    warehouse_type: WarehouseType | None = None
    country: str | None = Field(default=None, max_length=100)
    region: str | None = Field(default=None, max_length=100)
    city: str | None = Field(default=None, max_length=100)
    storage_capacity: float | None = Field(default=None, ge=0)
    capacity_unit: str | None = Field(default=None, max_length=50)
    timezone: str | None = None
    metadata: dict[str, object] | None = None

    @field_validator("timezone")
    @classmethod
    def validate_optional_timezone(cls, value: str | None) -> str | None:
        if value is None:
            return value
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("Timezone must be a valid IANA timezone name.") from exc
        return value


class WarehouseResponse(WarehouseBase):
    id: UUID
    organization_id: UUID
    created_at: datetime
    updated_at: datetime
