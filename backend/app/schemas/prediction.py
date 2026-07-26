"""Prediction request, job, and forecast contracts."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, field_validator

from app.schemas.common import APIModel


class PredictionRequest(APIModel):
    dataset_id: UUID | None = None
    warehouse_id: UUID
    store_id: UUID | None = None
    product_ids: list[UUID] = Field(min_length=1)
    forecast_start_date: date
    forecast_horizon_days: Literal[14, 30, 60, 90]
    scenario_overrides: dict[str, object] = Field(default_factory=dict)

    @field_validator("product_ids")
    @classmethod
    def unique_products(cls, value: list[UUID]) -> list[UUID]:
        if len(value) != len(set(value)):
            raise ValueError("product_ids must not contain duplicates.")
        return value


class ForecastPointResponse(APIModel):
    product_id: UUID | None = None
    warehouse_id: UUID | None = None
    store_id: UUID | None = None
    forecast_date: date
    predicted_units_sold: int = Field(ge=0)
    lower_bound: int | None = Field(default=None, ge=0)
    upper_bound: int | None = Field(default=None, ge=0)


class PredictionResponse(APIModel):
    job_id: UUID
    status: str
    model_name: str
    model_version: str
    target: str
    forecast_horizon_days: int
    is_mock: bool
    results: list[ForecastPointResponse]


class PredictionJobResponse(APIModel):
    id: UUID
    organization_id: UUID
    status: str
    is_mock: bool
    forecast_start_date: date
    forecast_horizon_days: int
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
