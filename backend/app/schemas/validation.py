"""Structured dataset validation contracts."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field

from app.schemas.common import APIModel


class ValidationIssue(APIModel):
    severity: Literal["error", "warning", "info"]
    code: str
    message: str
    column: str | None = None
    row_number: int | None = None
    value: object | None = None


class DatasetValidationResponse(APIModel):
    dataset_id: UUID
    valid: bool
    error_count: int
    warning_count: int
    issues: list[ValidationIssue] = Field(default_factory=list)
