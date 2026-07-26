"""Shared API models."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="ignore")


class CurrentUser(APIModel):
    id: UUID
    organization_id: UUID
    email: str | None = None
    role: str = "member"


class MessageResponse(APIModel):
    message: str


class DeleteResponse(APIModel):
    id: UUID
    deleted: bool = True


class ErrorDetail(APIModel):
    code: str
    message: str
    details: dict[str, object] = Field(default_factory=dict)


class ErrorResponse(APIModel):
    error: ErrorDetail


class TimestampedResponse(APIModel):
    id: UUID
    created_at: datetime
    updated_at: datetime
