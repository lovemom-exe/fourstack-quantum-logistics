"""Typed application settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    """Environment-backed settings with safe development defaults."""

    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Perishable Goods Forecast API"
    app_env: Literal["development", "test", "production"] = "development"
    api_v1_prefix: str = "/api/v1"
    frontend_origin: str = "http://localhost:5173"

    supabase_url: str | None = None
    supabase_anon_key: SecretStr | None = None
    supabase_service_role_key: SecretStr | None = None
    supabase_storage_bucket: str = "dataset-files"

    model_artifact_dir: Path = Field(
        default=Path("../ml/models/perishable_vqr")
    )
    allow_mock_predictions: bool = True
    max_csv_size_mb: int = Field(default=100, ge=1, le=1024)
    csv_insert_batch_size: int = Field(default=1000, ge=1, le=10_000)
    csv_preview_limit: int = Field(default=20, ge=1, le=200)

    @model_validator(mode="after")
    def validate_production_configuration(self) -> "Settings":
        if self.app_env == "production" and not self.supabase_configured:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required in production."
            )
        if self.app_env == "production" and self.frontend_origin == "*":
            raise ValueError("FRONTEND_ORIGIN cannot be '*' in production.")
        return self

    @property
    def supabase_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_role_key)

    @property
    def storage_configured(self) -> bool:
        return self.supabase_configured and bool(self.supabase_storage_bucket)

    @property
    def resolved_model_artifact_dir(self) -> Path:
        raw_path = self.model_artifact_dir.expanduser()
        resolved = (
            raw_path.resolve()
            if raw_path.is_absolute()
            else (BACKEND_ROOT / raw_path).resolve()
        )
        if resolved == Path(resolved.anchor):
            raise ValueError("MODEL_ARTIFACT_DIR cannot resolve to a filesystem root.")
        return resolved

    @property
    def public_model_artifact_dir(self) -> str:
        return str(self.model_artifact_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one immutable-by-convention settings instance."""
    return Settings()
