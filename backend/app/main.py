"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.dependencies import get_artifact_service
from app.api.v1.health import health
from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = settings or get_settings()
    configure_logging()
    application = FastAPI(
        title=configured.app_name,
        version="1.0.0",
        description=(
            "CSV onboarding, Supabase persistence, artifact readiness, "
            "and perishable-goods demand prediction orchestration."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[configured.frontend_origin],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )
    register_exception_handlers(application)
    application.include_router(api_router, prefix=configured.api_v1_prefix)

    @application.get("/", tags=["Health"], include_in_schema=False)
    def root() -> dict[str, str]:
        return {"message": configured.app_name}

    @application.get("/health", tags=["Health"])
    def root_health() -> dict[str, object]:
        artifacts = get_artifact_service(configured)
        return {
            "status": "ok",
            "database_configured": configured.supabase_configured,
            "storage_configured": configured.storage_configured,
            "model_ready": artifacts.status().ready,
        }

    return application


app = create_app()
