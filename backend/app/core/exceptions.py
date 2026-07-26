"""Application exceptions and FastAPI handlers."""

from __future__ import annotations

from collections.abc import Mapping
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


logger = logging.getLogger(__name__)


class AppError(Exception):
    """Safe error that can cross the HTTP boundary."""

    status_code = 400
    code = "APPLICATION_ERROR"
    default_message = "The request could not be completed."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message or self.default_message)
        self.message = message or self.default_message
        self.details = dict(details or {})


class ConfigurationError(AppError):
    status_code = 503
    code = "CONFIGURATION_ERROR"
    default_message = "A required backend service is not configured."


class AuthenticationError(AppError):
    status_code = 401
    code = "AUTHENTICATION_REQUIRED"
    default_message = "A valid bearer token is required."


class AuthorizationError(AppError):
    status_code = 403
    code = "FORBIDDEN"
    default_message = "You do not have access to this resource."


class DatasetNotFoundError(AppError):
    status_code = 404
    code = "DATASET_NOT_FOUND"
    default_message = "The requested dataset was not found."


class DatasetValidationError(AppError):
    status_code = 422
    code = "DATASET_VALIDATION_FAILED"
    default_message = "The dataset contains validation errors."


class MappingRequiredError(AppError):
    status_code = 409
    code = "MAPPING_REQUIRED"
    default_message = "Confirmed column mappings are required."


class IngestionError(AppError):
    status_code = 422
    code = "INGESTION_ERROR"
    default_message = "The dataset could not be ingested."


class ModelNotReadyError(AppError):
    status_code = 503
    code = "MODEL_NOT_READY"
    default_message = "The trained prediction model is not available yet."


class ModelArtifactError(AppError):
    status_code = 503
    code = "MODEL_ARTIFACT_ERROR"
    default_message = "The prediction model artifacts are invalid."


class ModelFeatureMismatchError(AppError):
    status_code = 422
    code = "MODEL_FEATURE_MISMATCH"
    default_message = "The resolved features do not match the saved model schema."


class FeatureResolutionError(AppError):
    status_code = 422
    code = "FEATURE_RESOLUTION_ERROR"
    default_message = "Required model features could not be resolved."


class PredictionError(AppError):
    status_code = 500
    code = "PREDICTION_ERROR"
    default_message = "The forecast could not be generated."


def register_exception_handlers(app: FastAPI) -> None:
    """Register a consistent safe error envelope."""

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        issues = [
            {
                "location": [str(part) for part in error["loc"]],
                "message": str(error["msg"]),
                "type": str(error["type"]),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "REQUEST_VALIDATION_ERROR",
                    "message": "The request payload is invalid.",
                    "details": {"issues": issues},
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        _: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "HTTP_ERROR",
                    "message": str(exc.detail),
                    "details": {},
                }
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled backend exception type=%s", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected backend error occurred.",
                    "details": {},
                }
            },
        )
