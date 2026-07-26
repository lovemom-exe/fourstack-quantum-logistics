"""Version 1 API router."""

from fastapi import APIRouter

from app.api.v1 import (
    datasets,
    health,
    ingestion,
    mappings,
    models,
    predictions,
    warehouses,
)


api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(warehouses.router)
api_router.include_router(datasets.router)
api_router.include_router(mappings.router)
api_router.include_router(ingestion.router)
api_router.include_router(models.router)
api_router.include_router(predictions.router)
