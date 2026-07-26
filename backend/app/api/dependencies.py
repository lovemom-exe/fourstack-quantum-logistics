"""FastAPI dependency providers."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.clients.supabase_client import (
    SupabaseClientProtocol,
    get_supabase_client,
)
from app.core.config import Settings, get_settings
from app.core.exceptions import AuthenticationError
from app.core.security import resolve_current_user
from app.model_adapters.mock_adapter import MockModelAdapter
from app.model_adapters.vqr_adapter import VQRModelAdapter
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.inventory_repository import InventoryRepository
from app.repositories.mapping_repository import MappingRepository
from app.repositories.prediction_repository import PredictionRepository
from app.repositories.product_repository import ProductRepository
from app.repositories.sales_repository import SalesRepository
from app.repositories.store_repository import StoreRepository
from app.repositories.supplier_repository import SupplierRepository
from app.repositories.warehouse_repository import WarehouseRepository
from app.schemas.common import CurrentUser
from app.services.csv_service import CSVService
from app.services.dataset_mapping_service import DatasetMappingService
from app.services.dataset_service import DatasetService
from app.services.dataset_validation_service import (
    DatasetValidationOrchestrator,
)
from app.services.feature_service import FeatureService
from app.services.ingestion_service import IngestionService
from app.services.mapping_service import MappingService
from app.services.model_artifact_service import ModelArtifactService
from app.services.prediction_service import PredictionService
from app.services.readiness_service import ReadinessOrchestrator
from app.services.storage_service import StorageService
from app.services.validation_service import ValidationService
from app.services.warehouse_service import WarehouseService


bearer_scheme = HTTPBearer(auto_error=False)


def get_client() -> SupabaseClientProtocol:
    return get_supabase_client()


def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ],
    client: Annotated[SupabaseClientProtocol, Depends(get_client)],
) -> CurrentUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise AuthenticationError()
    return resolve_current_user(client, credentials.credentials)


def get_artifact_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ModelArtifactService:
    return ModelArtifactService(
        settings.resolved_model_artifact_dir,
        settings.public_model_artifact_dir,
    )


def get_csv_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> CSVService:
    return CSVService(settings.max_csv_size_mb, settings.csv_preview_limit)


def get_storage_service(
    client: Annotated[SupabaseClientProtocol, Depends(get_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> StorageService:
    return StorageService(client, settings.supabase_storage_bucket)


def get_dataset_repository(
    client: Annotated[SupabaseClientProtocol, Depends(get_client)],
) -> DatasetRepository:
    return DatasetRepository(client)


def get_mapping_repository(
    client: Annotated[SupabaseClientProtocol, Depends(get_client)],
) -> MappingRepository:
    return MappingRepository(client)


def get_prediction_repository(
    client: Annotated[SupabaseClientProtocol, Depends(get_client)],
) -> PredictionRepository:
    return PredictionRepository(client)


def get_warehouse_repository(
    client: Annotated[SupabaseClientProtocol, Depends(get_client)],
) -> WarehouseRepository:
    return WarehouseRepository(client)


def get_dataset_service(
    datasets: Annotated[DatasetRepository, Depends(get_dataset_repository)],
    mappings: Annotated[MappingRepository, Depends(get_mapping_repository)],
    storage: Annotated[StorageService, Depends(get_storage_service)],
    csv: Annotated[CSVService, Depends(get_csv_service)],
) -> DatasetService:
    return DatasetService(
        repository=datasets,
        mapping_repository=mappings,
        storage=storage,
        csv=csv,
        mapping=MappingService(),
    )


def get_mapping_orchestrator(
    datasets: Annotated[DatasetRepository, Depends(get_dataset_repository)],
    mappings: Annotated[MappingRepository, Depends(get_mapping_repository)],
) -> DatasetMappingService:
    return DatasetMappingService(datasets, mappings, MappingService())


def get_validation_orchestrator(
    client: Annotated[SupabaseClientProtocol, Depends(get_client)],
    datasets: Annotated[DatasetRepository, Depends(get_dataset_repository)],
    mappings: Annotated[MappingRepository, Depends(get_mapping_repository)],
    storage: Annotated[StorageService, Depends(get_storage_service)],
    csv: Annotated[CSVService, Depends(get_csv_service)],
    artifacts: Annotated[ModelArtifactService, Depends(get_artifact_service)],
) -> DatasetValidationOrchestrator:
    return DatasetValidationOrchestrator(
        datasets=datasets,
        mappings=mappings,
        storage=storage,
        csv=csv,
        validator=ValidationService(),
        artifacts=artifacts,
        products=ProductRepository(client),
        suppliers=SupplierRepository(client),
        warehouses=WarehouseRepository(client),
        stores=StoreRepository(client),
    )


def get_readiness_orchestrator(
    datasets: Annotated[DatasetRepository, Depends(get_dataset_repository)],
    mappings: Annotated[MappingRepository, Depends(get_mapping_repository)],
    warehouses: Annotated[
        WarehouseRepository, Depends(get_warehouse_repository)
    ],
    artifacts: Annotated[ModelArtifactService, Depends(get_artifact_service)],
) -> ReadinessOrchestrator:
    return ReadinessOrchestrator(
        datasets=datasets,
        mappings=mappings,
        warehouses=warehouses,
        artifacts=artifacts,
    )


def get_warehouse_service(
    repository: Annotated[WarehouseRepository, Depends(get_warehouse_repository)],
) -> WarehouseService:
    return WarehouseService(repository)


def get_ingestion_service(
    client: Annotated[SupabaseClientProtocol, Depends(get_client)],
    datasets: Annotated[DatasetRepository, Depends(get_dataset_repository)],
    mappings: Annotated[MappingRepository, Depends(get_mapping_repository)],
    storage: Annotated[StorageService, Depends(get_storage_service)],
    csv: Annotated[CSVService, Depends(get_csv_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> IngestionService:
    return IngestionService(
        datasets=datasets,
        mappings=mappings,
        storage=storage,
        csv=csv,
        validator=ValidationService(),
        products=ProductRepository(client),
        suppliers=SupplierRepository(client),
        warehouses=WarehouseRepository(client),
        stores=StoreRepository(client),
        sales=SalesRepository(client),
        inventory=InventoryRepository(client),
        batch_size=settings.csv_insert_batch_size,
    )


def get_prediction_service(
    client: Annotated[SupabaseClientProtocol, Depends(get_client)],
    settings: Annotated[Settings, Depends(get_settings)],
    artifacts: Annotated[ModelArtifactService, Depends(get_artifact_service)],
) -> PredictionService:
    return PredictionService(
        settings=settings,
        artifacts=artifacts,
        features=FeatureService(),
        predictions=PredictionRepository(client),
        datasets=DatasetRepository(client),
        mappings=MappingRepository(client),
        products=ProductRepository(client),
        suppliers=SupplierRepository(client),
        warehouses=WarehouseRepository(client),
        stores=StoreRepository(client),
        sales=SalesRepository(client),
        inventory=InventoryRepository(client),
        vqr_adapter=VQRModelAdapter(
            settings.resolved_model_artifact_dir,
            artifact_service=artifacts,
        ),
        mock_adapter=MockModelAdapter(),
    )
