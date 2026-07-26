"""Dataset and forecasting readiness calculation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import UUID

from app.schemas.dataset import DatasetReadinessResponse
from app.schemas.model import ModelStatusResponse
from app.core.exceptions import DatasetNotFoundError
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.mapping_repository import MappingRepository
from app.repositories.warehouse_repository import WarehouseRepository
from app.schemas.common import CurrentUser
from app.services.model_artifact_service import ModelArtifactService


class ReadinessService:
    def evaluate(
        self,
        *,
        dataset_id: UUID,
        dataset: Mapping[str, object],
        organization_datasets: Sequence[Mapping[str, object]],
        mappings: Sequence[Mapping[str, object]],
        warehouse_exists: bool,
        model_status: ModelStatusResponse,
    ) -> DatasetReadinessResponse:
        ingested_types = {
            str(item.get("dataset_type"))
            for item in organization_datasets
            if item.get("status") == "ingested"
        }
        mappings_ready = bool(mappings) and all(
            bool(item.get("is_confirmed")) for item in mappings
        )
        status = str(dataset.get("status", ""))
        validation_ready = status in {"validated", "ready", "ingesting", "ingested"}
        ingestion_ready = status == "ingested"
        warnings: list[str] = []
        if not model_status.ready:
            warnings.append("The VQR model artifacts are still unavailable.")
        if not mappings_ready:
            warnings.append("Column mappings are not fully confirmed.")
        if not ingestion_ready:
            warnings.append("The selected dataset has not completed ingestion.")
        prerequisites = {
            "products": "products" in ingested_types,
            "suppliers": "suppliers" in ingested_types,
            "sales": "sales" in ingested_types,
            "inventory": "inventory" in ingested_types,
        }
        forecast_ready = (
            warehouse_exists
            and all(prerequisites.values())
            and mappings_ready
            and validation_ready
            and ingestion_ready
            and model_status.ready
        )
        return DatasetReadinessResponse(
            dataset_id=dataset_id,
            warehouse_ready=warehouse_exists,
            products_ready=prerequisites["products"],
            suppliers_ready=prerequisites["suppliers"],
            sales_ready=prerequisites["sales"],
            inventory_ready=prerequisites["inventory"],
            mappings_ready=mappings_ready,
            validation_ready=validation_ready,
            ingestion_ready=ingestion_ready,
            model_ready=model_status.ready,
            forecast_ready=forecast_ready,
            warnings=warnings,
        )


class ReadinessOrchestrator:
    def __init__(
        self,
        *,
        datasets: DatasetRepository,
        mappings: MappingRepository,
        warehouses: WarehouseRepository,
        artifacts: ModelArtifactService,
    ) -> None:
        self.datasets = datasets
        self.mappings = mappings
        self.warehouses = warehouses
        self.artifacts = artifacts
        self.calculator = ReadinessService()

    def for_dataset(
        self, dataset_id: UUID, user: CurrentUser
    ) -> DatasetReadinessResponse:
        dataset = self.datasets.get_for_organization(
            dataset_id, user.organization_id
        )
        if dataset is None:
            raise DatasetNotFoundError()
        organization_datasets = self.datasets.list_for_organization(
            user.organization_id
        )
        mappings = self.mappings.list_for_dataset(
            dataset_id, user.organization_id
        )
        warehouses = self.warehouses.list_for_organization(user.organization_id)
        return self.calculator.evaluate(
            dataset_id=dataset_id,
            dataset=dataset,
            organization_datasets=organization_datasets,
            mappings=mappings,
            warehouse_exists=bool(warehouses),
            model_status=self.artifacts.status(),
        )
