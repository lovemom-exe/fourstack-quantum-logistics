"""Validation orchestration over stored CSVs and confirmed mappings."""

from __future__ import annotations

from uuid import UUID

from app.core.exceptions import DatasetNotFoundError
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.mapping_repository import MappingRepository
from app.repositories.product_repository import ProductRepository
from app.repositories.store_repository import StoreRepository
from app.repositories.supplier_repository import SupplierRepository
from app.repositories.warehouse_repository import WarehouseRepository
from app.schemas.common import CurrentUser
from app.schemas.dataset import DatasetStatus, DatasetType
from app.schemas.validation import DatasetValidationResponse
from app.services.csv_service import CSVService
from app.services.model_artifact_service import ModelArtifactService
from app.services.storage_service import StorageService
from app.services.validation_service import ValidationService


class DatasetValidationOrchestrator:
    def __init__(
        self,
        *,
        datasets: DatasetRepository,
        mappings: MappingRepository,
        storage: StorageService,
        csv: CSVService,
        validator: ValidationService,
        artifacts: ModelArtifactService,
        products: ProductRepository,
        suppliers: SupplierRepository,
        warehouses: WarehouseRepository,
        stores: StoreRepository,
    ) -> None:
        self.datasets = datasets
        self.mappings = mappings
        self.storage = storage
        self.csv = csv
        self.validator = validator
        self.artifacts = artifacts
        self.products = products
        self.suppliers = suppliers
        self.warehouses = warehouses
        self.stores = stores

    def validate(
        self, dataset_id: UUID, user: CurrentUser
    ) -> DatasetValidationResponse:
        dataset = self.datasets.get_for_organization(
            dataset_id, user.organization_id
        )
        if dataset is None:
            raise DatasetNotFoundError()
        frame = self.csv.parse(
            self.storage.download(str(dataset["storage_path"]))
        )
        mappings = self.mappings.list_for_dataset(
            dataset_id, user.organization_id
        )
        dataset_type = DatasetType(str(dataset["dataset_type"]))
        model_features: list[str] = []
        supported: dict[str, set[str]] = {}
        if not self.artifacts.missing_files:
            schema = self.artifacts.feature_schema()
            feature_value = schema.get("feature_order", [])
            if isinstance(feature_value, list):
                model_features = [str(item) for item in feature_value]
            for group in ("category", "region"):
                encoding = schema.get("categorical_encoding")
                if isinstance(encoding, dict):
                    rule = encoding.get(group)
                    if isinstance(rule, dict) and isinstance(rule.get("categories"), list):
                        supported[group] = {
                            str(item) for item in rule["categories"]
                        }
        reference_catalog = {
            "product_id": {
                str(row["external_product_id"])
                for row in self.products.list_for_organization(
                    user.organization_id
                )
            },
            "supplier_id": {
                str(row["external_supplier_id"])
                for row in self.suppliers.list_for_organization(
                    user.organization_id
                )
            },
            "warehouse_id": {
                str(row["warehouse_code"])
                for row in self.warehouses.list_for_organization(
                    user.organization_id
                )
            },
            "store_id": {
                str(row["external_store_id"])
                for row in self.stores.list_for_organization(
                    user.organization_id
                )
            },
        }
        reference_fields = {
            DatasetType.PRODUCTS: {"supplier_id"},
            DatasetType.SALES: {
                "product_id",
                "supplier_id",
                "warehouse_id",
                "store_id",
            },
            DatasetType.INVENTORY: {"product_id", "warehouse_id"},
        }.get(dataset_type, set())
        result = self.validator.validate(
            dataset_id=dataset_id,
            frame=frame,
            dataset_type=dataset_type,
            mappings=mappings,
            known_references={
                field: reference_catalog[field] for field in reference_fields
            },
            supported_values=supported,
            model_required_features=model_features,
        )
        self.datasets.update_status(
            dataset_id,
            user.organization_id,
            DatasetStatus.VALIDATED.value
            if result.valid
            else DatasetStatus.MAPPING_REQUIRED.value,
        )
        return result
