"""Confirmed-mapping CSV ingestion in bounded batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from uuid import UUID

import pandas as pd

from app.core.exceptions import (
    DatasetNotFoundError,
    IngestionError,
    MappingRequiredError,
)
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.inventory_repository import InventoryRepository
from app.repositories.mapping_repository import MappingRepository
from app.repositories.product_repository import ProductRepository
from app.repositories.sales_repository import SalesRepository
from app.repositories.store_repository import StoreRepository
from app.repositories.supplier_repository import SupplierRepository
from app.repositories.warehouse_repository import WarehouseRepository
from app.schemas.common import CurrentUser
from app.schemas.dataset import DatasetStatus, DatasetType
from app.schemas.ingestion import IngestionResponse, RowFailure
from app.services.csv_service import CSVService
from app.services.mapping_service import MappingService
from app.services.storage_service import StorageService
from app.services.validation_service import ValidationService
from app.utils.dataframe import python_scalar


CANONICAL_FIELDS = {
    "product_id",
    "product_name",
    "category",
    "supplier_id",
    "shelf_life_days",
    "storage_temp",
    "spoilage_sensitivity",
    "base_price",
    "cost_price",
    "supplier_name",
    "supplier_score",
    "lead_time_days",
    "minimum_order_quantity",
    "supplier_country",
    "warehouse_id",
    "warehouse_code",
    "warehouse_name",
    "warehouse_type",
    "country",
    "region",
    "city",
    "storage_capacity",
    "capacity_unit",
    "timezone",
    "store_id",
    "transaction_date",
    "units_sold",
    "selling_price",
    "discount_pct",
    "is_promoted",
    "spoilage_risk",
    "snapshot_date",
    "current_inventory",
    "reserved_inventory",
    "incoming_inventory",
    "expiration_date",
    "batch_id",
}


class IngestionService:
    def __init__(
        self,
        *,
        datasets: DatasetRepository,
        mappings: MappingRepository,
        storage: StorageService,
        csv: CSVService,
        validator: ValidationService,
        products: ProductRepository,
        suppliers: SupplierRepository,
        warehouses: WarehouseRepository,
        stores: StoreRepository,
        sales: SalesRepository,
        inventory: InventoryRepository,
        batch_size: int,
        failure_sample_limit: int = 20,
    ) -> None:
        self.datasets = datasets
        self.mappings = mappings
        self.storage = storage
        self.csv = csv
        self.validator = validator
        self.products = products
        self.suppliers = suppliers
        self.warehouses = warehouses
        self.stores = stores
        self.sales = sales
        self.inventory = inventory
        self.batch_size = batch_size
        self.failure_sample_limit = failure_sample_limit

    def ingest(self, dataset_id: UUID, user: CurrentUser) -> IngestionResponse:
        dataset = self.datasets.get_for_organization(
            dataset_id, user.organization_id
        )
        if dataset is None:
            raise DatasetNotFoundError()
        if dataset.get("status") == DatasetStatus.INGESTED.value:
            return IngestionResponse(
                dataset_id=dataset_id,
                status=DatasetStatus.INGESTED.value,
                skipped_count=int(dataset.get("row_count", 0)),
            )
        mappings = self.mappings.list_for_dataset(
            dataset_id, user.organization_id
        )
        dataset_type = DatasetType(str(dataset["dataset_type"]))
        missing = MappingService.missing_required(mappings, dataset_type)
        if missing:
            raise MappingRequiredError(details={"missing_fields": missing})
        frame = self.csv.parse(
            self.storage.download(str(dataset["storage_path"]))
        )
        validation = self.validator.validate(
            dataset_id=dataset_id,
            frame=frame,
            dataset_type=dataset_type,
            mappings=mappings,
        )
        if not validation.valid:
            raise IngestionError(
                "Validation errors must be resolved before ingestion.",
                details={"error_count": validation.error_count},
            )
        normalized, _ = MappingService.apply_confirmed(frame, mappings)
        dataset_metadata = dataset.get("metadata")
        configured_warehouse_id = None
        if isinstance(dataset_metadata, Mapping):
            configured_value = dataset_metadata.get("warehouse_id")
            if configured_value:
                configured_warehouse_id = UUID(str(configured_value))
        self.datasets.update_status(
            dataset_id, user.organization_id, DatasetStatus.INGESTING.value
        )
        inserted = 0
        failed = 0
        failures: list[RowFailure] = []
        try:
            for start in range(0, len(normalized), self.batch_size):
                batch = normalized.iloc[start : start + self.batch_size]
                payloads: list[dict[str, object]] = []
                for index, row in batch.iterrows():
                    try:
                        payloads.append(
                            self._row_payload(
                                row=row,
                                source_row_number=int(index) + 2,
                                dataset_id=dataset_id,
                                organization_id=user.organization_id,
                                dataset_type=dataset_type,
                                configured_warehouse_id=configured_warehouse_id,
                            )
                        )
                    except (ValueError, KeyError) as exc:
                        failed += 1
                        if len(failures) < self.failure_sample_limit:
                            failures.append(
                                RowFailure(
                                    row_number=int(index) + 2,
                                    code="ROW_CONVERSION_FAILED",
                                    message=str(exc),
                                )
                            )
                inserted += len(self._persist(dataset_type, payloads))
            final_status = (
                DatasetStatus.INGESTED.value
                if failed == 0
                else DatasetStatus.FAILED.value
            )
            self.datasets.update_status(
                dataset_id, user.organization_id, final_status
            )
        except Exception as exc:
            self.datasets.update_status(
                dataset_id, user.organization_id, DatasetStatus.FAILED.value
            )
            raise IngestionError("A database batch failed during ingestion.") from exc
        return IngestionResponse(
            dataset_id=dataset_id,
            status=final_status,
            inserted_count=inserted,
            failed_count=failed,
            failures=failures,
        )

    def _persist(
        self,
        dataset_type: DatasetType,
        payloads: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        if dataset_type == DatasetType.PRODUCTS:
            return self.products.upsert_stable(payloads)
        if dataset_type == DatasetType.SUPPLIERS:
            return self.suppliers.upsert_stable(payloads)
        if dataset_type == DatasetType.WAREHOUSE:
            return self.warehouses.upsert_stable(payloads)
        if dataset_type == DatasetType.SALES:
            return self.sales.upsert_source_rows(payloads)
        if dataset_type == DatasetType.INVENTORY:
            return self.inventory.upsert_source_rows(payloads)
        raise IngestionError("Custom datasets require a project-specific ingestion handler.")

    def _row_payload(
        self,
        *,
        row: pd.Series,
        source_row_number: int,
        dataset_id: UUID,
        organization_id: UUID,
        dataset_type: DatasetType,
        configured_warehouse_id: UUID | None = None,
    ) -> dict[str, object]:
        values = {
            str(column): python_scalar(value)
            for column, value in row.items()
        }
        extra = {
            key: value
            for key, value in values.items()
            if key not in CANONICAL_FIELDS
        }
        base: dict[str, object] = {
            "organization_id": str(organization_id),
        }
        if dataset_type == DatasetType.PRODUCTS:
            external_id = self._required_text(values, "product_id")
            supplier_uuid = None
            if values.get("supplier_id"):
                supplier = self.suppliers.get_by_external_id(
                    str(values["supplier_id"]), organization_id
                )
                if supplier is None:
                    raise ValueError("Unknown supplier_id.")
                supplier_uuid = supplier["id"]
            return {
                **base,
                "external_product_id": external_id,
                "product_name": self._required_text(values, "product_name"),
                "category": values.get("category"),
                "supplier_id": supplier_uuid,
                "shelf_life_days": values.get("shelf_life_days"),
                "storage_temp": values.get("storage_temp"),
                "spoilage_sensitivity": values.get("spoilage_sensitivity"),
                "base_price": values.get("base_price"),
                "cost_price": values.get("cost_price"),
                "metadata": extra,
            }
        if dataset_type == DatasetType.SUPPLIERS:
            return {
                **base,
                "external_supplier_id": self._required_text(values, "supplier_id"),
                "supplier_name": self._required_text(values, "supplier_name"),
                "supplier_score": values.get("supplier_score"),
                "lead_time_days": values.get("lead_time_days"),
                "minimum_order_quantity": values.get("minimum_order_quantity"),
                "supplier_country": values.get("supplier_country"),
                "metadata": extra,
            }
        if dataset_type == DatasetType.WAREHOUSE:
            return {
                **base,
                "warehouse_code": self._required_text(values, "warehouse_code"),
                "warehouse_name": self._required_text(values, "warehouse_name"),
                "warehouse_type": self._required_text(values, "warehouse_type"),
                "country": values.get("country"),
                "region": values.get("region"),
                "city": values.get("city"),
                "storage_capacity": values.get("storage_capacity"),
                "capacity_unit": values.get("capacity_unit"),
                "timezone": values.get("timezone") or "UTC",
                "metadata": extra,
            }
        product = self.products.get_by_external_id(
            self._required_text(values, "product_id"), organization_id
        )
        if product is None:
            raise ValueError("Unknown product_id.")
        warehouse = None
        if values.get("warehouse_id"):
            warehouse_value = str(values["warehouse_id"])
            try:
                warehouse_uuid = UUID(warehouse_value)
            except ValueError:
                warehouse_uuid = None
            warehouse = (
                self.warehouses.get_for_organization(
                    warehouse_uuid, organization_id
                )
                if warehouse_uuid is not None
                else self.warehouses.get_by_code(
                    warehouse_value, organization_id
                )
            )
            if warehouse is None:
                raise ValueError("Unknown warehouse_id.")
        elif configured_warehouse_id is not None:
            warehouse = self.warehouses.get_for_organization(
                configured_warehouse_id, organization_id
            )
            if warehouse is None:
                raise ValueError("Configured warehouse is unavailable.")
        if dataset_type == DatasetType.SALES:
            store_uuid = None
            if values.get("store_id"):
                external_store_id = str(values["store_id"])
                store = self.stores.get_by_external_id(
                    external_store_id, organization_id
                )
                if store is None:
                    created = self.stores.upsert_stable(
                        [
                            {
                                "organization_id": str(organization_id),
                                "warehouse_id": warehouse["id"] if warehouse else None,
                                "external_store_id": external_store_id,
                                "store_name": external_store_id,
                                "metadata": {"generated_from_dataset": str(dataset_id)},
                            }
                        ]
                    )
                    store = created[0] if created else None
                if store is None:
                    raise ValueError("store_id could not be resolved.")
                store_uuid = store["id"]
            return {
                **base,
                "dataset_id": str(dataset_id),
                "product_id": product["id"],
                "store_id": store_uuid,
                "warehouse_id": warehouse["id"] if warehouse else None,
                "transaction_date": self._date_text(values, "transaction_date"),
                "units_sold": values.get("units_sold"),
                "selling_price": values.get("selling_price"),
                "discount_pct": values.get("discount_pct"),
                "is_promoted": self._boolean(values.get("is_promoted")),
                "spoilage_risk": values.get("spoilage_risk"),
                "source_row_number": source_row_number,
                "extra_features": extra,
            }
        if warehouse is None:
            raise ValueError("warehouse_id is required.")
        return {
            **base,
            "dataset_id": str(dataset_id),
            "product_id": product["id"],
            "warehouse_id": warehouse["id"],
            "snapshot_date": self._date_text(values, "snapshot_date"),
            "current_inventory": values.get("current_inventory"),
            "reserved_inventory": values.get("reserved_inventory"),
            "incoming_inventory": values.get("incoming_inventory"),
            "expiration_date": self._date_text(
                values, "expiration_date", required=False
            ),
            "batch_id": values.get("batch_id"),
            "spoilage_risk": values.get("spoilage_risk"),
            "source_row_number": source_row_number,
            "extra_features": extra,
        }

    @staticmethod
    def _required_text(values: Mapping[str, object], key: str) -> str:
        value = values.get(key)
        if value is None or not str(value).strip():
            raise ValueError(f"{key} is required.")
        return str(value).strip()

    @staticmethod
    def _date_text(
        values: Mapping[str, object], key: str, *, required: bool = True
    ) -> str | None:
        value = values.get(key)
        if value is None and not required:
            return None
        parsed = pd.to_datetime(value, errors="coerce", format="mixed")
        if pd.isna(parsed):
            raise ValueError(f"{key} is not a valid date.")
        return parsed.date().isoformat()

    @staticmethod
    def _boolean(value: object) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
        raise ValueError("Boolean value is invalid.")
