"""Synchronous prediction orchestration behind a queue-ready service boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from uuid import UUID

import numpy as np

from app.core.config import Settings
from app.core.exceptions import (
    AuthorizationError,
    DatasetNotFoundError,
    DatasetValidationError,
    FeatureResolutionError,
    ModelNotReadyError,
    PredictionError,
)
from app.model_adapters.base import PredictionModelAdapter
from app.model_adapters.mock_adapter import MockModelAdapter
from app.model_adapters.vqr_adapter import VQRModelAdapter
from app.repositories.inventory_repository import InventoryRepository
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.mapping_repository import MappingRepository
from app.repositories.prediction_repository import PredictionRepository
from app.repositories.product_repository import ProductRepository
from app.repositories.sales_repository import SalesRepository
from app.repositories.store_repository import StoreRepository
from app.repositories.supplier_repository import SupplierRepository
from app.repositories.warehouse_repository import WarehouseRepository
from app.schemas.common import CurrentUser
from app.schemas.prediction import (
    ForecastPointResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services.feature_service import FeatureService
from app.services.model_artifact_service import ModelArtifactService
from app.utils.dates import forecast_dates


class PredictionService:
    """Resolve time-safe inputs, invoke one adapter, and persist the job."""

    def __init__(
        self,
        *,
        settings: Settings,
        artifacts: ModelArtifactService,
        features: FeatureService,
        predictions: PredictionRepository,
        datasets: DatasetRepository,
        mappings: MappingRepository,
        products: ProductRepository,
        suppliers: SupplierRepository,
        warehouses: WarehouseRepository,
        stores: StoreRepository,
        sales: SalesRepository,
        inventory: InventoryRepository,
        vqr_adapter: PredictionModelAdapter | None = None,
        mock_adapter: PredictionModelAdapter | None = None,
    ) -> None:
        self.settings = settings
        self.artifacts = artifacts
        self.features = features
        self.predictions = predictions
        self.datasets = datasets
        self.mappings = mappings
        self.products = products
        self.suppliers = suppliers
        self.warehouses = warehouses
        self.stores = stores
        self.sales = sales
        self.inventory = inventory
        self.vqr_adapter = vqr_adapter or VQRModelAdapter(
            settings.resolved_model_artifact_dir,
            artifact_service=artifacts,
        )
        self.mock_adapter = mock_adapter or MockModelAdapter()

    def predict(
        self, request: PredictionRequest, user: CurrentUser
    ) -> PredictionResponse:
        self._verify_dataset_readiness(request, user)
        warehouse = self.warehouses.get_for_organization(
            request.warehouse_id, user.organization_id
        )
        if warehouse is None:
            raise AuthorizationError("Warehouse was not found in your organization.")
        store = None
        if request.store_id is not None:
            store = self.stores.get_for_organization(
                request.store_id, user.organization_id
            )
            if store is None:
                raise AuthorizationError("Store was not found in your organization.")
        product_rows = self.products.get_many(
            request.product_ids, user.organization_id
        )
        if len(product_rows) != len(request.product_ids):
            raise AuthorizationError(
                "One or more products were not found in your organization."
            )

        job = self.predictions.insert(
            {
                "organization_id": str(user.organization_id),
                "requested_by": str(user.id),
                "dataset_id": str(request.dataset_id) if request.dataset_id else None,
                "warehouse_id": str(request.warehouse_id),
                "store_id": str(request.store_id) if request.store_id else None,
                "forecast_start_date": request.forecast_start_date.isoformat(),
                "forecast_horizon_days": request.forecast_horizon_days,
                "status": "pending",
                "request_payload": request.model_dump(mode="json"),
                "is_mock": False,
            }
        )
        job_id = UUID(str(job["id"]))
        try:
            adapter = self._select_adapter()
        except ModelNotReadyError:
            self.predictions.update_for_organization(
                job_id,
                user.organization_id,
                {
                    "status": "model_not_ready",
                    "error_code": "MODEL_NOT_READY",
                    "error_message": "The trained prediction model is not available yet.",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            raise
        self.predictions.update_for_organization(
            job_id,
            user.organization_id,
            {
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "is_mock": adapter.is_mock,
            },
        )
        try:
            contexts, result_keys = self._contexts(
                request=request,
                organization_id=user.organization_id,
                products=product_rows,
                warehouse=warehouse,
                store=store,
            )
            if adapter.is_mock:
                feature_frame = self.features.build_mock_frame(contexts)
            else:
                feature_frame = self.features.build_frame(
                    contexts, self.artifacts.feature_schema()
                )
            predicted = adapter.predict(
                np.asarray(feature_frame.to_numpy(), dtype=float)
            )
            if len(predicted) != len(result_keys):
                raise PredictionError("Model returned an unexpected prediction count.")
            forecast_points = [
                ForecastPointResponse(
                    product_id=product_id,
                    warehouse_id=request.warehouse_id,
                    store_id=request.store_id,
                    forecast_date=forecast_date,
                    predicted_units_sold=max(0, int(round(float(value)))),
                )
                for (product_id, forecast_date), value in zip(
                    result_keys, predicted, strict=True
                )
            ]
            self.predictions.insert_results(
                [
                    {
                        "organization_id": str(user.organization_id),
                        "prediction_job_id": str(job_id),
                        "product_id": str(point.product_id),
                        "warehouse_id": str(request.warehouse_id),
                        "store_id": str(request.store_id) if request.store_id else None,
                        "forecast_date": point.forecast_date.isoformat(),
                        "predicted_units_sold": point.predicted_units_sold,
                        "lower_bound": point.lower_bound,
                        "upper_bound": point.upper_bound,
                        "metadata": {"is_mock": adapter.is_mock},
                    }
                    for point in forecast_points
                ]
            )
            self.predictions.update_for_organization(
                job_id,
                user.organization_id,
                {
                    "status": "completed",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as exc:
            code = getattr(exc, "code", "PREDICTION_ERROR")
            safe_message = (
                getattr(exc, "message", None)
                or "The forecast could not be generated."
            )
            self.predictions.update_for_organization(
                job_id,
                user.organization_id,
                {
                    "status": "failed",
                    "error_code": str(code),
                    "error_message": str(safe_message),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            if isinstance(exc, (FeatureResolutionError, PredictionError)):
                raise
            raise PredictionError() from exc
        return PredictionResponse(
            job_id=job_id,
            status="completed",
            model_name=adapter.model_name,
            model_version=adapter.model_version,
            target=adapter.target,
            forecast_horizon_days=request.forecast_horizon_days,
            is_mock=adapter.is_mock,
            results=forecast_points,
        )

    def _select_adapter(self) -> PredictionModelAdapter:
        if self.artifacts.status().ready:
            return self.vqr_adapter
        if self.settings.allow_mock_predictions:
            return self.mock_adapter
        raise ModelNotReadyError()

    def _verify_dataset_readiness(
        self, request: PredictionRequest, user: CurrentUser
    ) -> None:
        if request.dataset_id is None:
            return
        dataset = self.datasets.get_for_organization(
            request.dataset_id, user.organization_id
        )
        if dataset is None:
            raise DatasetNotFoundError()
        if dataset.get("status") != "ingested":
            raise DatasetValidationError(
                "The selected dataset has not completed ingestion.",
                details={"dataset_id": str(request.dataset_id)},
            )
        mappings = self.mappings.list_for_dataset(
            request.dataset_id, user.organization_id
        )
        if not mappings or not all(row.get("is_confirmed") for row in mappings):
            raise DatasetValidationError(
                "The selected dataset does not have fully confirmed mappings."
            )

    def _contexts(
        self,
        *,
        request: PredictionRequest,
        organization_id: UUID,
        products: Sequence[Mapping[str, object]],
        warehouse: Mapping[str, object],
        store: Mapping[str, object] | None,
    ) -> tuple[list[dict[str, object]], list[tuple[UUID, date]]]:
        contexts: list[dict[str, object]] = []
        keys: list[tuple[UUID, date]] = []
        for product in products:
            product_id = UUID(str(product["id"]))
            supplier = None
            supplier_id = product.get("supplier_id")
            if supplier_id:
                supplier = self.suppliers.get_for_organization(
                    UUID(str(supplier_id)), organization_id
                )
            latest_sale = self.sales.latest_before(
                organization_id=organization_id,
                product_id=product_id,
                forecast_date=request.forecast_start_date,
                warehouse_id=request.warehouse_id,
                store_id=request.store_id,
            )
            latest_inventory = self.inventory.latest_before(
                organization_id=organization_id,
                product_id=product_id,
                warehouse_id=request.warehouse_id,
                forecast_date=request.forecast_start_date,
            )
            base = self._canonical_context(
                product=product,
                supplier=supplier,
                warehouse=warehouse,
                store=store,
                latest_sale=latest_sale,
                latest_inventory=latest_inventory,
                overrides=request.scenario_overrides,
            )
            for forecast_date in forecast_dates(
                request.forecast_start_date, request.forecast_horizon_days
            ):
                contexts.append({**base, "forecast_date": forecast_date.isoformat()})
                keys.append((product_id, forecast_date))
        return contexts, keys

    @staticmethod
    def _canonical_context(
        *,
        product: Mapping[str, object],
        supplier: Mapping[str, object] | None,
        warehouse: Mapping[str, object],
        store: Mapping[str, object] | None,
        latest_sale: Mapping[str, object] | None,
        latest_inventory: Mapping[str, object] | None,
        overrides: Mapping[str, object],
    ) -> dict[str, object]:
        allowed_overrides = {
            "selling_price",
            "discount_pct",
            "is_promoted",
            "spoilage_risk",
        }
        unknown_overrides = sorted(set(overrides) - allowed_overrides)
        if unknown_overrides:
            raise FeatureResolutionError(
                "Unsupported scenario override fields.",
                details={"fields": unknown_overrides},
            )
        discount = overrides.get(
            "discount_pct",
            latest_sale.get("discount_pct") if latest_sale else 0.0,
        )
        if discount is not None and not 0 <= float(discount) <= 1:
            raise FeatureResolutionError("discount_pct must be between 0 and 1.")
        spoilage = overrides.get(
            "spoilage_risk",
            latest_sale.get("spoilage_risk")
            if latest_sale and latest_sale.get("spoilage_risk") is not None
            else latest_inventory.get("spoilage_risk")
            if latest_inventory
            else None,
        )
        context = {
            "shelf_life_days": product.get("shelf_life_days"),
            "cost_price": product.get("cost_price"),
            "spoilage_sensitivity": product.get("spoilage_sensitivity"),
            "storage_temp": product.get("storage_temp"),
            "category": product.get("category"),
            "supplier_score": supplier.get("supplier_score") if supplier else None,
            "region": (
                store.get("region")
                if store and store.get("region") is not None
                else warehouse.get("region")
            ),
            "selling_price": overrides.get(
                "selling_price",
                latest_sale.get("selling_price") if latest_sale else None,
            ),
            "discount_pct": discount,
            "is_promoted": overrides.get(
                "is_promoted",
                latest_sale.get("is_promoted") if latest_sale else False,
            ),
            "spoilage_risk": spoilage,
        }
        context.update(
            {
                key: value
                for key, value in overrides.items()
                if key in allowed_overrides
            }
        )
        context.pop("units_sold", None)
        return context
