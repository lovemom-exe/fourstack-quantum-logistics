"""Structured validation and readiness tests."""

from uuid import UUID

import pandas as pd

from app.schemas.dataset import DatasetType
from app.schemas.model import ModelStatusResponse
from app.services.readiness_service import ReadinessService
from app.services.validation_service import ValidationService


DATASET_ID = UUID("44444444-4444-4444-4444-444444444444")


def confirmed_sales_mappings():
    return [
        {
            "source_column": "date",
            "target_field": "transaction_date",
            "mapping_type": "manual",
            "is_confirmed": True,
        },
        {
            "source_column": "sku",
            "target_field": "product_id",
            "mapping_type": "manual",
            "is_confirmed": True,
        },
        {
            "source_column": "qty",
            "target_field": "units_sold",
            "mapping_type": "manual",
            "is_confirmed": True,
        },
        {
            "source_column": "discount",
            "target_field": "discount_pct",
            "mapping_type": "manual",
            "is_confirmed": True,
        },
    ]


def test_structured_validation_issues() -> None:
    frame = pd.DataFrame(
        {
            "date": ["invalid", "2026-01-02"],
            "sku": ["", "SKU-1"],
            "qty": [-1, 4],
            "discount": [1.2, 0.1],
        }
    )
    result = ValidationService().validate(
        dataset_id=DATASET_ID,
        frame=frame,
        dataset_type=DatasetType.SALES,
        mappings=confirmed_sales_mappings(),
    )
    codes = {issue.code for issue in result.issues}
    assert not result.valid
    assert {
        "INVALID_DATE",
        "EMPTY_IDENTIFIER",
        "NEGATIVE_VALUE",
        "DISCOUNT_OUT_OF_RANGE",
    }.issubset(codes)
    assert all(issue.message for issue in result.issues)


def test_readiness_reports_absent_model_without_blocking_ingestion() -> None:
    response = ReadinessService().evaluate(
        dataset_id=DATASET_ID,
        dataset={"status": "ingested"},
        organization_datasets=[
            {"dataset_type": kind, "status": "ingested"}
            for kind in ("products", "suppliers", "sales", "inventory")
        ],
        mappings=[{"is_confirmed": True}],
        warehouse_exists=True,
        model_status=ModelStatusResponse(
            ready=False,
            artifact_directory="../ml/models/perishable_vqr",
            missing_files=["vqr_model.dill"],
        ),
    )
    assert response.ingestion_ready is True
    assert response.model_ready is False
    assert response.forecast_ready is False
    assert "still unavailable" in response.warnings[0]
