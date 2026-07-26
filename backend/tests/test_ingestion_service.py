"""JSONB extras and stable source-row contract tests."""

from uuid import UUID

import pandas as pd

from app.schemas.dataset import DatasetType
from app.services.ingestion_service import IngestionService
from tests.conftest import ORG_ID


def test_extra_csv_columns_are_preserved_in_product_metadata() -> None:
    service = IngestionService(
        datasets=object(),
        mappings=object(),
        storage=object(),
        csv=object(),
        validator=object(),
        products=object(),
        suppliers=object(),
        warehouses=object(),
        stores=object(),
        sales=object(),
        inventory=object(),
        batch_size=100,
    )
    payload = service._row_payload(
        row=pd.Series(
            {
                "product_id": "SKU-1",
                "product_name": "Milk",
                "category": "Dairy",
                "customer_segment": "premium",
                "receipt_note": "keep",
            }
        ),
        source_row_number=2,
        dataset_id=UUID("44444444-4444-4444-4444-444444444444"),
        organization_id=ORG_ID,
        dataset_type=DatasetType.PRODUCTS,
    )
    assert payload["metadata"] == {
        "customer_segment": "premium",
        "receipt_note": "keep",
    }
    assert payload["external_product_id"] == "SKU-1"
