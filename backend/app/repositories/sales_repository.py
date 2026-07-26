"""Sales record persistence and time-safe lookup."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from uuid import UUID

from app.repositories.base import BaseRepository, response_rows


class SalesRepository(BaseRepository):
    table_name = "sales_records"

    def latest_before(
        self,
        *,
        organization_id: UUID,
        product_id: UUID,
        forecast_date: date,
        warehouse_id: UUID | None = None,
        store_id: UUID | None = None,
    ) -> dict[str, object] | None:
        query = (
            self.query()
            .select("*")
            .eq("organization_id", str(organization_id))
            .eq("product_id", str(product_id))
            .lte("transaction_date", forecast_date.isoformat())
        )
        if warehouse_id is not None:
            query = query.eq("warehouse_id", str(warehouse_id))
        if store_id is not None:
            query = query.eq("store_id", str(store_id))
        rows = response_rows(
            query.order("transaction_date", desc=True).limit(1).execute()
        )
        return rows[0] if rows else None

    def upsert_source_rows(
        self, payloads: Sequence[Mapping[str, object]]
    ) -> list[dict[str, object]]:
        return self.upsert_many(
            payloads, on_conflict="dataset_id,source_row_number"
        )
