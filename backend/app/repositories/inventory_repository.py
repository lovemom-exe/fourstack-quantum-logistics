"""Inventory snapshot persistence and time-safe lookup."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from uuid import UUID

from app.repositories.base import BaseRepository, response_rows


class InventoryRepository(BaseRepository):
    table_name = "inventory_snapshots"

    def latest_before(
        self,
        *,
        organization_id: UUID,
        product_id: UUID,
        warehouse_id: UUID,
        forecast_date: date,
    ) -> dict[str, object] | None:
        rows = response_rows(
            self.query()
            .select("*")
            .eq("organization_id", str(organization_id))
            .eq("product_id", str(product_id))
            .eq("warehouse_id", str(warehouse_id))
            .lte("snapshot_date", forecast_date.isoformat())
            .order("snapshot_date", desc=True)
            .limit(1)
            .execute()
        )
        return rows[0] if rows else None

    def upsert_source_rows(
        self, payloads: Sequence[Mapping[str, object]]
    ) -> list[dict[str, object]]:
        return self.upsert_many(
            payloads, on_conflict="dataset_id,source_row_number"
        )
