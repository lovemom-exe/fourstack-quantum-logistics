"""Warehouse persistence with organization ownership filters."""

from collections.abc import Mapping, Sequence
from uuid import UUID

from app.repositories.base import BaseRepository


class WarehouseRepository(BaseRepository):
    table_name = "warehouses"

    def get_by_code(
        self, organization_id: UUID, warehouse_code: str
    ) -> dict[str, object] | None:
        response = (
            self.query()
            .select("*")
            .eq("organization_id", str(organization_id))
            .eq("warehouse_code", warehouse_code)
            .limit(1)
            .execute()
        )
        from app.repositories.base import response_rows

        rows = response_rows(response)
        return rows[0] if rows else None

    def upsert_stable(
        self, payloads: Sequence[Mapping[str, object]]
    ) -> list[dict[str, object]]:
        return self.upsert_many(
            payloads, on_conflict="organization_id,warehouse_code"
        )
