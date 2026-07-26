"""Supplier master-data persistence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import UUID

from app.repositories.base import BaseRepository, response_rows


class SupplierRepository(BaseRepository):
    table_name = "suppliers"

    def get_by_external_id(
        self, external_id: str, organization_id: UUID
    ) -> dict[str, object] | None:
        result = (
            self.query()
            .select("*")
            .eq("organization_id", str(organization_id))
            .eq("external_supplier_id", external_id)
            .limit(1)
            .execute()
        )
        rows = response_rows(result)
        return rows[0] if rows else None

    def upsert_stable(
        self, payloads: Sequence[Mapping[str, object]]
    ) -> list[dict[str, object]]:
        return self.upsert_many(
            payloads, on_conflict="organization_id,external_supplier_id"
        )
