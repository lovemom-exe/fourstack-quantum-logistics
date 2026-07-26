"""Store persistence used by feature resolution."""

from collections.abc import Mapping, Sequence
from uuid import UUID

from app.repositories.base import BaseRepository, response_rows


class StoreRepository(BaseRepository):
    table_name = "stores"

    def get_by_external_id(
        self, external_id: str, organization_id: UUID
    ) -> dict[str, object] | None:
        rows = response_rows(
            self.query()
            .select("*")
            .eq("organization_id", str(organization_id))
            .eq("external_store_id", external_id)
            .limit(1)
            .execute()
        )
        return rows[0] if rows else None

    def upsert_stable(
        self, payloads: Sequence[Mapping[str, object]]
    ) -> list[dict[str, object]]:
        return self.upsert_many(
            payloads, on_conflict="organization_id,external_store_id"
        )
