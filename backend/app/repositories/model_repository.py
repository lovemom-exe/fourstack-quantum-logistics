"""Model registry persistence."""

from uuid import UUID

from app.repositories.base import BaseRepository, response_rows


class ModelRepository(BaseRepository):
    table_name = "model_registry"

    def active_for_organization(
        self, organization_id: UUID
    ) -> dict[str, object] | None:
        rows = response_rows(
            self.query()
            .select("*")
            .eq("organization_id", str(organization_id))
            .eq("is_active", True)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        return rows[0] if rows else None
