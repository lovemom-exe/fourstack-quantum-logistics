"""Dataset and dataset-column persistence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import UUID

from app.repositories.base import BaseRepository, response_rows


class DatasetRepository(BaseRepository):
    table_name = "datasets"

    def create_columns(
        self, payloads: Sequence[Mapping[str, object]]
    ) -> list[dict[str, object]]:
        if not payloads:
            return []
        response = self.client.table("dataset_columns")
        from typing import cast

        from app.repositories.base import QueryBuilderProtocol

        query = cast(QueryBuilderProtocol, response)
        return response_rows(
            query.insert([dict(payload) for payload in payloads]).execute()
        )

    def list_columns(self, dataset_id: UUID) -> list[dict[str, object]]:
        from typing import cast

        from app.repositories.base import QueryBuilderProtocol

        query = cast(
            QueryBuilderProtocol, self.client.table("dataset_columns")
        )
        return response_rows(
            query.select("*")
            .eq("dataset_id", str(dataset_id))
            .order("created_at")
            .execute()
        )

    def update_status(
        self, dataset_id: UUID, organization_id: UUID, status: str
    ) -> dict[str, object] | None:
        return self.update_for_organization(
            dataset_id, organization_id, {"status": status}
        )

    def list_ingested_types(self, organization_id: UUID) -> set[str]:
        rows = self.list_for_organization(organization_id)
        return {
            str(row["dataset_type"])
            for row in rows
            if row.get("status") == "ingested"
        }
