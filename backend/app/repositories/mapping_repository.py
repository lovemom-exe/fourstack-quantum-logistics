"""Column mapping persistence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast
from uuid import UUID

from app.repositories.base import (
    BaseRepository,
    QueryBuilderProtocol,
    response_rows,
)


class MappingRepository(BaseRepository):
    table_name = "column_mappings"

    def list_for_dataset(
        self, dataset_id: UUID, organization_id: UUID
    ) -> list[dict[str, object]]:
        return response_rows(
            self.query()
            .select("*")
            .eq("dataset_id", str(dataset_id))
            .eq("organization_id", str(organization_id))
            .order("source_column")
            .execute()
        )

    def replace_for_dataset(
        self,
        dataset_id: UUID,
        organization_id: UUID,
        mappings: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        self.query().delete().eq("dataset_id", str(dataset_id)).eq(
            "organization_id", str(organization_id)
        ).execute()
        if not mappings:
            return []
        payloads = [
            {
                **dict(mapping),
                "dataset_id": str(dataset_id),
                "organization_id": str(organization_id),
            }
            for mapping in mappings
        ]
        query = cast(
            QueryBuilderProtocol, self.client.table(self.table_name)
        )
        return response_rows(query.insert(payloads).execute())

    def confirmed_for_dataset(
        self, dataset_id: UUID, organization_id: UUID
    ) -> list[dict[str, object]]:
        return response_rows(
            self.query()
            .select("*")
            .eq("dataset_id", str(dataset_id))
            .eq("organization_id", str(organization_id))
            .eq("is_confirmed", True)
            .execute()
        )
