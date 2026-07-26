"""Small testable Supabase repository base."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, Self, cast
from uuid import UUID

from app.clients.supabase_client import SupabaseClientProtocol


class QueryResponseProtocol(Protocol):
    data: object


class QueryBuilderProtocol(Protocol):
    def select(self, columns: str = "*") -> Self: ...
    def insert(self, values: object) -> Self: ...
    def update(self, values: object) -> Self: ...
    def upsert(
        self, values: object, *, on_conflict: str | None = None
    ) -> Self: ...
    def delete(self) -> Self: ...
    def eq(self, column: str, value: object) -> Self: ...
    def lte(self, column: str, value: object) -> Self: ...
    def order(self, column: str, *, desc: bool = False) -> Self: ...
    def limit(self, count: int) -> Self: ...
    def execute(self) -> QueryResponseProtocol: ...


def response_rows(response: QueryResponseProtocol) -> list[dict[str, object]]:
    """Normalize Supabase response data without returning raw responses."""
    data = response.data
    if data is None:
        return []
    if isinstance(data, Mapping):
        return [dict(data)]
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, Mapping)]
    raise TypeError("Supabase returned an unsupported response payload.")


class BaseRepository:
    table_name: str

    def __init__(self, client: SupabaseClientProtocol) -> None:
        self.client = client

    def query(self) -> QueryBuilderProtocol:
        return cast(QueryBuilderProtocol, self.client.table(self.table_name))

    def list_for_organization(self, organization_id: UUID) -> list[dict[str, object]]:
        response = (
            self.query()
            .select("*")
            .eq("organization_id", str(organization_id))
            .execute()
        )
        return response_rows(response)

    def get_for_organization(
        self, record_id: UUID, organization_id: UUID
    ) -> dict[str, object] | None:
        response = (
            self.query()
            .select("*")
            .eq("id", str(record_id))
            .eq("organization_id", str(organization_id))
            .limit(1)
            .execute()
        )
        rows = response_rows(response)
        return rows[0] if rows else None

    def insert(self, payload: Mapping[str, object]) -> dict[str, object]:
        rows = response_rows(self.query().insert(dict(payload)).execute())
        if not rows:
            raise RuntimeError(f"Insert into {self.table_name} returned no row.")
        return rows[0]

    def update_for_organization(
        self,
        record_id: UUID,
        organization_id: UUID,
        payload: Mapping[str, object],
    ) -> dict[str, object] | None:
        response = (
            self.query()
            .update(dict(payload))
            .eq("id", str(record_id))
            .eq("organization_id", str(organization_id))
            .execute()
        )
        rows = response_rows(response)
        return rows[0] if rows else None

    def delete_for_organization(
        self, record_id: UUID, organization_id: UUID
    ) -> bool:
        response = (
            self.query()
            .delete()
            .eq("id", str(record_id))
            .eq("organization_id", str(organization_id))
            .execute()
        )
        return bool(response_rows(response))

    def insert_many(
        self, payloads: Sequence[Mapping[str, object]]
    ) -> list[dict[str, object]]:
        if not payloads:
            return []
        return response_rows(
            self.query().insert([dict(item) for item in payloads]).execute()
        )

    def upsert_many(
        self,
        payloads: Sequence[Mapping[str, object]],
        *,
        on_conflict: str,
    ) -> list[dict[str, object]]:
        if not payloads:
            return []
        return response_rows(
            self.query()
            .upsert(
                [dict(item) for item in payloads],
                on_conflict=on_conflict,
            )
            .execute()
        )
