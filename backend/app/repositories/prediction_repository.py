"""Prediction job and result persistence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast
from uuid import UUID

from app.repositories.base import (
    BaseRepository,
    QueryBuilderProtocol,
    response_rows,
)


class PredictionRepository(BaseRepository):
    table_name = "prediction_jobs"

    def insert_results(
        self, rows: Sequence[Mapping[str, object]]
    ) -> list[dict[str, object]]:
        if not rows:
            return []
        query = cast(
            QueryBuilderProtocol, self.client.table("forecast_results")
        )
        return response_rows(
            query.insert([dict(row) for row in rows]).execute()
        )

    def list_results(
        self, job_id: UUID, organization_id: UUID
    ) -> list[dict[str, object]]:
        job = self.get_for_organization(job_id, organization_id)
        if job is None:
            return []
        query = cast(
            QueryBuilderProtocol, self.client.table("forecast_results")
        )
        return response_rows(
            query.select("*")
            .eq("prediction_job_id", str(job_id))
            .order("forecast_date")
            .execute()
        )
