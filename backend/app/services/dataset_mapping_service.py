"""Dataset mapping persistence orchestration."""

from __future__ import annotations

from uuid import UUID

from app.core.exceptions import DatasetNotFoundError
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.mapping_repository import MappingRepository
from app.schemas.common import CurrentUser
from app.schemas.dataset import DatasetType
from app.schemas.mapping import (
    ColumnMappingResponse,
    ColumnMappingsUpdate,
)
from app.services.mapping_service import MappingService


class DatasetMappingService:
    def __init__(
        self,
        datasets: DatasetRepository,
        mappings: MappingRepository,
        mapper: MappingService,
    ) -> None:
        self.datasets = datasets
        self.mappings = mappings
        self.mapper = mapper

    def _dataset(
        self, dataset_id: UUID, user: CurrentUser
    ) -> dict[str, object]:
        row = self.datasets.get_for_organization(
            dataset_id, user.organization_id
        )
        if row is None:
            raise DatasetNotFoundError()
        return row

    def list(
        self, dataset_id: UUID, user: CurrentUser
    ) -> list[ColumnMappingResponse]:
        self._dataset(dataset_id, user)
        return [
            ColumnMappingResponse.model_validate(row)
            for row in self.mappings.list_for_dataset(
                dataset_id, user.organization_id
            )
        ]

    def auto_map(
        self, dataset_id: UUID, user: CurrentUser
    ) -> list[ColumnMappingResponse]:
        dataset = self._dataset(dataset_id, user)
        columns = self.datasets.list_columns(dataset_id)
        suggestions = self.mapper.suggest(
            [str(item["source_column"]) for item in columns],
            DatasetType(str(dataset["dataset_type"])),
        )
        rows = self.mappings.replace_for_dataset(
            dataset_id,
            user.organization_id,
            [
                {
                    "source_column": item.source_column,
                    "target_field": item.suggested_target,
                    "mapping_type": item.mapping_type.value,
                    "confidence": item.confidence,
                    "is_confirmed": False,
                    "default_value": None,
                }
                for item in suggestions
            ],
        )
        return [ColumnMappingResponse.model_validate(row) for row in rows]

    def update(
        self,
        dataset_id: UUID,
        payload: ColumnMappingsUpdate,
        user: CurrentUser,
    ) -> list[ColumnMappingResponse]:
        self._dataset(dataset_id, user)
        rows = self.mappings.replace_for_dataset(
            dataset_id,
            user.organization_id,
            [
                {
                    **item.model_dump(mode="json"),
                    "confidence": 1.0 if item.is_confirmed else 0.0,
                }
                for item in payload.mappings
            ],
        )
        return [ColumnMappingResponse.model_validate(row) for row in rows]
