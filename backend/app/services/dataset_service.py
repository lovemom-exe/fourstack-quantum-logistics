"""Dataset upload, listing, preview, and deletion orchestration."""

from __future__ import annotations

from datetime import date
from uuid import UUID, uuid4

from app.core.exceptions import DatasetNotFoundError, DatasetValidationError
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.mapping_repository import MappingRepository
from app.schemas.common import CurrentUser
from app.schemas.dataset import (
    DatasetColumnResponse,
    DatasetPreviewResponse,
    DatasetResponse,
    DatasetStatus,
    DatasetType,
    DatasetUploadResponse,
)
from app.services.csv_service import CSVService
from app.services.mapping_service import MappingService
from app.services.storage_service import StorageService
from app.utils.files import storage_path


class DatasetService:
    def __init__(
        self,
        *,
        repository: DatasetRepository,
        mapping_repository: MappingRepository,
        storage: StorageService,
        csv: CSVService,
        mapping: MappingService,
    ) -> None:
        self.repository = repository
        self.mapping_repository = mapping_repository
        self.storage = storage
        self.csv = csv
        self.mapping = mapping

    def upload(
        self,
        *,
        content: bytes,
        filename: str | None,
        dataset_name: str,
        dataset_type: DatasetType,
        warehouse_id: UUID | None,
        user: CurrentUser,
    ) -> DatasetUploadResponse:
        original_filename = self.csv.validate_upload(filename, content)
        if not dataset_name.strip():
            raise DatasetValidationError("dataset_name cannot be blank.")
        analysis = self.csv.analyze(content)
        dataset_id = uuid4()
        object_path = storage_path(
            user.organization_id, dataset_id, original_filename
        )
        self.storage.upload_csv(object_path, content)
        dataset_created = False
        try:
            dataset_row = self.repository.insert(
                {
                    "id": str(dataset_id),
                    "organization_id": str(user.organization_id),
                    "dataset_name": dataset_name.strip(),
                    "dataset_type": dataset_type.value,
                    "status": DatasetStatus.MAPPING_REQUIRED.value,
                    "original_filename": original_filename,
                    "storage_path": object_path,
                    "file_size_bytes": len(content),
                    "row_count": analysis.row_count,
                    "column_count": analysis.column_count,
                    "date_min": analysis.date_min,
                    "date_max": analysis.date_max,
                    "uploaded_by": str(user.id),
                    "metadata": {
                        "warehouse_id": str(warehouse_id) if warehouse_id else None,
                        "duplicate_row_count": analysis.duplicate_row_count,
                    },
                }
            )
            dataset_created = True
            column_rows = self.repository.create_columns(
                [
                    {
                        **column,
                        "dataset_id": str(dataset_id),
                    }
                    for column in analysis.columns
                ]
            )
            suggestions = self.mapping.suggest(
                [str(column["source_column"]) for column in analysis.columns],
                dataset_type,
            )
            mapping_rows = self.mapping_repository.replace_for_dataset(
                dataset_id,
                user.organization_id,
                [
                    {
                        "source_column": item.source_column,
                        "target_field": item.suggested_target,
                        "mapping_type": item.mapping_type.value,
                        "confidence": item.confidence,
                        "is_confirmed": item.is_confirmed,
                        "default_value": None,
                    }
                    for item in suggestions
                ],
            )
        except Exception:
            if dataset_created:
                self.repository.delete_for_organization(
                    dataset_id, user.organization_id
                )
            self.storage.delete(object_path)
            raise
        return DatasetUploadResponse(
            **DatasetResponse.model_validate(dataset_row).model_dump(),
            duplicate_row_count=analysis.duplicate_row_count,
            columns=[
                DatasetColumnResponse.model_validate(row) for row in column_rows
            ],
            suggested_mapping_count=len(mapping_rows),
        )

    def list(self, user: CurrentUser) -> list[DatasetResponse]:
        return [
            DatasetResponse.model_validate(row)
            for row in self.repository.list_for_organization(user.organization_id)
        ]

    def get(self, dataset_id: UUID, user: CurrentUser) -> DatasetResponse:
        row = self.repository.get_for_organization(
            dataset_id, user.organization_id
        )
        if row is None:
            raise DatasetNotFoundError()
        return DatasetResponse.model_validate(row)

    def columns(
        self, dataset_id: UUID, user: CurrentUser
    ) -> list[DatasetColumnResponse]:
        self.get(dataset_id, user)
        return [
            DatasetColumnResponse.model_validate(row)
            for row in self.repository.list_columns(dataset_id)
        ]

    def preview(
        self, dataset_id: UUID, user: CurrentUser, limit: int
    ) -> DatasetPreviewResponse:
        dataset = self.get(dataset_id, user)
        content = self.storage.download(dataset.storage_path)
        rows = self.csv.preview(content, limit)
        return DatasetPreviewResponse(
            dataset_id=dataset_id,
            columns=list(rows[0].keys()) if rows else [],
            rows=rows,
            returned_rows=len(rows),
            total_rows=dataset.row_count,
        )

    def delete(self, dataset_id: UUID, user: CurrentUser) -> bool:
        dataset = self.get(dataset_id, user)
        deleted = self.repository.delete_for_organization(
            dataset_id, user.organization_id
        )
        if deleted:
            self.storage.delete(dataset.storage_path)
        return deleted
