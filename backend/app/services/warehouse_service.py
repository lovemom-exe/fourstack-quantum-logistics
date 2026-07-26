"""Warehouse CRUD business boundary."""

from uuid import UUID

from app.core.exceptions import AuthorizationError
from app.repositories.warehouse_repository import WarehouseRepository
from app.schemas.common import CurrentUser
from app.schemas.warehouse import (
    WarehouseCreate,
    WarehouseResponse,
    WarehouseUpdate,
)


class WarehouseService:
    def __init__(self, repository: WarehouseRepository) -> None:
        self.repository = repository

    def create(
        self, payload: WarehouseCreate, user: CurrentUser
    ) -> WarehouseResponse:
        row = self.repository.insert(
            {
                **payload.model_dump(mode="json"),
                "organization_id": str(user.organization_id),
            }
        )
        return WarehouseResponse.model_validate(row)

    def list(self, user: CurrentUser) -> list[WarehouseResponse]:
        return [
            WarehouseResponse.model_validate(row)
            for row in self.repository.list_for_organization(user.organization_id)
        ]

    def get(self, warehouse_id: UUID, user: CurrentUser) -> WarehouseResponse:
        row = self.repository.get_for_organization(
            warehouse_id, user.organization_id
        )
        if row is None:
            raise AuthorizationError("Warehouse was not found in your organization.")
        return WarehouseResponse.model_validate(row)

    def update(
        self,
        warehouse_id: UUID,
        payload: WarehouseUpdate,
        user: CurrentUser,
    ) -> WarehouseResponse:
        self.get(warehouse_id, user)
        values = payload.model_dump(mode="json", exclude_unset=True)
        row = self.repository.update_for_organization(
            warehouse_id, user.organization_id, values
        )
        if row is None:
            raise AuthorizationError("Warehouse update was not permitted.")
        return WarehouseResponse.model_validate(row)

    def delete(self, warehouse_id: UUID, user: CurrentUser) -> bool:
        self.get(warehouse_id, user)
        return self.repository.delete_for_organization(
            warehouse_id, user.organization_id
        )
