"""Warehouse DTO, ownership, and route tests."""

from datetime import datetime, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.api.dependencies import get_warehouse_service
from app.core.exceptions import AuthorizationError
from app.schemas.warehouse import WarehouseCreate, WarehouseResponse
from app.services.warehouse_service import WarehouseService
from tests.conftest import ORG_ID


WAREHOUSE_ID = UUID("33333333-3333-3333-3333-333333333333")


def warehouse_response() -> WarehouseResponse:
    now = datetime.now(timezone.utc)
    return WarehouseResponse(
        id=WAREHOUSE_ID,
        organization_id=ORG_ID,
        warehouse_code="WH-01",
        warehouse_name="Central Cold Store",
        warehouse_type="chilled",
        storage_capacity=1000,
        capacity_unit="pallets",
        timezone="Asia/Ho_Chi_Minh",
        created_at=now,
        updated_at=now,
    )


def test_warehouse_dto_validates_capacity_type_name_and_timezone() -> None:
    with pytest.raises(ValidationError):
        WarehouseCreate(
            warehouse_code="WH",
            warehouse_name=" ",
            warehouse_type="invalid",
            storage_capacity=-1,
            timezone="Not/A_Timezone",
        )


def test_warehouse_route_uses_typed_service(app, client) -> None:
    class FakeService:
        def list(self, user):
            assert user.organization_id == ORG_ID
            return [warehouse_response()]

    app.dependency_overrides[get_warehouse_service] = lambda: FakeService()
    response = client.get("/api/v1/warehouses")
    assert response.status_code == 200
    assert response.json()[0]["warehouse_code"] == "WH-01"
    assert "organization_id" in response.json()[0]


def test_warehouse_service_enforces_organization_ownership(authenticated_user) -> None:
    class FakeRepository:
        def get_for_organization(self, warehouse_id, organization_id):
            assert organization_id == ORG_ID
            return None

    service = WarehouseService(FakeRepository())
    with pytest.raises(AuthorizationError):
        service.get(WAREHOUSE_ID, authenticated_user)
