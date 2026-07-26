"""Warehouse CRUD routes."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_current_user, get_warehouse_service
from app.schemas.common import CurrentUser, DeleteResponse
from app.schemas.warehouse import (
    WarehouseCreate,
    WarehouseResponse,
    WarehouseUpdate,
)
from app.services.warehouse_service import WarehouseService


router = APIRouter(prefix="/warehouses", tags=["Warehouses"])


@router.post(
    "",
    response_model=WarehouseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a warehouse",
)
def create_warehouse(
    payload: WarehouseCreate,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[WarehouseService, Depends(get_warehouse_service)],
) -> WarehouseResponse:
    return service.create(payload, user)


@router.get(
    "",
    response_model=list[WarehouseResponse],
    summary="List organization warehouses",
)
def list_warehouses(
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[WarehouseService, Depends(get_warehouse_service)],
) -> list[WarehouseResponse]:
    return service.list(user)


@router.get(
    "/{warehouse_id}",
    response_model=WarehouseResponse,
    summary="Get one organization warehouse",
)
def get_warehouse(
    warehouse_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[WarehouseService, Depends(get_warehouse_service)],
) -> WarehouseResponse:
    return service.get(warehouse_id, user)


@router.put(
    "/{warehouse_id}",
    response_model=WarehouseResponse,
    summary="Update an organization warehouse",
)
def update_warehouse(
    warehouse_id: UUID,
    payload: WarehouseUpdate,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[WarehouseService, Depends(get_warehouse_service)],
) -> WarehouseResponse:
    return service.update(warehouse_id, payload, user)


@router.delete(
    "/{warehouse_id}",
    response_model=DeleteResponse,
    summary="Delete an organization warehouse",
)
def delete_warehouse(
    warehouse_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    service: Annotated[WarehouseService, Depends(get_warehouse_service)],
) -> DeleteResponse:
    return DeleteResponse(
        id=warehouse_id,
        deleted=service.delete(warehouse_id, user),
    )
