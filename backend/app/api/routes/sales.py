# ==========================================================================
# Author: Hoang Anh Quan
# Version: 0.1.0
# Purpose: Define sales-related API endpoints
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
from fastapi import APIRouter
from repositories.sales_repo import sales_data

# ==========================================================================
# PARAMETERS
# ==========================================================================
router = APIRouter(prefix="/sales", tags=["sales"])


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
@router.get("/")
def get_sales():
    return sales_data


@router.get("/{sale_id}")
def get_sale(sale_id: int):
    for sale in sales_data:
        if sale["sale_id"] == sale_id:
            return sale


@router.post("/")
def create_sales():
    pass
