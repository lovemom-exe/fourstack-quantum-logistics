# ==========================================================================
# Author: Hoang Anh Quan
# Version: 0.1.0
# Purpose: Define system-level API endpoints
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
# Routes
# API
from api.routes.forecast import router as forecast_router
from api.routes.sales import router as sales_router
from api.routes.system import router as system_router
from fastapi import APIRouter

# ==========================================================================
# PARAMETERS
# ==========================================================================
router = APIRouter()

router.include_router(system_router)
router.include_router(forecast_router)
router.include_router(sales_router)
