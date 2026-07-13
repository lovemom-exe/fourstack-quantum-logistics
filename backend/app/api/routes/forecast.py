# ==========================================================================
# Author: Hoang Anh Quan
# Version: 0.1.0
# Purpose: Define forecast-related API endpoints
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
from fastapi import APIRouter
from repositories.forecast_repo import forecast_data

# ==========================================================================
# PARAMETERS
# ==========================================================================
router = APIRouter(prefix="/forecast", tags=["forecast"])


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
@router.get("/")
def get_forecasts():
    return forecast_data


@router.get("/{forecast_id}")
def get_forecast(forecast_id: int):
    for forecast in forecast_data:
        if forecast["forecast_id"] == forecast_id:
            return forecast


@router.post("/")
def create_forecast():
    pass
