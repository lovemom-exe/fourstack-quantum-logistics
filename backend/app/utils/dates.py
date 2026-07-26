"""Date utilities."""

from datetime import date, timedelta


def forecast_dates(start: date, horizon_days: int) -> list[date]:
    return [start + timedelta(days=offset) for offset in range(horizon_days)]
