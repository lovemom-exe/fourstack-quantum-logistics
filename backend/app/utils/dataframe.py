"""Pandas-to-JSON conversion helpers."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd


def python_scalar(value: object) -> object:
    """Convert pandas/NumPy scalars into JSON-safe Python values."""
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    return value


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Return JSON-safe records while preserving original column names."""
    return [
        {str(column): python_scalar(value) for column, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]
