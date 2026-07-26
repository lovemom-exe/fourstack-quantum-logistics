"""CSV parsing and metadata inspection."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import pandas as pd

from app.core.exceptions import DatasetValidationError
from app.utils.column_names import normalize_column_name
from app.utils.dataframe import dataframe_records, python_scalar
from app.utils.files import validate_csv_filename


BOOLEAN_VALUES = {"true", "false", "yes", "no", "1", "0", "y", "n"}


@dataclass(frozen=True)
class CSVAnalysis:
    frame: pd.DataFrame
    row_count: int
    column_count: int
    duplicate_row_count: int
    columns: list[dict[str, object]]
    date_min: str | None
    date_max: str | None


class CSVService:
    """Read CSV bytes once and derive safe metadata."""

    def __init__(self, max_size_mb: int, preview_limit: int = 20) -> None:
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.preview_limit = preview_limit

    def validate_upload(self, filename: str | None, content: bytes) -> str:
        try:
            safe_name = validate_csv_filename(filename)
        except ValueError as exc:
            raise DatasetValidationError(str(exc)) from exc
        if not content:
            raise DatasetValidationError("The uploaded CSV is empty.")
        if len(content) > self.max_size_bytes:
            raise DatasetValidationError(
                f"CSV exceeds the configured {self.max_size_bytes // (1024 * 1024)} MB limit."
            )
        return safe_name

    def parse(self, content: bytes) -> pd.DataFrame:
        try:
            frame = pd.read_csv(BytesIO(content))
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            raise DatasetValidationError(f"CSV parsing failed: {exc}") from exc
        if frame.empty:
            raise DatasetValidationError("The CSV contains no data rows.")
        if not len(frame.columns):
            raise DatasetValidationError("The CSV contains no columns.")
        return frame

    @staticmethod
    def detect_type(series: pd.Series) -> str:
        non_null = series.dropna()
        if non_null.empty:
            return "unknown"
        normalized = non_null.astype(str).str.strip().str.lower()
        if normalized.isin(BOOLEAN_VALUES).all():
            return "boolean"
        numeric = pd.to_numeric(non_null, errors="coerce")
        if numeric.notna().all():
            return "numeric"
        date_values = pd.to_datetime(
            non_null, errors="coerce", format="mixed", utc=False
        )
        if float(date_values.notna().mean()) >= 0.9:
            return "date"
        unique_ratio = non_null.nunique(dropna=True) / max(1, len(non_null))
        return "categorical" if unique_ratio <= 0.5 else "text"

    def analyze(self, content: bytes) -> CSVAnalysis:
        frame = self.parse(content)
        column_metadata: list[dict[str, object]] = []
        date_series: pd.Series | None = None
        for source_column in frame.columns:
            series = frame[source_column]
            detected_type = self.detect_type(series)
            if date_series is None and detected_type == "date":
                date_series = pd.to_datetime(
                    series, errors="coerce", format="mixed"
                )
            samples = [
                python_scalar(value)
                for value in series.dropna().drop_duplicates().head(5).tolist()
            ]
            column_metadata.append(
                {
                    "source_column": str(source_column),
                    "normalized_column": normalize_column_name(str(source_column)),
                    "detected_type": detected_type,
                    "sample_values": samples,
                    "missing_count": int(series.isna().sum()),
                    "unique_count": int(series.nunique(dropna=True)),
                }
            )
        date_min = None
        date_max = None
        if date_series is not None and date_series.notna().any():
            date_min = date_series.min().date().isoformat()
            date_max = date_series.max().date().isoformat()
        return CSVAnalysis(
            frame=frame,
            row_count=int(len(frame)),
            column_count=int(len(frame.columns)),
            duplicate_row_count=int(frame.duplicated().sum()),
            columns=column_metadata,
            date_min=date_min,
            date_max=date_max,
        )

    def preview(self, content: bytes, limit: int | None = None) -> list[dict[str, object]]:
        frame = self.parse(content)
        safe_limit = min(limit or self.preview_limit, self.preview_limit)
        return dataframe_records(frame.head(safe_limit))
