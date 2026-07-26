"""CSV filename, parsing, type inference, and metadata tests."""

import pytest

from app.core.exceptions import DatasetValidationError
from app.services.csv_service import CSVService
from app.utils.files import sanitize_filename


def test_csv_filename_validation_and_sanitization() -> None:
    assert sanitize_filename("../../July Sales (final).CSV") == "July_Sales_final.csv"
    with pytest.raises(ValueError):
        sanitize_filename("sales.xlsx")


def test_csv_type_detection_and_original_columns() -> None:
    content = (
        b"Sale Date,Units Sold,Promotion,Category,Notes\n"
        b"2026-01-01,10,true,Dairy,first\n"
        b"2026-01-02,12,false,Dairy,second\n"
        b"2026-01-03,14,true,Produce,third\n"
    )
    analysis = CSVService(1).analyze(content)
    detected = {
        row["source_column"]: row["detected_type"]
        for row in analysis.columns
    }
    assert list(analysis.frame.columns) == [
        "Sale Date",
        "Units Sold",
        "Promotion",
        "Category",
        "Notes",
    ]
    assert detected["Sale Date"] == "date"
    assert detected["Units Sold"] == "numeric"
    assert detected["Promotion"] == "boolean"
    assert analysis.row_count == 3


def test_csv_rejects_empty_or_oversized_content() -> None:
    service = CSVService(1)
    with pytest.raises(DatasetValidationError):
        service.validate_upload("data.csv", b"")
    with pytest.raises(DatasetValidationError):
        service.validate_upload("data.csv", b"x" * (1024 * 1024 + 1))
