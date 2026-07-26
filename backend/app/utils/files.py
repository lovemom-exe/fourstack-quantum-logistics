"""Upload filename and storage-path helpers."""

from __future__ import annotations

import re
from pathlib import Path
from uuid import UUID, uuid4


def validate_csv_filename(filename: str | None) -> str:
    if not filename:
        raise ValueError("A filename is required.")
    if "\x00" in filename:
        raise ValueError("The filename contains invalid characters.")
    basename = Path(filename).name
    if Path(basename).suffix.lower() != ".csv":
        raise ValueError("Only .csv files are accepted.")
    return basename


def sanitize_filename(filename: str) -> str:
    basename = validate_csv_filename(filename)
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(basename).stem).strip("._")
    safe_stem = stem[:120] or "dataset"
    return f"{safe_stem}.csv"


def storage_path(
    organization_id: UUID,
    dataset_id: UUID,
    filename: str,
) -> str:
    safe_name = sanitize_filename(filename)
    return f"{organization_id}/{dataset_id}/{uuid4().hex}_{safe_name}"
