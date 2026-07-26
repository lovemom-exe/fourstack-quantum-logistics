"""Structured CSV validation after mapping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import UUID

import pandas as pd

from app.schemas.dataset import DatasetType
from app.schemas.validation import DatasetValidationResponse, ValidationIssue
from app.services.mapping_service import MappingService, REQUIRED_FIELDS


NUMERIC_NON_NEGATIVE = {
    "units_sold",
    "selling_price",
    "base_price",
    "cost_price",
    "current_inventory",
    "reserved_inventory",
    "incoming_inventory",
    "storage_capacity",
}
DATE_FIELDS = {"transaction_date", "snapshot_date", "expiration_date"}
IDENTIFIER_FIELDS = {
    "product_id",
    "supplier_id",
    "warehouse_id",
    "store_id",
}


class ValidationService:
    def __init__(self, issue_limit: int = 100) -> None:
        self.issue_limit = issue_limit

    def validate(
        self,
        *,
        dataset_id: UUID,
        frame: pd.DataFrame,
        dataset_type: DatasetType,
        mappings: Sequence[Mapping[str, object]],
        known_references: Mapping[str, set[str]] | None = None,
        supported_values: Mapping[str, set[str]] | None = None,
        model_required_features: Sequence[str] = (),
    ) -> DatasetValidationResponse:
        issues: list[ValidationIssue] = []
        missing_required = MappingService.missing_required(mappings, dataset_type)
        for field in missing_required:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="MISSING_REQUIRED_MAPPING",
                    message=f"Required target field {field!r} is not confirmed.",
                    column=field,
                )
            )
        try:
            normalized, _ = MappingService.apply_confirmed(frame, mappings)
        except ValueError as exc:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="DUPLICATE_TARGET_MAPPING",
                    message=str(exc),
                )
            )
            normalized = frame.copy()

        duplicate_count = int(normalized.duplicated().sum())
        if duplicate_count:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="DUPLICATE_ROWS",
                    message=f"{duplicate_count} duplicate rows were found.",
                )
            )

        for column in normalized.columns:
            missing_count = int(normalized[column].isna().sum())
            if missing_count:
                severity = (
                    "error"
                    if column in REQUIRED_FIELDS[dataset_type]
                    else "warning"
                )
                issues.append(
                    ValidationIssue(
                        severity=severity,
                        code="MISSING_VALUES",
                        message=f"{missing_count} values are missing.",
                        column=str(column),
                    )
                )

        for column in NUMERIC_NON_NEGATIVE.intersection(normalized.columns):
            converted = pd.to_numeric(normalized[column], errors="coerce")
            invalid = converted.isna() & normalized[column].notna()
            for index in normalized.index[invalid][:5]:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="INVALID_NUMERIC",
                        message="Value is not numeric.",
                        column=column,
                        row_number=int(index) + 2,
                        value=str(normalized.at[index, column]),
                    )
                )
            negative = converted < 0
            for index in normalized.index[negative.fillna(False)][:5]:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="NEGATIVE_VALUE",
                        message=f"{column} cannot be negative.",
                        column=column,
                        row_number=int(index) + 2,
                        value=float(converted.at[index]),
                    )
                )

        if "discount_pct" in normalized.columns:
            discount = pd.to_numeric(normalized["discount_pct"], errors="coerce")
            invalid_discount = discount.notna() & ((discount < 0) | (discount > 1))
            for index in normalized.index[invalid_discount][:5]:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="DISCOUNT_OUT_OF_RANGE",
                        message="discount_pct must be between 0 and 1.",
                        column="discount_pct",
                        row_number=int(index) + 2,
                        value=float(discount.at[index]),
                    )
                )

        for column in DATE_FIELDS.intersection(normalized.columns):
            parsed = pd.to_datetime(
                normalized[column], errors="coerce", format="mixed"
            )
            invalid = parsed.isna() & normalized[column].notna()
            for index in normalized.index[invalid][:5]:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="INVALID_DATE",
                        message="Value is not a valid date.",
                        column=column,
                        row_number=int(index) + 2,
                        value=str(normalized.at[index, column]),
                    )
                )
        if {"expiration_date", "snapshot_date"}.issubset(normalized.columns):
            expiration = pd.to_datetime(
                normalized["expiration_date"], errors="coerce", format="mixed"
            )
            snapshot = pd.to_datetime(
                normalized["snapshot_date"], errors="coerce", format="mixed"
            )
            invalid_order = expiration.notna() & snapshot.notna() & (expiration < snapshot)
            for index in normalized.index[invalid_order][:5]:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="EXPIRATION_BEFORE_SNAPSHOT",
                        message="Expiration date is before snapshot date.",
                        column="expiration_date",
                        row_number=int(index) + 2,
                    )
                )

        for column in IDENTIFIER_FIELDS.intersection(normalized.columns):
            values = normalized[column].fillna("").astype(str).str.strip()
            empty = values.eq("")
            for index in normalized.index[empty][:5]:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="EMPTY_IDENTIFIER",
                        message=f"{column} cannot be empty.",
                        column=column,
                        row_number=int(index) + 2,
                    )
                )
            if known_references and column in known_references:
                unknown = ~values.isin(known_references[column]) & ~empty
                for index in normalized.index[unknown][:5]:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code=f"UNKNOWN_{column.upper()}",
                            message=f"Unknown {column} reference.",
                            column=column,
                            row_number=int(index) + 2,
                            value=values.at[index],
                        )
                    )

        for column in {"category", "region"}.intersection(normalized.columns):
            if supported_values and column in supported_values:
                values = normalized[column].dropna().astype(str)
                unsupported = ~values.isin(supported_values[column])
                for index in values.index[unsupported][:5]:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            code=f"UNSUPPORTED_{column.upper()}",
                            message=f"Value is not represented in the model schema.",
                            column=column,
                            row_number=int(index) + 2,
                            value=values.at[index],
                        )
                    )

        available = set(str(column) for column in normalized.columns)
        for feature in model_required_features:
            if feature == "units_sold":
                continue
            raw_group = (
                "category"
                if feature.startswith("category_")
                else "region"
                if feature.startswith("region_")
                else feature
            )
            if raw_group not in available and feature not in available:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="MODEL_FEATURE_UNAVAILABLE",
                        message=(
                            f"Model feature {feature!r} is not directly available "
                            "in this dataset and must be resolved from related records."
                        ),
                        column=feature,
                    )
                )

        issues = issues[: self.issue_limit]
        error_count = sum(issue.severity == "error" for issue in issues)
        warning_count = sum(issue.severity == "warning" for issue in issues)
        return DatasetValidationResponse(
            dataset_id=dataset_id,
            valid=error_count == 0,
            error_count=error_count,
            warning_count=warning_count,
            issues=issues,
        )
