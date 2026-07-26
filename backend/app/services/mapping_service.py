"""Automatic and confirmed column mapping logic."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.schemas.dataset import DatasetType
from app.schemas.mapping import MappingSuggestion, MappingType
from app.utils.column_names import alias_lookup, normalize_column_name


TARGET_FIELDS: dict[DatasetType, set[str]] = {
    DatasetType.WAREHOUSE: {
        "warehouse_id",
        "warehouse_code",
        "warehouse_name",
        "warehouse_type",
        "country",
        "region",
        "city",
        "storage_capacity",
        "capacity_unit",
        "timezone",
    },
    DatasetType.PRODUCTS: {
        "product_id",
        "product_name",
        "category",
        "supplier_id",
        "shelf_life_days",
        "storage_temp",
        "spoilage_sensitivity",
        "base_price",
        "cost_price",
    },
    DatasetType.SUPPLIERS: {
        "supplier_id",
        "supplier_name",
        "supplier_score",
        "lead_time_days",
        "minimum_order_quantity",
        "supplier_country",
    },
    DatasetType.SALES: {
        "transaction_date",
        "product_id",
        "store_id",
        "warehouse_id",
        "units_sold",
        "selling_price",
        "discount_pct",
        "is_promoted",
        "spoilage_risk",
    },
    DatasetType.INVENTORY: {
        "snapshot_date",
        "product_id",
        "warehouse_id",
        "current_inventory",
        "reserved_inventory",
        "incoming_inventory",
        "expiration_date",
        "batch_id",
        "spoilage_risk",
    },
    DatasetType.CUSTOM: set(),
}

REQUIRED_FIELDS: dict[DatasetType, set[str]] = {
    DatasetType.WAREHOUSE: {"warehouse_code", "warehouse_name", "warehouse_type"},
    DatasetType.PRODUCTS: {"product_id", "product_name"},
    DatasetType.SUPPLIERS: {"supplier_id", "supplier_name"},
    DatasetType.SALES: {"transaction_date", "product_id", "units_sold"},
    DatasetType.INVENTORY: {
        "snapshot_date",
        "product_id",
        "warehouse_id",
        "current_inventory",
    },
    DatasetType.CUSTOM: set(),
}


class MappingService:
    def suggest(
        self,
        columns: Sequence[str],
        dataset_type: DatasetType,
    ) -> list[MappingSuggestion]:
        aliases = alias_lookup()
        allowed = TARGET_FIELDS[dataset_type]
        suggestions: list[MappingSuggestion] = []
        for source in columns:
            normalized = normalize_column_name(source)
            if normalized in allowed:
                suggestions.append(
                    MappingSuggestion(
                        source_column=source,
                        suggested_target=normalized,
                        confidence=1.0,
                        mapping_type=MappingType.EXACT,
                    )
                )
            elif aliases.get(normalized) in allowed:
                suggestions.append(
                    MappingSuggestion(
                        source_column=source,
                        suggested_target=aliases[normalized],
                        confidence=0.9,
                        mapping_type=MappingType.ALIAS,
                    )
                )
            else:
                suggestions.append(
                    MappingSuggestion(
                        source_column=source,
                        suggested_target=None,
                        confidence=0.0,
                        mapping_type=MappingType.IGNORED,
                    )
                )
        return suggestions

    @staticmethod
    def missing_required(
        mappings: Sequence[Mapping[str, object]],
        dataset_type: DatasetType,
    ) -> list[str]:
        confirmed_targets = {
            str(item["target_field"])
            for item in mappings
            if item.get("is_confirmed")
            and item.get("target_field")
            and item.get("mapping_type") != MappingType.IGNORED.value
        }
        return sorted(REQUIRED_FIELDS[dataset_type] - confirmed_targets)

    @staticmethod
    def apply_confirmed(
        frame: "pd.DataFrame",
        mappings: Sequence[Mapping[str, object]],
    ) -> tuple["pd.DataFrame", set[str]]:
        import pandas as pd

        rename: dict[str, str] = {}
        ignored: set[str] = set()
        defaults: dict[str, object] = {}
        for item in mappings:
            source = str(item["source_column"])
            mapping_type = str(item.get("mapping_type", ""))
            if mapping_type == MappingType.IGNORED.value:
                ignored.add(source)
                continue
            if not item.get("is_confirmed"):
                continue
            target = item.get("target_field")
            if isinstance(target, str) and source in frame.columns:
                rename[source] = target
            if (
                isinstance(target, str)
                and mapping_type == MappingType.DEFAULT.value
            ):
                defaults[target] = item.get("default_value")
        normalized = frame.rename(columns=rename).copy()
        for target, value in defaults.items():
            if target not in normalized.columns:
                normalized[target] = value
        if normalized.columns.duplicated().any():
            duplicates = normalized.columns[normalized.columns.duplicated()].tolist()
            raise ValueError(f"Multiple source columns map to the same target: {duplicates}")
        return normalized, ignored
