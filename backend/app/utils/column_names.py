"""Column normalization and alias definitions."""

from __future__ import annotations

import re


COLUMN_ALIASES: dict[str, list[str]] = {
    "product_id": ["sku", "sku_id", "product_code", "item_id"],
    "units_sold": [
        "quantity_sold",
        "qty_sold",
        "sold_units",
        "sales_quantity",
    ],
    "transaction_date": ["date", "sales_date", "sale_date"],
    "selling_price": ["price", "sale_price", "retail_price", "unit_price"],
    "supplier_id": ["vendor_id", "supplier_code", "vendor_code"],
    "warehouse_id": ["warehouse_code", "depot_id", "location_id"],
    "store_id": ["store_code", "shop_id", "outlet_id"],
    "snapshot_date": ["inventory_date", "stock_date", "as_of_date"],
    "current_inventory": ["stock_on_hand", "on_hand", "current_stock"],
    "is_promoted": ["promotion_flag", "promoted", "on_promotion"],
    "discount_pct": ["discount", "discount_rate", "discount_percent"],
}


def normalize_column_name(value: str) -> str:
    """Normalize common separators and casing to snake_case."""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")


def alias_lookup() -> dict[str, str]:
    """Return normalized alias -> canonical target."""
    lookup: dict[str, str] = {}
    for target, aliases in COLUMN_ALIASES.items():
        lookup[normalize_column_name(target)] = target
        for alias in aliases:
            lookup[normalize_column_name(alias)] = target
    return lookup
