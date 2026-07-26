"""Column normalization, aliases, and required mapping tests."""

from app.schemas.dataset import DatasetType
from app.schemas.mapping import MappingType
from app.services.mapping_service import MappingService
from app.utils.column_names import normalize_column_name


def test_column_name_normalization_equivalence() -> None:
    variants = ["Units Sold", "units-sold", "units sold", "UNITS_SOLD"]
    assert {normalize_column_name(value) for value in variants} == {"units_sold"}


def test_alias_mapping_and_unknown_default_ignored() -> None:
    suggestions = MappingService().suggest(
        ["qty_sold", "Sale Date", "receipt_note"],
        DatasetType.SALES,
    )
    assert suggestions[0].suggested_target == "units_sold"
    assert suggestions[0].mapping_type == MappingType.ALIAS
    assert suggestions[1].suggested_target == "transaction_date"
    assert suggestions[2].mapping_type == MappingType.IGNORED
    assert suggestions[2].suggested_target is None


def test_missing_required_mappings() -> None:
    mappings = [
        {
            "target_field": "transaction_date",
            "is_confirmed": True,
            "mapping_type": "exact",
        }
    ]
    assert MappingService.missing_required(mappings, DatasetType.SALES) == [
        "product_id",
        "units_sold",
    ]
