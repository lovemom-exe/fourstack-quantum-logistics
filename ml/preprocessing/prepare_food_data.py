# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: One-time orchestrator for the food (FreshRetailNet-50K) data pipeline:
#          loader -> cleaner -> feature_engineering. Reuses the existing
#          functions from those modules instead of re-implementing them.
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
import os

import pandas as pd

from utils.path import (
    FEATURE_ENGINEERING_DATA_PATH,
    FRESH_RETAIL_EVAL_PATH,
    FRESH_RETAIL_EVAL_PATH_FEATURE,
    FRESH_RETAIL_EVAL_PATH_PROCESS,
    FRESH_RETAIL_TRAIN_PATH,
    FRESH_RETAIL_TRAIN_PATH_FEATURE,
    FRESH_RETAIL_TRAIN_PATH_PROCESS,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
)

# ==========================================================================
# PARAMETERS
# ==========================================================================
# Columns dropped after feature engineering: raw IDs/categoricals, the raw
# datetime + constant hour_sin/hour_cos (this dataset is daily, not hourly,
# so hour is always 0), and the raw per-hour array columns. `hours_sale` in
# particular sums directly into `sale_amount` and must never survive.
DROP_COLUMNS = [
    "dt",
    "hour",
    "day_of_week",
    "day",
    "month",
    "hour_sin",
    "hour_cos",
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
    "hours_sale",
    "hours_stock_status",
]

# Hard leakage guard: these must never appear in the final featured columns.
LEAKAGE_COLUMNS = ["hours_sale"]

TARGET = "sale_amount"


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
def _ensure_dirs() -> None:
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(FEATURE_ENGINEERING_DATA_PATH, exist_ok=True)


def _ensure_raw() -> None:
    if os.path.exists(FRESH_RETAIL_TRAIN_PATH) and os.path.exists(FRESH_RETAIL_EVAL_PATH):
        print("[prepare_food_data] Raw CSVs already present, skipping download.")
        return

    print("[prepare_food_data] Raw CSVs missing - downloading FreshRetailNet-50K from HuggingFace...")
    import preprocessing.loader as loader

    loader.main()


def _build_processed() -> None:
    import preprocessing.cleaner as cleaner

    train_df = pd.read_csv(FRESH_RETAIL_TRAIN_PATH)
    eval_df = pd.read_csv(FRESH_RETAIL_EVAL_PATH)

    cleaner.encode_datetime(train_df, "dt")
    cleaner.encode_datetime(eval_df, "dt")

    cleaner.export_csv(train_df, FRESH_RETAIL_TRAIN_PATH_PROCESS)
    cleaner.export_csv(eval_df, FRESH_RETAIL_EVAL_PATH_PROCESS)


def _build_features() -> None:
    import preprocessing.feature_engineering as fe

    train_df = pd.read_csv(FRESH_RETAIL_TRAIN_PATH_PROCESS)
    eval_df = pd.read_csv(FRESH_RETAIL_EVAL_PATH_PROCESS)

    for df in (train_df, eval_df):
        fe.decode_list(df, "hours_stock_status")
        fe.engineer_interaction_features(df)
        fe.drop_columns(df, DROP_COLUMNS)

    rows_before = (len(train_df), len(eval_df))
    train_df.dropna(inplace=True)
    eval_df.dropna(inplace=True)
    print(
        f"[prepare_food_data] Dropped {rows_before[0] - len(train_df)} train row(s) and "
        f"{rows_before[1] - len(eval_df)} eval row(s) with missing values."
    )

    for name, df in (("train", train_df), ("eval", eval_df)):
        leaked = [c for c in LEAKAGE_COLUMNS if c in df.columns]
        if leaked:
            raise RuntimeError(
                f"[LEAKAGE GUARD] {leaked} still present in the {name} featured columns "
                "after drop_columns. These sum directly into sale_amount. Stopping."
            )

    fe.export_csv(train_df, FRESH_RETAIL_TRAIN_PATH_FEATURE)
    fe.export_csv(eval_df, FRESH_RETAIL_EVAL_PATH_FEATURE)


def _report(path: str, label: str) -> None:
    df = pd.read_csv(path)

    print("=" * 60)
    print(f"[{label}] rows: {len(df)}")
    print(f"[{label}] columns ({len(df.columns)}): {list(df.columns)}")
    print("=" * 60)

    assert TARGET in df.columns, f"'{TARGET}' missing from {label} featured data!"
    leaked = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    assert not leaked, f"[LEAKAGE GUARD] {leaked} present in {label} featured data!"


# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================
def main() -> None:
    _ensure_dirs()

    features_ready = os.path.exists(FRESH_RETAIL_TRAIN_PATH_FEATURE) and os.path.exists(
        FRESH_RETAIL_EVAL_PATH_FEATURE
    )

    if features_ready:
        print("[prepare_food_data] Featured CSVs already exist, skipping generation.")
    else:
        _ensure_raw()
        _build_processed()
        _build_features()

    _report(FRESH_RETAIL_TRAIN_PATH_FEATURE, "train")
    _report(FRESH_RETAIL_EVAL_PATH_FEATURE, "eval")


if __name__ == "__main__":
    main()
