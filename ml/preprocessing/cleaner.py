# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: This module provides utility functions for cleaning, preprocessing data
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
import json
import re

import numpy as np
import pandas as pd

from utils.path import (
    DISTRICT_STORE_PATH,
    # DISTRICT_STORE_PATH_PROCESS,
    FRESH_RETAIL_EVAL_PATH,
    FRESH_RETAIL_EVAL_PATH_PROCESS,
    FRESH_RETAIL_TRAIN_PATH,
    FRESH_RETAIL_TRAIN_PATH_PROCESS,
    NATIONAL_CENTRAL_MEDICAL_STORE_PATH,
    # NATIONAL_CENTRAL_MEDICAL_STORE_PATH_PROCESS,
    REGIONAL_WAREHOUSE_PATH,
    # REGIONAL_WAREHOUSE_PATH_PROCESS,
)

# ==========================================================================
# PARAMETERS
# ==========================================================================

# Food Data Path
food_train = FRESH_RETAIL_TRAIN_PATH
food_eval = FRESH_RETAIL_EVAL_PATH

# Medical Data Path
medical_district = DISTRICT_STORE_PATH
medical_central_store = NATIONAL_CENTRAL_MEDICAL_STORE_PATH
medical_region = REGIONAL_WAREHOUSE_PATH

# Food Data Frame
food_train_df = pd.read_csv(FRESH_RETAIL_TRAIN_PATH)
food_eval_df = pd.read_csv(FRESH_RETAIL_EVAL_PATH)


# Medical Data Frame
medical_district_df = pd.read_csv(DISTRICT_STORE_PATH)
medical_central_store_df = pd.read_csv(NATIONAL_CENTRAL_MEDICAL_STORE_PATH)
medical_region_df = pd.read_csv(REGIONAL_WAREHOUSE_PATH)


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
# Print And View the Data
def view(df: pd.DataFrame):
    print("=" * 50)
    json_df = df.iloc[0].to_dict()
    print(json.dumps(json_df, indent=4, default=str))
    # Basic View
    print("=" * 50)
    print("INFO:")
    print(df.info(verbose=True))
    print("=" * 50)
    # print(df.nunique())

    # print("=" * 50)
    # print(f"SHAPE: {df.shape[0]} rows, {df.shape[1]} columns")
    # print("=" * 50)

    # # Missing or Duplicate data report
    # print("DUPLICATE DATA :")
    # print("DUPLICATES: ", df.duplicated().sum())

    # missing_df = df.isnull().mean() * 100
    # print(f"MISSING DATA PERCENTAGES:\n{missing_df}%")
    # print("=" * 50)


def encode_datetime(df: pd.DataFrame, column: str) -> None:
    """
    Convert a datetime column into cyclical features.
    """
    df[column] = pd.to_datetime(df[column])

    # Extract time components
    df["hour"] = df[column].dt.hour
    df["day_of_week"] = df[column].dt.dayofweek
    df["day"] = df[column].dt.day
    df["month"] = df[column].dt.month

    # Hour (24h)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Day of week (7d)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Day of month (31d)
    df["day_sin_month"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos_month"] = np.cos(2 * np.pi * df["day"] / 31)

    # Month (12m)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)


def encode_array(df: pd.DataFrame, column: str, isStr: bool = True) -> None:
    """
    Convert a dataframe column containing lists (or list strings)
    into NumPy arrays.
    """

    if isStr:

        def parse(x):
            if pd.isna(x):
                return np.array([])

            x = x.strip("[]")

            # Split by comma OR any whitespace (space, tab, newline)
            tokens = re.split(r"[\s,]+", x.strip())

            return np.array([float(i) for i in tokens if i])

        df[column] = df[column].apply(parse)

    else:
        df[column] = df[column].apply(np.array)


def export_csv(df: pd.DataFrame, filepath: str, index: bool = False) -> None:
    """
    Export a DataFrame to a CSV file.
    """
    df.to_csv(filepath, index=index)
    print(f"[DONE] Data exported to '{filepath}'")


# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================
def main():
    # =============================================
    # Cleanning Data: food_eval and food_train
    # =============================================

    # # turn "dt" type into date-time
    # encode_datetime(food_eval_df, "dt", True)
    # encode_datetime(food_train_df, "dt", True)
    # # turn "hours_sale" and "hours_stock_status" from string of list into np.narray
    # encode_array(food_eval_df, "hours_sale")
    # encode_array(food_eval_df, "hours_stock_status")

    # encode_array(food_train_df, "hours_sale")
    # encode_array(food_train_df, "hours_stock_status")

    # =============================================
    # Cleaning Data: medical
    # =============================================

    # view(food_eval_df)

    # =============================================
    # Export to CSV
    # =============================================

    # export_csv(food_train_df, FRESH_RETAIL_TRAIN_PATH_PROCESS)
    # export_csv(food_eval_df, FRESH_RETAIL_EVAL_PATH_PROCESS)
    pass


if __name__ == "__main__":
    main()
