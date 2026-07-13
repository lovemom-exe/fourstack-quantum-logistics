# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: This module provides utility functions feature engineering for machine learning tasks
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
import json

import re
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Literal

from utils.path import (
    # DISTRICT_STORE_PATH_FEATURE,
    # DISTRICT_STORE_PATH_PROCESS,
    FEATURE_ENGINEERING_DATA_PATH,
    FRESH_RETAIL_EVAL_PATH_FEATURE,
    FRESH_RETAIL_EVAL_PATH_PROCESS,
    FRESH_RETAIL_TRAIN_PATH_FEATURE,
    FRESH_RETAIL_TRAIN_PATH_PROCESS,
    # NATIONAL_CENTRAL_MEDICAL_STORE_PATH_FEATURE,
    # NATIONAL_CENTRAL_MEDICAL_STORE_PATH_PROCESS,
    # REGIONAL_WAREHOUSE_PATH_FEATURE,
    # REGIONAL_WAREHOUSE_PATH_PROCESS,
)

# ==========================================================================
# PARAMETERS
# ==========================================================================
# Export Data Path
export_path = FEATURE_ENGINEERING_DATA_PATH

# Food Data Path
food_train = FRESH_RETAIL_TRAIN_PATH_PROCESS
food_eval = FRESH_RETAIL_EVAL_PATH_PROCESS

# Medical Data Path
# medical_district = DISTRICT_STORE_PATH_PROCESS
# medical_central_store = NATIONAL_CENTRAL_MEDICAL_STORE_PATH_PROCESS
# medical_region = REGIONAL_WAREHOUSE_PATH_PROCESS

# Food Data Frame
food_train_df = pd.read_csv(food_train)
food_eval_df = pd.read_csv(food_eval)


# Medical Data Frame
# medical_district_df = pd.read_csv(medical_district)
# medical_central_store_df = pd.read_csv(medical_central_store)
# medical_region_df = pd.read_csv(medical_region)

# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================


# Print And View the Data
def view(df: pd.DataFrame):
    if df is None:
        print("the dataframe is not exist")
        return
    print("=" * 50)
    json_df = df.iloc[0].to_dict()
    print(json.dumps(json_df, indent=4, default=str))
    # Basic View
    print("=" * 50)
    print("INFO:")
    print(df.info(verbose=True))
    print("=" * 50)
    print(df.nunique())

    # print("=" * 50)
    # print(f"SHAPE: {df.shape[0]} rows, {df.shape[1]} columns")
    # print("=" * 50)

    # # Missing or Duplicate data report
    # print("DUPLICATE DATA :")
    # print("DUPLICATES: ", df.duplicated().sum())

    # missing_df = df.isnull().mean() * 100
    # print(f"MISSING DATA PERCENTAGES:\n{missing_df}%")
    # print("=" * 50)


def drop_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """
    Drop columns that exist in the DataFrame.
    """
    existing = [col for col in columns if col in df.columns]

    if not existing:
        print("[WARNING!] No matching columns found.")
        return

    df.drop(columns=existing, inplace=True)
    print(f"[DONE] Dropped {len(existing)} column(s): {existing}")

    missing = [col for col in columns if col not in df.columns]
    if missing:
        print(f"[I] Ignored missing column(s): {missing}")

# Inspect rows
def inspect(
    df: pd.DataFrame,
    column: str,
    *,
    equals=None,
    greater=None,
    less=None,
    largest: int | None = None,
    smallest: int | None = None,
    columns: list[str] | None = None,
) -> None:
    """
    Inspect rows satisfying a condition or the largest/smallest values.

    Parameters
    ----------
    column : Column to inspect.
    equals : Show rows where column == value.
    greater : Show rows where column > value.
    less : Show rows where column < value.
    largest : Show top N largest values.
    smallest : Show top N smallest values.
    columns : Columns to display (default: all).
    """

    if columns is None:
        columns = list(df.columns)

    if largest is not None:
        print(df.nlargest(largest, column)[columns])
        return

    if smallest is not None:
        print(df.nsmallest(smallest, column)[columns])
        return

    mask = pd.Series(True, index=df.index)

    if equals is not None:
        mask &= df[column] == equals

    if greater is not None:
        mask &= df[column] > greater

    if less is not None:
        mask &= df[column] < less

    print(df.loc[mask, columns])

# Describe a feature
def describe_feature(
    df: pd.DataFrame,
    column: str,
    top_n: int = 10,
    sample_n: int = 5,
) -> None:
    """
    Display a complete summary of one feature.
    """

    from pandas.api.types import is_numeric_dtype

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    s = df[column]

    print('=' * 40)
    print(f'FEATURE: {column}')
    print('=' * 40)

    print(f'Dtype           : {s.dtype}')
    print(f'Shape           : {len(s):,}')
    print(f'Missing         : {s.isna().sum():,} ({100 * s.isna().mean():.2f}%)')
    print(f'Duplicates      : {s.duplicated().sum():,}')
    print(f'Unique          : {s.nunique():,}')

    if is_numeric_dtype(s):
        print()
        print('STATISTICS')
        print('-' * 40)
        print(s.describe())

        print()
        print(f'Skewness        : {s.skew():.4f}')
        print(f'Kurtosis        : {s.kurtosis():.4f}')

        print()
        print('LARGEST VALUES')
        print('-' * 40)
        print(df.nlargest(top_n, column))

        print()
        print('SMALLEST VALUES')
        print('-' * 40)
        print(df.nsmallest(top_n, column))

    else:
        print()
        print('TOP VALUE COUNTS')
        print('-' * 40)
        print(s.value_counts(dropna=False).head(top_n))

    print()
    print('FIRST SAMPLE')
    print('-' * 40)
    print(json.dumps(df.iloc[0].to_dict(), indent=4, default=str))

    print()
    print(f'RANDOM {sample_n} ROWS')
    print('-' * 50)
    print(df.sample(min(sample_n, len(df))))

    print('=' * 40)
def encode_array(df: pd.DataFrame, column: str, from_string: bool = True) -> None:
    """
    Convert a dataframe column containing lists (or list strings)
    into NumPy arrays.
    [WARNING!] YOU NEED TO IMPORT re and numpy. AND PAY ATTENTION TO YOUR FORMAT
    """

    if from_string:
        def parse(x):
            if pd.isna(x):
                return np.array([])

            x = str(x).strip().strip('[]')

            # Split by commas OR any whitespace (space, tab, newline)
            tokens = re.split(r'[\s,]+', x)

            return np.array([float(i) for i in tokens if i])

        df[column] = df[column].apply(parse)

    else:
        df[column] = df[column].apply(np.array)

def decode_list(
    df: pd.DataFrame,
    stock_column: str):
    '''
    FRESH RETAIL DATASET ONLY
    '''
    if not isinstance(df[stock_column].iloc[0], np.ndarray):
        try:
            encode_array(df, stock_column)
        except Exception as e:
            print(f"[ERROR] This Type of {stock_column} is not a numpe array!\n{e}")
            return

    stock = np.stack(df[stock_column].tolist())

    # ---------------------------
    # Stock Features
    # ---------------------------
    mask = stock == 1

    df["stockout_hours"] = stock.sum(axis=1)

    df["stockout_ratio"] = (
        df["stockout_hours"] / 24
    )


    def longest_stockout(arr):
        longest = current = 0

        for x in arr:
            if x == 1:
                current += 1
                longest = max(longest, current)
            else:
                current = 0

        return longest

    df["longest_stockout"] = df[stock_column].apply(
        longest_stockout
    )

    df["first_stockout_hour"] = np.where(
        mask.any(axis=1),
        mask.argmax(axis=1),
        -1
    )

    rev = np.flip(mask, axis=1)

    df["last_stockout_hour"] = np.where(
        mask.any(axis=1),
        23 - rev.argmax(axis=1),
        -1
    )

    df["business_stockout"] = stock[:,8:18].sum(axis=1)

def engineer_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    FRESH RETAIL DATASET ONLY

    Required Columns
    ----------------
    discount
    stockout_ratio
    stock_hour6_22_cnt
    business_stockout
    holiday_flag
    activity_flag
    precpt
    avg_humidity
    day_of_week
    """

    # ----------------------------------------------------------
    # Promotion × Stock
    # ----------------------------------------------------------

    df["discount_stockout_ratio"] = (
        df["discount"] * df["stockout_ratio"]
    )

    df["discount_stock_hour6_22_cnt"] = (
        df["discount"] * df["stock_hour6_22_cnt"]
    )

    df["discount_business_stockout"] = (
        df["discount"] * df["business_stockout"]
    )

    # ----------------------------------------------------------
    # Calendar × Stock
    # ----------------------------------------------------------

    df["holiday_stockout_ratio"] = (
        df["holiday_flag"] * df["stockout_ratio"]
    )

    df["activity_stockout_ratio"] = (
        df["activity_flag"] * df["stockout_ratio"]
    )

    # ----------------------------------------------------------
    # Weather
    # ----------------------------------------------------------

    df["bad_weather"] = (
        df["precpt"] * df["avg_humidity"]
    )

    # ----------------------------------------------------------
    # Weekend
    # ----------------------------------------------------------

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(np.int8)

    return df

def export_csv(df: pd.DataFrame, filename: str, index: bool = False) -> None:
    """
    Export a DataFrame to a CSV file.
    """
    df.to_csv(filename, index=index)
    print(f"[DONE] Data exported to '{filename}'")
# ==========================================================================
# PLOT / STYLE PLOT
# ==========================================================================
def plot_style() -> None:
    custom = {
        "grid.linestyle": "--",
        "grid.color": "#101010",
        "axes.edgecolor": "#101010",
        "axes.labelcolor": "#101010",
        "xtick.color": "#101010",
        "ytick.color": "#101010",
    }

    sns.set_theme(style="whitegrid", rc=custom)


def plot_histogram(df: pd.DataFrame, columns: list[str], bins: int = 30) -> None:
    """
    Plot histograms for multiple numerical columns.
    """

    plot_style()

    n = len(columns)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))

    axes = np.array(axes).flatten()

    for ax, column in zip(axes, columns):
        ax.hist(
            df[column].dropna(),
            bins=bins,
            color="#123666",
            edgecolor="#101010",
            alpha=0.9,
        )

        ax.set_title(column)
        ax.set_xlabel(column)
        ax.set_ylabel("Count")

    for ax in axes[len(columns) :]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()

def plot_boxplot(df: pd.DataFrame, columns: list[str]) -> None:
    """
    Plot box plots for multiple numerical columns.
    """

    plot_style()

    n = len(columns)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(6 * cols, 3 * rows)
    )

    axes = np.array(axes).flatten()

    for ax, column in zip(axes, columns):
        sns.boxplot(
            x=df[column],
            ax=ax,
            color="#ebc564",
            linewidth=1.5
        )

        ax.set_title(column)
        ax.set_xlabel(column)

    for ax in axes[len(columns):]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()

def plot_correlation(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: Literal["pearson", "spearman", "kendall"]="pearson",
) -> None:
    """
    Plot a lower-triangle correlation heatmap.

    Parameters
    ----------
    columns : Columns to include (default: all numeric columns).
    method  : 'pearson', 'spearman', or 'kendall'.
    """

    plot_style()

    if columns is None:
        corr = df.corr(numeric_only=True, method=method)
    else:
        corr = df[columns].corr(method=method) # type: ignore

    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(max(8, len(corr) * 0.7), max(6, len(corr) * 0.7)))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        linecolor="#101010",
        cbar_kws={"shrink": 0.8}
    )

    plt.title(f"Correlation Heatmap ({method.title()})")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()

def plot_scatter(
    df: pd.DataFrame,
    target: str,
    columns: list[str],
) -> None:
    """
    Plot scatter plots comparing each feature against a target column.
    """
    import math

    plot_style()

    n = len(columns)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(6 * cols, 4 * rows)
    )

    axes = np.array(axes).flatten()

    for ax, column in zip(axes, columns):
        ax.scatter(
            df[column],
            df[target],
            color="#ad2434",
            edgecolors="#101010",
            linewidths=0.3,
            alpha=0.5,
            s=12,
        )

        ax.set_title(column)
        ax.set_xlabel(column)
        ax.set_ylabel(target)

    for ax in axes[len(columns):]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()
# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================
def main():
    # ----------------------------------------------------------
    # Feature Engineering: food_eval and food_train
    # ----------------------------------------------------------

    decode_list(food_eval_df, "hours_stock_status")
    engineer_interaction_features(food_eval_df)
    decode_list(food_train_df, "hours_stock_status")
    engineer_interaction_features(food_train_df)
    # Drop Column
    drop_columns(food_eval_df, [
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
        "hours_stock_status"
    ])
    drop_columns(food_train_df, [
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
        "hours_stock_status"
    ])

    # ----------------------------------------------------------
    # # Histogram
    # plot_histogram(
    #     food_train_df,
    #     [
    #         "sale_amount",
    #         "discount",
    #         "precpt",
    #         "avg_temperature",
    #         "avg_humidity",
    #         "avg_wind_level",
    #     ],
    # )

    # # [NOTE] For training Neural Network or Linear Model...
    # # "sale_amount" and "precpt" is skew_right and the tail is to long, so i apply numpy.log1p() for this feature
    # food_train_df["sale_amount"] = np.log1p(food_train_df["sale_amount"])
    # food_train_df["precpt"] = np.log1p(food_train_df["precpt"])

    # ----------------------------------------------------------
    # Box Plot
    # plot_boxplot(food_train_df, ["sale_amount", "discount", "precpt", "avg_temperature", "avg_humidity", "avg_wind_level"])

    # ----------------------------------------------------------
    # Correlation Heatmap plot
    # plot_correlation(food_eval_df, method="pearson")

    # ----------------------------------------------------------
    # Scatter Plot
    # plot_scatter(food_train_df, "sale_amount", ["stock_hour6_22_cnt","discount", "precpt", "holiday_flag","activity_flag", "avg_temperature", "avg_humidity", "avg_wind_level","day_sin", "day_cos", "day_sin_month", "day_cos_month", "month_sin", "month_cos"])

    # =============================================
    # Feature Engineering: medical
    # =============================================

    # ----------------------------------------------------------
    # view(food_eval_df)
    # inspect(food_train_df, "precpt", greater=20, columns=["avg_temperature", "avg_humidity"])
    # inspect(food_train_df, "discount", less=0.2, columns=["sale_amount", "holiday_flag", "activity_flag"])
    # describe_feature(food_train_df, "sale_amount")
    # export_csv(food_eval_df, FRESH_RETAIL_EVAL_PATH_FEATURE)
    # export_csv(food_train_df, FRESH_RETAIL_TRAIN_PATH_FEATURE)


if __name__ == "__main__":
    main()
