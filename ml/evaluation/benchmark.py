# ==========================================================================# Author: Hoang Anh Quan
# Purpose: Do a Benchmark between models
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)

# from sklearn.feature_selection import SelectKBest, mutual_info_regression

from preprocessing.split import split
from algorithms.xgboost import xgboost
from algorithms.vqr import vqr
from training.ml_train import (
    create_food_sample,
    feature_scaler,
    y_feature_scaler,
    create_sample_v2,
)
from utils.path import BENCHMARK_FOOD, PERISHABLE_GOODS_BM, PERISHABLE_GOODS_DATA_PATH

# ==========================================================================
# PARAMETERS
# ==========================================================================
# Perishable Data
perishable_data = pd.read_csv(PERISHABLE_GOODS_DATA_PATH)
X_perish_train, X_perish_test, y_perish_train, y_perish_test = split(perishable_data, 0)

# Feature List
feature_list = [
    [
        "shelf_life_days",
        "cost_price",
        "spoilage_sensitivity",
    ],
    [
        "shelf_life_days",
        "cost_price",
        "spoilage_sensitivity",
        "spoilage_risk",
        "selling_price",
        "discount_pct",
        "is_promoted",
    ],
    [
        "shelf_life_days",
        "cost_price",
        "spoilage_sensitivity",
        "spoilage_risk",
        "selling_price",
        "discount_pct",
        "is_promoted",
        "supplier_score",
        "storage_temp",
    ],
    [
        "shelf_life_days",
        "cost_price",
        "spoilage_sensitivity",
        "spoilage_risk",
        "selling_price",
        "discount_pct",
        "is_promoted",
        "supplier_score",
        "storage_temp",
        "region_Midwest",
        "region_Northeast",
        "region_Southeast",
        "region_Southwest",
        "region_West",
    ],
    [
        "shelf_life_days",
        "cost_price",
        "spoilage_sensitivity",
        "spoilage_risk",
        "selling_price",
        "discount_pct",
        "is_promoted",
        "supplier_score",
        "storage_temp",
        "category_Bakery",
        "category_Beverages",
        "category_Dairy",
        "category_Deli",
        "category_Frozen_Meals",
        "category_Meat",
        "category_Pharmaceuticals",
        "category_Produce",
        "category_Ready_to_Eat",
        "category_Seafood",
        "region_Midwest",
        "region_Northeast",
        "region_Southeast",
        "region_Southwest",
        "region_West",
    ],
]
feature_list_index = [
    [0, 1, 2],
    [0, 1, 2, 3, 4, 5, 6],
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
]

sample_number = 200
is_scale_x = True
is_scale_y = True

X_train, X_test, y_train, y_test = create_sample_v2(
    X_perish_train, X_perish_test, y_perish_train, y_perish_test, n_sample=sample_number
)

scale_name_path = "xy"
if is_scale_x and not is_scale_y:
    scale_name_path = "x"
elif is_scale_y and not is_scale_x:
    scale_name_path = "y"
elif not is_scale_x and not is_scale_y:
    scale_name_path = "non"

xgboost_score_path = os.path.join(
    PERISHABLE_GOODS_BM, f"xgboost/score_{sample_number}s_{scale_name_path}.csv"
)
vqr_score_path = os.path.join(
    PERISHABLE_GOODS_BM, f"vqr/score_{sample_number}s_{scale_name_path}.csv"
)
export_data_path = vqr_score_path

vqr_bm_data_path = vqr_score_path
xgboost_bm_data_path = xgboost_score_path


error_metrics = ["r2_score", "mae", "mse", "rmse"]


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
def export_csv(df: pd.DataFrame, filename: str, index: bool = False) -> None:
    """
    Export a DataFrame to a CSV file.
    """
    df.to_csv(filename, index=index)
    print(f"[DONE] Data exported to '{filename}'")


def benchmark_error(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    results_list = []

    for k in range(4, 5):
        print("=" * 40)
        feature_num = len(feature_list_index[k])
        print(f"[{feature_num}] training:")
        # selector = SelectKBest(score_func=mutual_info_regression, k=k)
        # X_train_k = selector.fit_transform(X_train_all, y_train)
        # X_test_k = selector.transform(X_test_all)

        # print("XGBoost: ", end="")
        # model_xgb = xgboost(X_train[:, feature_list_index[k]], y_train)
        # if model_xgb is not None:
        #     print("Done")
        #     y_pred_xgb = model_xgb.predict(X_test[:, feature_list_index[k]])
        #     results_list.append(
        #         {
        #             "model_name": "XGBoost",
        #             "feature_number": feature_num,
        #             "r2_score": r2_score(y_test, y_pred_xgb),
        #             "mae": mean_absolute_error(y_test, y_pred_xgb),
        #             "mse": mean_squared_error(y_test, y_pred_xgb),
        #             "rmse": root_mean_squared_error(y_test, y_pred_xgb),
        #         }
        #     )
        # else:
        #     print("Fail")
        #     return
        # print("=" * 20)

        print("VQR: ", end="")
        model_vqr = vqr(
            X_train[:, feature_list_index[k]],
            y_train,
            k=feature_num,
        )
        if model_vqr is not None:
            print("Done")
            if isinstance(X_test[:, feature_list_index[k]], np.ndarray):
                y_pred_vqr = model_vqr.predict(X_test[:, feature_list_index[k]])
                results_list.append(
                    {
                        "model_name": "VQR",
                        "feature_number": feature_num,
                        "r2_score": r2_score(y_test, y_pred_vqr),
                        "mae": mean_absolute_error(y_test, y_pred_vqr),
                        "mse": mean_squared_error(y_test, y_pred_vqr),
                        "rmse": root_mean_squared_error(y_test, y_pred_vqr),
                    }
                )
        else:
            print("Fail")
            return

        # Export Error Benchmark Data
        df_results = pd.DataFrame(results_list)
        os.makedirs(os.path.dirname(export_data_path), exist_ok=True)
        export_csv(df_results, export_data_path)
    print("Done")


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


def benchmark_plots(csv_path: str) -> None:
    """
    Read benchmark CSV and generate independent line plots for each error metric.
    """
    if not os.path.exists(csv_path):
        print(f"[ERROR] This Path doesn't exis: '{csv_path}'")
        return

    df = pd.read_csv(csv_path)

    df_xgb_raw = df[df["model_name"] == "XGBoost"]
    df_vqr_raw = df[df["model_name"] == "VQR"]

    df_xgb = pd.DataFrame(df_xgb_raw).sort_values(by="feature_number")
    df_vqr = pd.DataFrame(df_vqr_raw).sort_values(by="feature_number")

    k_ticks = sorted(df["feature_number"].unique())

    plot_style()
    # columns = 3
    # rows = len(error_metrics) // columns + 1

    # fig, axes = plt.subplots(rows, columns, figsize=(6 * rows, 4 * columns))

    # axes = np.array(axes).flatten()

    # for ax, metric in zip(axes, error_metrics):
    for metric in error_metrics:
        if metric not in df.columns:
            continue

        # plt.figure(figsize=(9, 5.5))

        # ax.plot(
        #     df_xgb["feature_number"],
        #     df_xgb[metric],
        #     marker="o",
        #     linewidth=2,
        #     color="#d62728",
        #     label="XGBoost",
        # )

        # ax.plot(
        #     df_vqr["feature_number"],
        #     df_vqr[metric],
        #     marker="s",
        #     linewidth=2,
        #     linestyle="--",
        #     color="#1f77b4",
        #     label="VQR",
        # )

        # metric_title = metric.replace("_", " ").upper()
        # ax.set_title(
        #     f"{metric_title}",
        #     fontsize=13,
        #     fontweight="bold",
        #     pad=15,
        # )
        # ax.set_xlabel("k", fontsize=11, fontweight="semibold")
        # ax.set_ylabel(metric_title, fontsize=11, fontweight="semibold")

        # ax.set_xticks(k_ticks)
        # ax.legend(frameon=True, facecolor="white", edgecolor="#101010", fontsize=10)
        # ax.tight_layout()
        plt.plot(
            df_xgb["feature_number"],
            df_xgb[metric],
            marker="o",
            linewidth=2,
            color="#d62728",
            label="XGBoost",
        )

        plt.plot(
            df_vqr["feature_number"],
            df_vqr[metric],
            marker="s",
            linewidth=2,
            linestyle="--",
            color="#1f77b4",
            label="VQR",
        )

        metric_title = metric.replace("_", " ").upper()
        plt.title(
            f"{metric_title}",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        plt.xlabel("k", fontsize=11, fontweight="semibold")
        plt.ylabel(metric_title, fontsize=11, fontweight="semibold")

        plt.xticks(k_ticks)
        plt.legend(frameon=True, facecolor="white", edgecolor="#101010", fontsize=10)
        plt.tight_layout()

        plt.show()


def benchmark_plots_v2(csv_path_xgb: str, csv_path_vqr: str) -> None:
    """
    Read benchmark CSV and generate independent line plots for each error metric.
    """
    if not os.path.exists(csv_path_xgb):
        print(f"[ERROR] This Path doesn't exis: '{csv_path_xgb}'")
        return
    if not os.path.exists(csv_path_vqr):
        print(f"[ERROR] This Path doesn't exis: '{csv_path_vqr}'")
        return

    df_xgb = pd.read_csv(csv_path_xgb)
    df_vqr = pd.read_csv(csv_path_vqr)

    plot_style()
    # columns = 3
    # rows = len(error_metrics) // columns + 1

    # fig, axes = plt.subplots(rows, columns, figsize=(6 * rows, 4 * columns))

    # axes = np.array(axes).flatten()

    # for ax, metric in zip(axes, error_metrics):
    for metric in error_metrics:

        # plt.figure(figsize=(9, 5.5))

        # ax.plot(
        #     df_xgb["feature_number"],
        #     df_xgb[metric],
        #     marker="o",
        #     linewidth=2,
        #     color="#d62728",
        #     label="XGBoost",
        # )

        # ax.plot(
        #     df_vqr["feature_number"],
        #     df_vqr[metric],
        #     marker="s",
        #     linewidth=2,
        #     linestyle="--",
        #     color="#1f77b4",
        #     label="VQR",
        # )

        # metric_title = metric.replace("_", " ").upper()
        # ax.set_title(
        #     f"{metric_title}",
        #     fontsize=13,
        #     fontweight="bold",
        #     pad=15,
        # )
        # ax.set_xlabel("k", fontsize=11, fontweight="semibold")
        # ax.set_ylabel(metric_title, fontsize=11, fontweight="semibold")

        # ax.set_xticks(k_ticks)
        # ax.legend(frameon=True, facecolor="white", edgecolor="#101010", fontsize=10)
        # ax.tight_layout()
        plt.plot(
            df_xgb["feature_number"],
            df_xgb[metric],
            marker="o",
            linewidth=2,
            color="#d62728",
            label="XGBoost",
        )

        plt.plot(
            df_vqr["feature_number"],
            df_vqr[metric],
            marker="s",
            linewidth=2,
            linestyle="--",
            color="#1f77b4",
            label="VQR",
        )

        metric_title = metric.replace("_", " ").upper()
        plt.title(
            f"{metric_title}",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        plt.xlabel("k", fontsize=11, fontweight="semibold")
        plt.ylabel(metric_title, fontsize=11, fontweight="semibold")

        plt.legend(frameon=True, facecolor="white", edgecolor="#101010", fontsize=10)
        plt.tight_layout()

        plt.show()


# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================


def main():

    global X_train, X_test, y_train, y_test

    if is_scale_x:
        X_train, X_test = feature_scaler(X_train, X_test)
    if is_scale_y:
        y_train, y_test = y_feature_scaler(y_train, y_test)

    # benchmark_error(
    #     X_train=X_train,
    #     y_train=y_train,
    #     X_test=X_test,
    #     y_test=y_test,
    # )
    benchmark_plots_v2(xgboost_bm_data_path, vqr_bm_data_path)


if __name__ == "__main__":
    main()
