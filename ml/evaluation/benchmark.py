# ==========================================================================
# Author: Hoang Anh Quan
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
from sklearn.feature_selection import SelectKBest, mutual_info_regression

from training.ml_train import create_food_sample, feature_scaler, vqr_train, xgb_train
from utils.path import BENCHMARK_FOOD

# ==========================================================================
# PARAMETERS
# ==========================================================================
feature_list = [4, 8, 12, 16, 20, 24, 27]
error_benchmark_result_path = os.path.join(BENCHMARK_FOOD, "error_benchmark.csv")

error_metrics = ["r2_score", "mae", "mse", "rmse", "mape"]


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
def export_csv(df: pd.DataFrame, filename: str, index: bool = False) -> None:
    """
    Export a DataFrame to a CSV file.
    """
    df.to_csv(filename, index=index)
    print(f"[DONE] Data exported to '{filename}'")


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def benchmark_error(
    X_train_all: np.ndarray,
    y_train: np.ndarray,
    X_test_all: np.ndarray,
    y_test: np.ndarray,
):
    results_list = []

    for k in feature_list:
        print("=" * 40)
        print(f"[{k}] training:")
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        X_train_k = selector.fit_transform(X_train_all, y_train)
        X_test_k = selector.transform(X_test_all)

        print("XGBoost: ", end="")
        model_xgb = xgb_train(X_train_k, y_train)
        if model_xgb is not None:
            print("Done")
            y_pred_xgb = model_xgb.predict(X_test_k)
            results_list.append(
                {
                    "model_name": "XGBoost",
                    "feature_number": k,
                    "r2_score": r2_score(y_test, y_pred_xgb),
                    "mae": mean_absolute_error(y_test, y_pred_xgb),
                    "mse": mean_squared_error(y_test, y_pred_xgb),
                    "rmse": root_mean_squared_error(y_test, y_pred_xgb),
                    "mape": calculate_mape(y_test, y_pred_xgb),
                }
            )
        else:
            print("Fail")
            return
        print("=" * 20)

        # print("VQR: ", end="")
        # model_vqr = vqr_train(X_train_k, y_train, k=k)
        # if model_vqr is not None:
        #     print("Done")
        #     if isinstance(X_test_k, np.ndarray):
        #         y_pred_vqr = model_vqr.predict(X_test_k)
        #         results_list.append(
        #             {
        #                 "model_name": "VQR",
        #                 "feature_number": k,
        #                 "r2_score": r2_score(y_test, y_pred_vqr),
        #                 "mae": mean_absolute_error(y_test, y_pred_vqr),
        #                 "mse": mean_squared_error(y_test, y_pred_vqr),
        #                 "rmse": root_mean_squared_error(y_test, y_pred_vqr),
        #                 "mape": calculate_mape(y_test, y_pred_vqr),
        #             }
        #         )
        # else:
        #     print("Fail")
        #     return

        # Export Error Benchmark Data
        df_results = pd.DataFrame(results_list)
        os.makedirs(os.path.dirname(error_benchmark_result_path), exist_ok=True)
        export_csv(df_results, error_benchmark_result_path)
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
    columns = 3
    rows = len(error_metrics) // columns + 1

    fig, axes = plt.subplots(rows, columns, figsize=(6 * rows, 4 * columns))

    axes = np.array(axes).flatten()

    for ax, metric in zip(axes, error_metrics):
        if metric not in df.columns:
            continue

        # plt.figure(figsize=(9, 5.5))

        ax.plot(
            df_xgb["feature_number"],
            df_xgb[metric],
            marker="o",
            linewidth=2,
            color="#d62728",
            label="XGBoost",
        )

        ax.plot(
            df_vqr["feature_number"],
            df_vqr[metric],
            marker="s",
            linewidth=2,
            linestyle="--",
            color="#1f77b4",
            label="VQR",
        )

        metric_title = metric.replace("_", " ").upper()
        ax.set_title(
            f"{metric_title}",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel("k", fontsize=11, fontweight="semibold")
        ax.set_ylabel(metric_title, fontsize=11, fontweight="semibold")

        ax.set_xticks(k_ticks)
        ax.legend(frameon=True, facecolor="white", edgecolor="#101010", fontsize=10)
        # ax.tight_layout()

    plt.show()


# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================


def main():
    X_train_raw, y_train, X_test_raw, y_test = create_food_sample(n_sample=10)
    X_train_scaled, X_test_scaled = feature_scaler(X_train_raw, X_test_raw)

    benchmark_error(
        X_train_all=X_train_scaled,
        y_train=y_train,
        X_test_all=X_test_scaled,
        y_test=y_test,
    )
    # benchmark_plots(error_benchmark_result_path)


if __name__ == "__main__":
    main()
