# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: XGBoost same-day "nowcasting" baseline for sale_amount.
#
#          NOTE: several engineered features (stockout ratio, stock-hour
#          counts, etc.) describe the SAME day being predicted, and
#          sale_amount is itself censored by stockout. That makes this a
#          same-day/contemporaneous regression baseline, not a forecast.
#          This script runs two variants to quantify how much of the signal
#          is contemporaneous stock information:
#            - "full": all engineered features
#            - "leakage_free": same-day stock/stockout columns excluded
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler

from algorithms.xgboost import xgboost
from evaluation.metrics import mae, mape, r2, rmse
from training.ml_train import DEMAND, create_food_sample, food_train_df
from utils.path import TRAINING_EVA_RESULT

# ==========================================================================
# PARAMETERS
# ==========================================================================
N_CLASSICAL = 50000
K_FEATURES = 20
RANDOM_STATE = 42

# Same-day stock/stockout columns (and interactions built from them). Kept in
# the "full" variant, excluded in the "leakage_free" variant.
SAME_DAY_STOCK_COLUMNS = [
    "stockout_ratio",
    "stockout_hours",
    "longest_stockout",
    "first_stockout_hour",
    "last_stockout_hour",
    "business_stockout",
    "stock_hour6_22_cnt",
    "discount_stockout_ratio",
    "discount_stock_hour6_22_cnt",
    "discount_business_stockout",
    "holiday_stockout_ratio",
    "activity_stockout_ratio",
]

VARIANTS = [
    {"label": "full", "exclude": []},
    {"label": "leakage_free", "exclude": SAME_DAY_STOCK_COLUMNS},
]

ALL_FEATURE_COLUMNS = food_train_df.drop(columns=[DEMAND]).columns.tolist()


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
def _select_columns(exclude_columns: list) -> tuple[list[int], list[str]]:
    keep = [(i, c) for i, c in enumerate(ALL_FEATURE_COLUMNS) if c not in exclude_columns]
    keep_idx = [i for i, _ in keep]
    keep_names = [c for _, c in keep]
    return keep_idx, keep_names


def _gain_importances(model, selected_names: list[str]) -> list[tuple[str, float]]:
    gain_scores = model.get_booster().get_score(importance_type="gain")
    by_index = {int(f[1:]): score for f, score in gain_scores.items()}
    importances = [(name, by_index.get(i, 0.0)) for i, name in enumerate(selected_names)]
    return sorted(importances, key=lambda x: x[1], reverse=True)


def _print_report(metrics: dict, importances: list, y_test: np.ndarray, y_pred: np.ndarray) -> None:
    print("=" * 60)
    print(f"XGBoost same-day nowcasting baseline [{metrics['label']}]")
    print(f"n = {metrics['n']}, k = {metrics['k']}")
    print(f"R^2:  {metrics['r2']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print("-" * 60)
    print("Top feature importances (gain):")
    for name, score in importances[:15]:
        print(f"  {name:35s} {score:12.4f}")
    print("-" * 60)
    sample_n = min(10, len(y_test))
    print(f"Sample predictions (first {sample_n}):")
    for i in range(sample_n):
        print(f"  y_true={y_test[i]:10.2f}  y_pred={y_pred[i]:10.2f}")
    print("=" * 60)


def _save_outputs(y_test: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    os.makedirs(TRAINING_EVA_RESULT, exist_ok=True)

    results_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    csv_path = os.path.join(TRAINING_EVA_RESULT, f"xgboost_predictions_{label}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"[DONE] Predictions saved to {csv_path}")

    plt.figure(figsize=(7, 7))
    plt.scatter(
        y_test, y_pred, alpha=0.4, s=12, color="#123666", edgecolors="#101010", linewidths=0.3
    )
    lims = [0, max(float(np.max(y_test)), float(np.max(y_pred)))]
    plt.plot(lims, lims, color="#d62728", linestyle="--", linewidth=1.5)
    plt.xlabel("Actual sale_amount")
    plt.ylabel("Predicted sale_amount")
    plt.title(f"XGBoost nowcasting baseline: actual vs predicted [{label}]")
    plt.tight_layout()
    plot_path = os.path.join(TRAINING_EVA_RESULT, f"xgboost_actual_vs_predicted_{label}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[DONE] Plot saved to {plot_path}")


def run(
    n_classical: int = N_CLASSICAL,
    k: int = K_FEATURES,
    exclude_columns: list | None = None,
    label: str = "full",
    random_state: int = RANDOM_STATE,
) -> dict:
    exclude_columns = exclude_columns or []

    X_train_raw, y_train, X_test_raw, y_test = create_food_sample(
        n_sample=n_classical, random_state=random_state
    )

    keep_idx, keep_names = _select_columns(exclude_columns)
    X_train_raw = X_train_raw[:, keep_idx]
    X_test_raw = X_test_raw[:, keep_idx]

    k_eff = min(k, X_train_raw.shape[1])
    selector = SelectKBest(score_func=mutual_info_regression, k=k_eff)
    X_train_sel = selector.fit_transform(X_train_raw, y_train)
    X_test_sel = selector.transform(X_test_raw)
    selected_names = [keep_names[i] for i in selector.get_support(indices=True)]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    y_train_log = np.log1p(y_train)

    model = xgboost(X_train_scaled, y_train_log)
    if model is None:
        raise RuntimeError("XGBoost training failed (invalid input types).")

    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)

    metrics = {
        "label": label,
        "n": n_classical,
        "k": k_eff,
        "r2": r2(y_test, y_pred),
        "mae": mae(y_test, y_pred),
        "rmse": rmse(y_test, y_pred),
        "mape": mape(y_test, y_pred),
    }
    importances = _gain_importances(model, selected_names)

    _print_report(metrics, importances, y_test, y_pred)
    _save_outputs(y_test, y_pred, label)

    return {"metrics": metrics, "importances": importances}


def _evaluate_sanity_gates(full_result: dict, leak_free_result: dict) -> None:
    full_metrics = full_result["metrics"]
    full_importances = full_result["importances"]
    leak_free_metrics = leak_free_result["metrics"]

    print("=" * 60)
    print("SANITY GATE SUMMARY")
    print("=" * 60)

    tripped = False

    if full_metrics["r2"] < 0:
        tripped = True
        print(f"[FAIL] 'full' R^2 = {full_metrics['r2']:.4f} < 0 -> measurement bug likely remains.")
    else:
        print(f"[ok]   'full' R^2 = {full_metrics['r2']:.4f} >= 0.")

    top3 = {name for name, _ in full_importances[:3]}
    dominated_by_stock = bool(top3 & set(SAME_DAY_STOCK_COLUMNS))
    if full_metrics["r2"] > 0.9 or dominated_by_stock:
        tripped = True
        print(
            f"[FLAG] 'full' R^2 = {full_metrics['r2']:.4f} (>0.9: {full_metrics['r2'] > 0.9}); "
            f"top-3 gain importances dominated by same-day stock/stockout features: "
            f"{dominated_by_stock} (top-3: {sorted(top3)}) -> likely a nowcasting artifact "
            "from contemporaneous stockout info, not a genuine forecasting result."
        )
    else:
        print("[ok]   'full' R^2 <= 0.9 and top-3 importances not dominated by same-day stock features.")

    if leak_free_metrics["r2"] <= 0:
        tripped = True
        print(
            f"[FLAG] 'leakage_free' R^2 = {leak_free_metrics['r2']:.4f} <= 0 -> nearly all signal "
            "in the 'full' variant was contemporaneous same-day stock info, not genuine "
            "forecasting signal from calendar/weather/promo features."
        )
    else:
        print(f"[ok]   'leakage_free' R^2 = {leak_free_metrics['r2']:.4f} > 0.")

    print("=" * 60)
    print(
        "[RESULT] One or more sanity-gate conditions tripped - see flags above."
        if tripped
        else "[RESULT] No sanity-gate conditions tripped."
    )
    print("=" * 60)


# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================
def main() -> None:
    results = {}
    for variant in VARIANTS:
        results[variant["label"]] = run(
            n_classical=N_CLASSICAL,
            k=K_FEATURES,
            exclude_columns=variant["exclude"],
            label=variant["label"],
        )

    _evaluate_sanity_gates(results["full"], results["leakage_free"])


if __name__ == "__main__":
    main()
