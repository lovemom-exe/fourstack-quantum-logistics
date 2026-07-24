# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: PART 3 - Protocol-Q XGBoost control, matched to the VQR runs.
#
#          The Part-B grid trained XGBoost on 1,000 rows, while VQR trains on
#          the 800-row train split (1,000 minus the 20% validation slice). To
#          make the quantum-vs-classical comparison exactly fair, this control
#          reuses train_vqr_ksweep.prepare(k) verbatim, so it gets the IDENTICAL
#          split, feature selection, feature scaler and target scaler as the
#          VQR run at that k. Only the learner differs.
#
#          Protocol Q: winsorize(train, p99) -> log1p -> MinMaxScaler(-1,1),
#          inverted at prediction time. Scored on the shared 10k test set.
#
#          Produces the definitive VQR-vs-XGBoost figure (both Protocol Q).
#
#          Local only - ZERO QuApp cloud calls.
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from algorithms.xgboost import xgboost
from evaluation.metrics import mae, mape, r2, rmse
from training.eval_ksweep import (
    AXIS,
    INK_SECONDARY,
    SERIES,
    SURFACE,
    TEST_N,
    VQR_COLOR,
    _style_axes,
    build_test_set,
    diagnostics,
)
from training.train_vqr_ksweep import K_GRID_VQR, RESULTS_JSON, prepare
from utils.path import TRAINING_EVA_RESULT

# ==========================================================================
# PARAMETERS
# ==========================================================================
CONTROL_CSV = os.path.join(TRAINING_EVA_RESULT, "xgb_control_q.csv")
CONTROL_JSON = os.path.join(TRAINING_EVA_RESULT, "xgb_control_q.json")


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
def run_control(k: int, X_test_pool: np.ndarray, y_test_raw: np.ndarray) -> dict:
    """XGBoost on the EXACT split/transforms the VQR run at this k used."""
    data = prepare(k, X_test_pool)

    model = xgboost(data["X_tr"], data["y_tr_scaled"])
    if model is None:
        raise RuntimeError(f"XGBoost control failed at k={k}.")

    pred_scaled = np.asarray(model.predict(data["X_test"])).ravel()
    pred_log = data["y_scaler"].inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    pred_real = np.expm1(pred_log)
    y_test_log = np.log1p(y_test_raw)

    d_real = diagnostics(y_test_raw, pred_real)
    d_log = diagnostics(y_test_log, pred_log)

    return {
        "model": "XGBoost", "protocol": "Q",
        "n_train": len(data["y_tr_scaled"]), "k": k,
        "r2_real": r2(y_test_raw, pred_real), "r2_log": r2(y_test_log, pred_log),
        "mae": mae(y_test_raw, pred_real), "rmse": rmse(y_test_raw, pred_real),
        "mape": mape(y_test_raw, pred_real),
        "std_ratio_real": d_real["std_ratio"], "r_real": d_real["r"],
        "std_ratio_log": d_log["std_ratio"], "r_log": d_log["r"],
        "selected": ";".join(data["selected"]),
    }


def plot_matched(vqr_rows: list[dict], xgb_rows: list[dict], metric: str,
                 ylabel: str, title: str, filename: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    _style_axes(ax, title, ylabel)

    for rows, color, label, marker in (
        (xgb_rows, SERIES[1000], "XGBoost (giao thuc Q)", "o"),
        (vqr_rows, VQR_COLOR, "VQR (giao thuc Q)", "s"),
    ):
        pts = sorted(rows, key=lambda r: r["k"])
        xs = [p["k"] for p in pts]
        ys = [p[metric] for p in pts]
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=8,
                label=label, zorder=3)
        ax.annotate(label.split(" ")[0], (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, 0), color=INK_SECONDARY, fontsize=9, va="center")

    ax.axhline(0, color=AXIS, linewidth=1, linestyle="--", zorder=1)
    ax.set_xticks(K_GRID_VQR)
    ax.set_xlabel("so qubit / so feature (k)", color=INK_SECONDARY, fontsize=10)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9, loc="best")
    fig.tight_layout()
    path = os.path.join(TRAINING_EVA_RESULT, filename)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================
def main() -> None:
    os.makedirs(TRAINING_EVA_RESULT, exist_ok=True)
    X_test_pool, y_test_raw = build_test_set()

    with open(RESULTS_JSON) as fh:
        vqr_rows = json.load(fh)
    done_k = [r["k"] for r in vqr_rows]
    print(f"[SETUP] VQR results available for k={done_k}")

    xgb_rows = [run_control(k, X_test_pool, y_test_raw) for k in done_k]

    print("=" * 104)
    print(f"PART 3 - VQR vs XGBoost, BOTH Protocol Q, matched 800-row split, "
          f"shared {TEST_N:,}-row test set")
    print("=" * 104)
    print(f"{'model':<9}{'k':>3}{'R2(real)':>11}{'R2(log)':>10}{'MAE':>9}{'RMSE':>9}"
          f"{'std ratio':>11}{'r':>9}")
    print("-" * 104)
    for k in done_k:
        v = next(r for r in vqr_rows if r["k"] == k)
        x = next(r for r in xgb_rows if r["k"] == k)
        for tag, r in (("VQR", v), ("XGBoost", x)):
            print(f"{tag:<9}{k:>3}{r['r2_real']:>11.4f}{r['r2_log']:>10.4f}"
                  f"{r['mae']:>9.4f}{r['rmse']:>9.4f}{r['std_ratio_real']:>11.4f}"
                  f"{r['r_real']:>9.4f}")
        print("-" * 104)
    print("=" * 104)

    pd.DataFrame(xgb_rows).to_csv(CONTROL_CSV, index=False)
    with open(CONTROL_JSON, "w") as fh:
        json.dump(xgb_rows, fh, indent=2)

    p1 = plot_matched(vqr_rows, xgb_rows, "r2_real", "R² (don vi that)",
                      f"VQR vs XGBoost theo so qubit — giao thuc Q, test {TEST_N:,} dong",
                      "vqr_vs_xgb_q_r2_real.png")
    p2 = plot_matched(vqr_rows, xgb_rows, "r2_log", "R² (log space)",
                      f"VQR vs XGBoost theo so qubit — giao thuc Q (log), test {TEST_N:,} dong",
                      "vqr_vs_xgb_q_r2_log.png")

    print(f"[DONE] control CSV : {CONTROL_CSV}")
    print(f"[DONE] plot        : {p1}")
    print(f"[DONE] plot        : {p2}")
    print("[DONE] Zero QuApp cloud calls.")


if __name__ == "__main__":
    main()
