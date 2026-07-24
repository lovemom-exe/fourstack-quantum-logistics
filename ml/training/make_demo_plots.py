# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: PART 4 - demo artifacts for the pitch video.
#
#          Loads the PERSISTED Protocol-P XGBoost baseline (no retraining) and
#          renders three PNGs plus the cold-chain error statistics:
#            1. actual vs predicted over an ordered 200-row test slice
#            2. error histogram (y_pred - y_true) with the mean marked
#            3. y_true vs y_pred scatter with the y=x reference line
#
#          The cold-chain angle: under-forecasting leads to under-ordering,
#          which leads to stockout. So the SIGN of the error matters, not just
#          its magnitude - hence the explicit bias / under-forecast stats.
#
#          Local only - ZERO QuApp cloud calls.
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from training.eval_ksweep import (
    AXIS,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SERIES,
    SURFACE,
    build_test_set,
)
from training.train_protocol_p import BUNDLE_PATH
from utils.path import TRAINING_EVA_RESULT

# ==========================================================================
# PARAMETERS
# ==========================================================================
SLICE_N = 200
ACTUAL_COLOR = SERIES[1000]    # categorical slot 1 (blue)
PRED_COLOR = SERIES[50000]     # categorical slot 2 (orange)
STATS_PATH = os.path.join(TRAINING_EVA_RESULT, "demo_error_stats.json")


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, pad=12, loc="left")
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    ax.grid(True, color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def load_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Re-apply the SAVED Protocol-P model to the shared test set. No retraining."""
    bundle = joblib.load(BUNDLE_PATH)
    X_test_pool, y_test_raw = build_test_set()

    keep = bundle["keep_idx_in_pool"]
    X_test = bundle["x_scaler"].transform(X_test_pool[:, keep])
    pred_real = np.expm1(np.asarray(bundle["model"].predict(X_test)).ravel())
    return y_test_raw, pred_real


def plot_slice(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=SURFACE)
    _style(ax, f"Thuc te vs du bao — lat {SLICE_N} dong test (giao thuc P)",
           "chi so dong trong lat test", "sale_amount")

    idx = np.arange(SLICE_N)
    ax.plot(idx, y_true[:SLICE_N], color=ACTUAL_COLOR, linewidth=2, label="Thuc te")
    ax.plot(idx, y_pred[:SLICE_N], color=PRED_COLOR, linewidth=2, label="Du bao")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9, loc="upper right")
    fig.tight_layout()

    path = os.path.join(TRAINING_EVA_RESULT, "demo_actual_vs_pred_slice.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_error_hist(errors: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    _style(ax, "Phan bo sai so du bao (y_pred - y_true) — giao thuc P",
           "sai so (duong = du bao thua, am = du bao thieu)", "so luong")

    # The error tail runs to about -30; plotting the full range squeezes all the
    # mass into a few pixels. Clip the VIEW to the central 99% and disclose how
    # many rows fall outside it.
    lo, hi = np.percentile(errors, [0.5, 99.5])
    pad = 0.1 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    outside = int(np.sum((errors < lo) | (errors > hi)))

    ax.hist(errors, bins=80, range=(lo, hi), color=ACTUAL_COLOR,
            edgecolor=SURFACE, linewidth=0.5)
    mean_err = float(np.mean(errors))
    ax.axvline(0, color=AXIS, linewidth=1, linestyle="--", zorder=4)
    ax.axvline(mean_err, color=INK_PRIMARY, linewidth=2, linestyle="-", zorder=5)
    ax.set_xlim(lo, hi)

    # Anchor the label in the upper-left so it never sits on top of the bars.
    ax.annotate(
        f"trung binh = {mean_err:+.3f}  (du bao thieu)",
        xy=(0.02, 0.95), xycoords="axes fraction", color=INK_PRIMARY, fontsize=10,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=SURFACE,
                  edgecolor=AXIS, linewidth=1),
    )
    ax.annotate(
        f"{outside:,}/{len(errors):,} diem nam ngoai khung nhin",
        xy=(0.02, 0.86), xycoords="axes fraction", color=INK_MUTED, fontsize=8.5,
        va="top", ha="left",
    )
    fig.tight_layout()

    path = os.path.join(TRAINING_EVA_RESULT, "demo_error_hist.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 6.5), facecolor=SURFACE)
    _style(ax, "Thuc te vs du bao (giao thuc P)", "y_true", "y_pred")

    ax.scatter(y_true, y_pred, s=10, alpha=0.25, color=ACTUAL_COLOR,
               edgecolors="none", zorder=3)
    lim = float(max(y_true.max(), y_pred.max())) * 1.02
    ax.plot([0, lim], [0, lim], color=INK_PRIMARY, linewidth=1.5, linestyle="--",
            zorder=4, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9, loc="upper left")
    fig.tight_layout()

    path = os.path.join(TRAINING_EVA_RESULT, "demo_scatter.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================
def main() -> None:
    os.makedirs(TRAINING_EVA_RESULT, exist_ok=True)
    y_true, y_pred = load_predictions()
    errors = y_pred - y_true

    under = errors < 0
    stats = {
        "bias_mean_signed_error": float(np.mean(errors)),
        "under_forecast_rate": float(np.mean(under)),
        "mean_under_forecast_magnitude": float(np.mean(-errors[under])) if under.any() else 0.0,
        "over_forecast_rate": float(np.mean(~under)),
        "median_signed_error": float(np.median(errors)),
        "n_test": int(len(y_true)),
    }

    print("=" * 74)
    print("PART 4 - thong ke sai so (giao thuc P, model da luu, KHONG train lai)")
    print("=" * 74)
    print(f"  So dong test                     : {stats['n_test']:,}")
    print(f"  Sai so trung binh co dau (bias)  : {stats['bias_mean_signed_error']:+.4f}")
    print(f"  Trung vi sai so co dau           : {stats['median_signed_error']:+.4f}")
    print(f"  Ty le ca DU BAO THIEU            : {stats['under_forecast_rate'] * 100:.1f}%")
    print(f"  Muc thieu trung binh (khi thieu) : {stats['mean_under_forecast_magnitude']:.4f}")
    print(f"  Ty le ca du bao thua             : {stats['over_forecast_rate'] * 100:.1f}%")
    print("=" * 74)

    p1 = plot_slice(y_true, y_pred)
    p2 = plot_error_hist(errors)
    p3 = plot_scatter(y_true, y_pred)

    with open(STATS_PATH, "w") as fh:
        json.dump(stats, fh, indent=2)

    print(f"[DONE] {p1}")
    print(f"[DONE] {p2}")
    print(f"[DONE] {p3}")
    print(f"[DONE] {STATS_PATH}")
    print("[DONE] Zero QuApp cloud calls.")


if __name__ == "__main__":
    main()
