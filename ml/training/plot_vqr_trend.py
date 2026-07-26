# ==========================================================================
# Author: Nguyen Minh Hoang
# Purpose: PART C (charts) - VQR R^2 trend vs qubit count k, plus a PROJECTION
#          (linear extrapolation) out to k=12.
#
#          Measured and projected regions are drawn differently on purpose:
#            - MEASURED  : square markers, SOLID line, saturated colour. Read
#                          straight from the results CSV; no value is hardcoded.
#            - PROJECTED : DASHED line, pale colour, with an uncertainty band
#                          (+-1 OLS prediction standard error). NOT a measurement.
#
#          If the measured series is not monotonically increasing the data is
#          still drawn exactly as measured - never smoothed, no point dropped -
#          and the break is annotated directly on the chart.
#
#          Reads CSV and draws - ZERO cloud calls, zero training.
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training.eval_ksweep import (
    AXIS,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SERIES,
    SURFACE,
    TEST_N,
    VQR_COLOR,
    _style_axes,
)
from utils.path import TRAINING_EVA_RESULT

# ==========================================================================
# PARAMETERS
# ==========================================================================
# Which VQR family to chart. The noisy and exact protocols are DIFFERENT
# experiments (see train_vqr_ksweep.py); never mix them on one axis. The
# protocol is printed in the chart subtitle so a saved PNG is self-identifying.
PROTOCOL = "noisy"          # "noisy" -> vqr_ksweep_results.csv
                            # "exact" -> vqr_ksweep_exact_results.csv
PROTOCOL_SUBTITLE = {
    "noisy": "VQR protocol: sampled estimator (default_precision=0.015625)",
    "exact": "VQR protocol: exact statevector (precision=0.0)",
}
VQR_CSV = os.path.join(
    TRAINING_EVA_RESULT,
    "vqr_ksweep_results.csv" if PROTOCOL == "noisy"
    else f"vqr_ksweep_{PROTOCOL}_results.csv",
)
XGB_CSV = os.path.join(TRAINING_EVA_RESULT, "ksweep_results.csv")

K_PROJECT_TO = 12          # extrapolate this far and no further
XGB_N_TRAIN = 50000        # classical reference line (protocol Q)

PROJ_COLOR = "#8fd9c2"     # pale VQR_COLOR - reserved for the projected region
BAND_ALPHA = 0.18

PROJ_LABEL = "Projection (linear extrapolation — unverified)"


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
def load_measured() -> pd.DataFrame:
    """The MEASURED VQR points, taken verbatim from the results CSV."""
    if not os.path.exists(VQR_CSV):
        raise FileNotFoundError(
            f"No VQR results at {VQR_CSV}. Run training/train_vqr_ksweep.py first."
        )
    df = pd.read_csv(VQR_CSV).sort_values("k").reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Need at least 2 measured k points to fit a projection.")
    return df


def load_xgb_reference() -> pd.DataFrame:
    """Classical reference: XGBoost protocol Q at n_train = 50,000."""
    df = pd.read_csv(XGB_CSV)
    ref = df[(df["n_train"] == XGB_N_TRAIN) & (df["k"] <= K_PROJECT_TO)]
    return ref.sort_values("k").reset_index(drop=True)


def linear_projection(xs: np.ndarray, ys: np.ndarray, x_grid: np.ndarray) -> dict:
    """Degree-1 fit + OLS PREDICTION standard error (widens away from the data).

        se(x0) = s * sqrt(1 + 1/n + (x0 - x_mean)^2 / Sxx)

    One s.e. is used for the shaded band. With few points that band is already
    wide; the point is to say "unknown", not to look precise.
    """
    slope, intercept = np.polyfit(xs, ys, 1)
    fit = slope * x_grid + intercept

    n = len(xs)
    resid = ys - (slope * xs + intercept)
    dof = max(n - 2, 1)
    s = float(np.sqrt(np.sum(resid**2) / dof))
    x_mean = float(np.mean(xs))
    sxx = float(np.sum((xs - x_mean) ** 2))

    se = s * np.sqrt(1.0 + 1.0 / n + (x_grid - x_mean) ** 2 / sxx) if sxx > 0 else \
        np.full_like(x_grid, s, dtype=float)

    # where the projection crosses y = 0 (the mean-prediction baseline)
    k_cross = float(-intercept / slope) if slope != 0 else None
    if k_cross is not None and not (x_grid[0] <= k_cross <= x_grid[-1]):
        k_cross = None

    return {"slope": float(slope), "intercept": float(intercept), "fit": fit,
            "se": se, "resid_std": s, "k_cross": k_cross}


def find_breaks(xs: np.ndarray, ys: np.ndarray) -> list[tuple[float, float]]:
    """Steps where R^2 does NOT rise with k -> breaks in the trend."""
    diffs = np.diff(ys)
    return [(float(xs[i + 1]), float(ys[i + 1]))
            for i, d in enumerate(diffs) if d <= 0]


# ==========================================================================
# PLOTTING
# ==========================================================================
def plot_trend(measured: pd.DataFrame, xgb: pd.DataFrame, metric: str,
               ylabel: str, title: str, filename: str) -> dict:
    xs = measured["k"].to_numpy(dtype=float)
    ys = measured[metric].to_numpy(dtype=float)

    x_grid = np.linspace(xs.min(), K_PROJECT_TO, 200)
    proj = linear_projection(xs, ys, x_grid)
    breaks = find_breaks(xs, ys)

    fig, ax = plt.subplots(figsize=(9, 6.2), facecolor=SURFACE)
    _style_axes(ax, title, ylabel)
    ax.set_xlabel("k (qubits = number of features)", color=INK_SECONDARY,
                  fontsize=10)

    # protocol stamp, so a saved PNG can never be mistaken for the other family.
    # the title pad is widened here to leave room for it under the title.
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, pad=28, loc="left")
    ax.annotate(PROTOCOL_SUBTITLE[PROTOCOL], xy=(0, 1.012), xycoords="axes fraction",
                color=INK_MUTED, fontsize=8.5, va="bottom")

    # --- 4. mean-prediction baseline -------------------------------------
    ax.axhline(0, color=INK_MUTED, linewidth=1.2, linestyle="--", zorder=1)
    ax.annotate("y = 0 · mean-prediction baseline", (K_PROJECT_TO, 0),
                textcoords="offset points", xytext=(-4, 6), ha="right",
                color=INK_MUTED, fontsize=8)

    # --- 3. XGBoost reference (classical, protocol Q) ---------------------
    if not xgb.empty:
        ax.plot(xgb["k"], xgb[metric], color=SERIES[XGB_N_TRAIN], linewidth=1.2,
                marker="o", markersize=4.5,
                label=f"XGBoost protocol Q, n_train = {XGB_N_TRAIN:,} (measured)",
                zorder=3)

    # --- 2. uncertainty band + projection line ---------------------------
    ax.fill_between(x_grid, proj["fit"] - proj["se"], proj["fit"] + proj["se"],
                    color=PROJ_COLOR, alpha=BAND_ALPHA, linewidth=0,
                    label="Projection uncertainty (±1 s.e. of prediction)", zorder=2)
    ax.plot(x_grid, proj["fit"], color=PROJ_COLOR, linewidth=2, linestyle="--",
            label=PROJ_LABEL, zorder=3)

    # --- 1. MEASURED points ----------------------------------------------
    ax.plot(xs, ys, color=VQR_COLOR, linewidth=2.4, marker="s", markersize=8,
            label=f"VQR measured, n_train = {int(measured['n_train'].iloc[0]):,}",
            zorder=5)

    # --- 5. projection crossing y = 0 ------------------------------------
    if proj["k_cross"] is not None:
        kc = proj["k_cross"]
        ax.plot([kc], [0], marker="X", markersize=11, color=PROJ_COLOR,
                markeredgecolor=INK_SECONDARY, markeredgewidth=0.8, zorder=6)
        ax.annotate(f"k ≈ {kc:.1f} (projected)", (kc, 0),
                    textcoords="offset points", xytext=(6, -16),
                    color=INK_SECONDARY, fontsize=9)

    # --- trend breaks, if any --------------------------------------------
    for kb, yb in breaks:
        ax.annotate(f"trend break at k={int(kb)}", (kb, yb),
                    textcoords="offset points", xytext=(0, -22), ha="center",
                    color=INK_SECONDARY, fontsize=8,
                    arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.9))

    ax.set_xticks(list(range(int(xs.min()), K_PROJECT_TO + 1)))
    ax.set_xlim(xs.min() - 0.4, K_PROJECT_TO + 0.4)

    # legend + caption sit BELOW the axes so they never cover the data
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=8.5,
              loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2,
              handlelength=2.4, columnspacing=1.8)

    if breaks:
        note = ("Measured series is NOT monotonically increasing — shown as "
                "measured, not smoothed; the linear extrapolation is "
                "correspondingly less reliable.")
    else:
        note = "Measured series increases monotonically over the measured range."
    note += (f"\nOnly k ≤ {int(xs.max())} is MEASURED; "
             f"k > {int(xs.max())} is extrapolation, not a result.")
    fig.text(0.5, 0.012, note, fontsize=8, color=INK_MUTED, ha="center",
             linespacing=1.5)

    fig.tight_layout(rect=(0, 0.075, 1, 1))

    path = os.path.join(TRAINING_EVA_RESULT, filename)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    return {"path": path, "metric": metric, "xs": xs, "ys": ys, "breaks": breaks,
            "slope": proj["slope"], "intercept": proj["intercept"],
            "resid_std": proj["resid_std"], "k_cross": proj["k_cross"],
            "y_at_12": float(proj["fit"][-1]), "se_at_12": float(proj["se"][-1])}


# ==========================================================================
# REPORTING
# ==========================================================================
def report(res: dict) -> None:
    ks = ", ".join(f"k={int(k)}: {y:+.4f}" for k, y in zip(res["xs"], res["ys"]))
    print(f"  [{res['metric']}] measured   -> {ks}")
    print(f"  [{res['metric']}] linear fit : y = {res['slope']:+.5f}*k "
          f"{res['intercept']:+.5f}   (residual std = {res['resid_std']:.4f})")
    print(f"  [{res['metric']}] monotonic? "
          f"{'NO - breaks at k=' + ','.join(str(int(b[0])) for b in res['breaks']) if res['breaks'] else 'YES'}")
    if res["k_cross"] is not None:
        print(f"  [{res['metric']}] projection crosses y=0 at k ~ "
              f"{res['k_cross']:.1f}  (EXTRAPOLATED, unverified)")
    else:
        print(f"  [{res['metric']}] projection does NOT cross y=0 before "
              f"k={K_PROJECT_TO}")
    print(f"  [{res['metric']}] projected value at k={K_PROJECT_TO}: "
          f"{res['y_at_12']:+.4f} +/- {res['se_at_12']:.4f}")
    print(f"  [{res['metric']}] PNG -> {res['path']}")


# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================
def main() -> None:
    # Windows consoles default to cp1252 and cannot encode the symbols used here
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    measured = load_measured()
    xgb = load_xgb_reference()

    print("=" * 96)
    print(f"VQR TREND CHARTS - {len(measured)} measured points "
          f"(k = {', '.join(str(int(k)) for k in measured['k'])}), "
          f"projected to k={K_PROJECT_TO}")
    print(f"protocol = {PROTOCOL}  |  source = {os.path.basename(VQR_CSV)}")
    print(f"shared test set: {TEST_N:,} rows. Zero QuApp cloud calls.")
    print("=" * 96)

    r1 = plot_trend(
        measured, xgb, "r2_real", "R² (real units)",
        f"VQR — R² (real units) vs qubit count, shared {TEST_N:,}-row test set",
        "vqr_trend_r2_real.png",
    )
    report(r1)
    print("-" * 96)

    r2_ = plot_trend(
        measured, xgb, "r2_log", "R² (log space)",
        f"VQR — R² (log space) vs qubit count, shared {TEST_N:,}-row test set",
        "vqr_trend_r2_log.png",
    )
    report(r2_)
    print("=" * 96)
    print(f"[NOTE] every k > {int(measured['k'].max())} on these charts is "
          "EXTRAPOLATION, not a measurement.")


if __name__ == "__main__":
    main()
