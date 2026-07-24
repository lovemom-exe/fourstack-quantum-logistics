# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: PART A/B - shared 10k test set + XGBoost k-sweep ablation.
#
#          Builds ONE large held-out test set (10,000 rows drawn from the eval
#          feature file, seeded) and scores every configuration on exactly that
#          set, so all numbers are mutually comparable.
#
#          PART B sweeps k in {4,6,8,12,15} x n_train in {1000,50000} using the
#          leakage-free feature pool and the SAME target treatment as the VQR
#          run: winsorize(train, p99) -> log1p -> MinMaxScaler(-1,1) fit on
#          train, inverted at prediction time. The TEST target is never
#          winsorized and never used to fit anything.
#
#          This is a DIAGNOSTIC ABLATION, not model selection. The whole curve
#          is reported; no k is cherry-picked off the test set as "the result".
#
#          Runs entirely locally - ZERO QuApp cloud calls.
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
import json
import os
from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler

from algorithms.xgboost import xgboost
from evaluation.metrics import mae, mape, r2, rmse
from training.ml_train import DEMAND, food_eval_df, food_train_df
from training.train_vqr_local import LEAKAGE_FREE_FEATURES, WINSOR_PERCENTILE
from utils.path import TRAINING_EVA_RESULT

# ==========================================================================
# PARAMETERS
# ==========================================================================
TEST_N = 10000
TEST_SEED = 2024          # distinct, documented seed for the large test draw
RANDOM_STATE = 42
K_GRID = [4, 6, 8, 12, 15]
N_TRAIN_GRID = [1000, 50000]

ALL_FEATURE_COLUMNS = food_train_df.drop(columns=[DEMAND]).columns.tolist()
POOL_IDX = [i for i, c in enumerate(ALL_FEATURE_COLUMNS) if c in LEAKAGE_FREE_FEATURES]
POOL_NAMES = [ALL_FEATURE_COLUMNS[i] for i in POOL_IDX]

# --- chart tokens (validated categorical slots 1-3 + chrome ink) ----------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = {1000: "#2a78d6", 50000: "#eb6834"}  # slot 1 blue, slot 2 orange
VQR_COLOR = "#1baf7a"                          # slot 3 aqua (Part C)


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
def build_test_set() -> tuple[np.ndarray, np.ndarray]:
    """Draw the shared 10k test set from the eval feature file.

    Train/validation are drawn from fresh_retail_train_data.csv while the test
    set comes from fresh_retail_eval_data.csv, so the two cannot overlap by
    construction. Seeded and reproducible.
    """
    assert (
        food_eval_df.drop(columns=[DEMAND]).columns.tolist() == ALL_FEATURE_COLUMNS
    ), "eval/train feature columns are misaligned"

    test_df = food_eval_df.sample(n=TEST_N, random_state=TEST_SEED)
    X_test_pool = test_df.drop(columns=[DEMAND]).to_numpy()[:, POOL_IDX]
    y_test_raw = test_df[DEMAND].to_numpy()
    return X_test_pool, y_test_raw


def mi_ranking(X_pool: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mutual-information ranking of the pool, computed ONCE per train set.

    SelectKBest(k) simply keeps the top-k of this ranking, so scoring once and
    slicing per k is identical to refitting per k, and far cheaper.
    """
    scorer = partial(mutual_info_regression, random_state=RANDOM_STATE)
    scores = scorer(X_pool, y)
    return np.argsort(scores)[::-1]  # descending


def fit_target_transform(y_train_raw: np.ndarray):
    """winsorize(p99) -> log1p -> MinMaxScaler(-1,1); all fit on TRAIN only."""
    clip = float(np.percentile(y_train_raw, WINSOR_PERCENTILE))
    n_clipped = int(np.sum(y_train_raw > clip))
    y_log = np.log1p(np.clip(y_train_raw, None, clip))
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(y_log.reshape(-1, 1))
    return scaler, scaler.transform(y_log.reshape(-1, 1)).ravel(), clip, n_clipped


def diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Shrinkage diagnostics: spread ratio, range, Pearson r."""
    std_true = float(np.std(y_true))
    std_pred = float(np.std(y_pred))
    r = float(np.corrcoef(y_pred, y_true)[0, 1]) if std_pred > 0 else 0.0
    return {
        "std_pred": std_pred,
        "std_true": std_true,
        "std_ratio": std_pred / std_true if std_true > 0 else 0.0,
        "pred_min": float(np.min(y_pred)),
        "pred_max": float(np.max(y_pred)),
        "true_min": float(np.min(y_true)),
        "true_max": float(np.max(y_true)),
        "r": r,
    }


def run_cell(n_train: int, k: int, rank_cache: dict,
             X_test_pool: np.ndarray, y_test_raw: np.ndarray) -> dict:
    """Train XGBoost for one (n_train, k) cell and score on the shared test set."""
    if n_train not in rank_cache:
        train_df = food_train_df.sample(n=n_train, random_state=RANDOM_STATE)
        X_tr_pool = train_df.drop(columns=[DEMAND]).to_numpy()[:, POOL_IDX]
        y_tr_raw = train_df[DEMAND].to_numpy()
        rank_cache[n_train] = (X_tr_pool, y_tr_raw, mi_ranking(X_tr_pool, y_tr_raw))
    X_tr_pool, y_tr_raw, ranking = rank_cache[n_train]

    keep = np.sort(ranking[:k])
    selected = [POOL_NAMES[i] for i in keep]

    x_scaler = MinMaxScaler().fit(X_tr_pool[:, keep])
    X_tr = x_scaler.transform(X_tr_pool[:, keep])
    X_test = x_scaler.transform(X_test_pool[:, keep])

    y_scaler, y_tr_scaled, clip, n_clipped = fit_target_transform(y_tr_raw)

    model = xgboost(X_tr, y_tr_scaled)
    if model is None:
        raise RuntimeError("XGBoost training failed.")

    pred_log = y_scaler.inverse_transform(
        np.asarray(model.predict(X_test)).reshape(-1, 1)
    ).ravel()
    pred_real = np.expm1(pred_log)
    y_test_log = np.log1p(y_test_raw)

    d_real = diagnostics(y_test_raw, pred_real)
    d_log = diagnostics(y_test_log, pred_log)

    return {
        "model": "XGBoost",
        "n_train": n_train,
        "k": k,
        "r2_real": r2(y_test_raw, pred_real),
        "r2_log": r2(y_test_log, pred_log),
        "mae": mae(y_test_raw, pred_real),
        "rmse": rmse(y_test_raw, pred_real),
        "mape": mape(y_test_raw, pred_real),
        "std_ratio_real": d_real["std_ratio"],
        "r_real": d_real["r"],
        "std_ratio_log": d_log["std_ratio"],
        "r_log": d_log["r"],
        "winsor_clip": clip,
        "winsor_rows": n_clipped,
        "selected": ";".join(selected),
    }


# ==========================================================================
# PLOTTING
# ==========================================================================
def _style_axes(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, pad=12, loc="left")
    ax.set_xlabel("k (number of features)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    ax.grid(True, color=GRIDLINE, linewidth=1, alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def plot_curve(rows: list[dict], metric: str, ylabel: str, title: str, filename: str,
               vqr_rows: list[dict] | None = None) -> str:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    _style_axes(ax, title, ylabel)

    for n_train in N_TRAIN_GRID:
        pts = sorted([r for r in rows if r["n_train"] == n_train], key=lambda r: r["k"])
        xs = [p["k"] for p in pts]
        ys = [p[metric] for p in pts]
        ax.plot(xs, ys, color=SERIES[n_train], linewidth=2, marker="o",
                markersize=8, label=f"XGBoost n_train={n_train:,}", zorder=3)
        # direct label in ink (the colored line carries identity)
        ax.annotate(f"n={n_train:,}", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, 0), color=INK_SECONDARY, fontsize=9, va="center")

    if vqr_rows:
        pts = sorted(vqr_rows, key=lambda r: r["k"])
        xs = [p["k"] for p in pts]
        ys = [p[metric] for p in pts]
        ax.plot(xs, ys, color=VQR_COLOR, linewidth=2, marker="s", markersize=8,
                label="VQR n_train=1,000", zorder=3)
        ax.annotate("VQR", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, 0), color=INK_SECONDARY, fontsize=9, va="center")

    ax.axhline(0, color=AXIS, linewidth=1, linestyle="--", zorder=1)
    ax.set_xticks(K_GRID)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9, loc="best")
    fig.tight_layout()

    path = os.path.join(TRAINING_EVA_RESULT, filename)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# ==========================================================================
# REPORTING
# ==========================================================================
def print_grid(rows: list[dict]) -> None:
    print("=" * 104)
    print(f"PART B - XGBoost k-sweep, all scored on the SAME {TEST_N:,}-row test set")
    print("=" * 104)
    print(f"{'n_train':>8}{'k':>4}{'R2 (real)':>12}{'R2 (log)':>11}{'MAE':>10}"
          f"{'RMSE':>10}{'std ratio':>11}{'r':>8}")
    print("-" * 104)
    for r in sorted(rows, key=lambda r: (r["n_train"], r["k"])):
        print(f"{r['n_train']:>8,}{r['k']:>4}{r['r2_real']:>12.4f}{r['r2_log']:>11.4f}"
              f"{r['mae']:>10.4f}{r['rmse']:>10.4f}{r['std_ratio_real']:>11.4f}"
              f"{r['r_real']:>8.4f}")
    print("=" * 104)


# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================
def main() -> None:
    os.makedirs(TRAINING_EVA_RESULT, exist_ok=True)
    print(f"[SETUP] shared test set: {TEST_N:,} rows from eval file, seed={TEST_SEED}")
    print(f"[SETUP] leakage-free pool = {len(POOL_NAMES)} cols: {POOL_NAMES}")
    print("[SETUP] estimator-free (XGBoost only) - ZERO QuApp cloud calls")

    X_test_pool, y_test_raw = build_test_set()
    print(f"[TEST] y_true: mean={y_test_raw.mean():.4f} median={np.median(y_test_raw):.4f} "
          f"p99={np.percentile(y_test_raw, 99):.4f} max={y_test_raw.max():.4f} "
          f"zeros={100.0 * np.mean(y_test_raw == 0):.1f}%")

    rank_cache: dict = {}
    rows = []
    for n_train in N_TRAIN_GRID:
        for k in K_GRID:
            row = run_cell(n_train, k, rank_cache, X_test_pool, y_test_raw)
            rows.append(row)
            print(f"  [cell] n_train={n_train:>6,} k={k:>2} -> "
                  f"R2_real={row['r2_real']:+.4f} R2_log={row['r2_log']:+.4f} "
                  f"std_ratio={row['std_ratio_real']:.3f} r={row['r_real']:.3f}")

    print_grid(rows)

    p1 = plot_curve(rows, "r2_real", "R² (real units)",
                    f"XGBoost R² (real units) vs k — shared {TEST_N:,}-row test set",
                    "ksweep_r2_real.png")
    p2 = plot_curve(rows, "r2_log", "R² (log space)",
                    f"XGBoost R² (log space) vs k — shared {TEST_N:,}-row test set",
                    "ksweep_r2_log.png")

    csv_path = os.path.join(TRAINING_EVA_RESULT, "ksweep_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    meta = {
        "test_n": TEST_N,
        "test_seed": TEST_SEED,
        "random_state": RANDOM_STATE,
        "pool": POOL_NAMES,
        "k_grid": K_GRID,
        "n_train_grid": N_TRAIN_GRID,
        "note": "Diagnostic ablation. Full curve reported; no k selected on test.",
    }
    with open(os.path.join(TRAINING_EVA_RESULT, "ksweep_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[DONE] grid CSV : {csv_path}")
    print(f"[DONE] plot     : {p1}")
    print(f"[DONE] plot     : {p2}")
    print("[DONE] Zero QuApp cloud calls.")


if __name__ == "__main__":
    main()
