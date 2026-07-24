# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: PART 1 - isolate the effect of target winsorization.
#
#          Two target protocols are currently mixed up in the results, which
#          makes their numbers incomparable:
#
#            Protocol P (production):  log1p -> expm1.
#                                      NO winsorize, NO MinMax on the target.
#            Protocol Q (vs quantum):  winsorize(train, p99) -> log1p ->
#                                      MinMaxScaler(-1,1) -> inverted.
#                                      Required because VQR output is bounded
#                                      to [-1, 1].
#
#          Both use the same leakage-free pool and are scored on the SAME
#          10,000-row test set. This script runs 4 cells so the delta is
#          attributable to winsorization alone (feature selection is shared
#          within each pair, so only the target treatment differs).
#
#          Hypothesis under test: winsorizing at p99 (threshold ~6.5) truncates
#          the tail, while the test target reaches 41.8 and real-unit R^2 is
#          dominated by exactly that tail.
#
#          Local only - ZERO QuApp cloud calls.
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
import json
import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from algorithms.xgboost import xgboost
from evaluation.metrics import mae, r2, rmse
from training.eval_ksweep import (
    POOL_IDX,
    POOL_NAMES,
    RANDOM_STATE,
    TEST_N,
    build_test_set,
    diagnostics,
    mi_ranking,
)
from training.ml_train import DEMAND, food_train_df
from training.train_vqr_local import WINSOR_PERCENTILE
from utils.path import TRAINING_EVA_RESULT

# ==========================================================================
# PARAMETERS
# ==========================================================================
CELLS = [
    {"n_train": 50000, "k": 15, "winsorize": True},
    {"n_train": 50000, "k": 15, "winsorize": False},
    {"n_train": 1000, "k": 4, "winsorize": True},
    {"n_train": 1000, "k": 4, "winsorize": False},
]


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
def run_cell(n_train: int, k: int, winsorize: bool, cache: dict,
             X_test_pool: np.ndarray, y_test_raw: np.ndarray) -> dict:
    """One (n_train, k, winsorize) cell, scored on the shared test set."""
    if n_train not in cache:
        train_df = food_train_df.sample(n=n_train, random_state=RANDOM_STATE)
        X_pool = train_df.drop(columns=[DEMAND]).to_numpy()[:, POOL_IDX]
        y_raw = train_df[DEMAND].to_numpy()
        cache[n_train] = (X_pool, y_raw, mi_ranking(X_pool, y_raw))
    X_pool, y_raw, ranking = cache[n_train]

    keep = np.sort(ranking[:k])
    selected = [POOL_NAMES[i] for i in keep]

    x_scaler = MinMaxScaler().fit(X_pool[:, keep])
    X_tr = x_scaler.transform(X_pool[:, keep])
    X_test = x_scaler.transform(X_test_pool[:, keep])

    if winsorize:
        # --- Protocol Q ---
        clip = float(np.percentile(y_raw, WINSOR_PERCENTILE))
        n_clipped = int(np.sum(y_raw > clip))
        y_log = np.log1p(np.clip(y_raw, None, clip))
        y_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(y_log.reshape(-1, 1))
        y_fit = y_scaler.transform(y_log.reshape(-1, 1)).ravel()
    else:
        # --- Protocol P ---
        clip, n_clipped, y_scaler = float("nan"), 0, None
        y_fit = np.log1p(y_raw)

    model = xgboost(X_tr, y_fit)
    if model is None:
        raise RuntimeError("XGBoost training failed.")

    raw_pred = np.asarray(model.predict(X_test)).ravel()
    if winsorize:
        pred_log = y_scaler.inverse_transform(raw_pred.reshape(-1, 1)).ravel()
    else:
        pred_log = raw_pred
    pred_real = np.expm1(pred_log)
    y_test_log = np.log1p(y_test_raw)

    d_real = diagnostics(y_test_raw, pred_real)

    return {
        "n_train": n_train,
        "k": k,
        "winsorize": winsorize,
        "protocol": "Q" if winsorize else "P",
        "r2_real": r2(y_test_raw, pred_real),
        "r2_log": r2(y_test_log, pred_log),
        "mae": mae(y_test_raw, pred_real),
        "rmse": rmse(y_test_raw, pred_real),
        "std_ratio": d_real["std_ratio"],
        "r": d_real["r"],
        "pred_max": d_real["pred_max"],
        "clip_threshold": clip,
        "rows_clipped": n_clipped,
        "selected": ";".join(selected),
    }


# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================
def main() -> None:
    os.makedirs(TRAINING_EVA_RESULT, exist_ok=True)
    X_test_pool, y_test_raw = build_test_set()
    print(f"[SETUP] shared test set: {TEST_N:,} rows | "
          f"y_true max={y_test_raw.max():.2f} p99={np.percentile(y_test_raw, 99):.2f}")
    print("[SETUP] XGBoost only - ZERO QuApp cloud calls")

    cache: dict = {}
    rows = [run_cell(c["n_train"], c["k"], c["winsorize"], cache,
                     X_test_pool, y_test_raw) for c in CELLS]

    print("=" * 114)
    print(f"PART 1 - WINSORIZE ABLATION (XGBoost, leakage-free pool, "
          f"shared {TEST_N:,}-row test set)")
    print("=" * 114)
    print(f"{'proto':>6}{'n_train':>9}{'k':>4}{'winsor':>8}{'R2(real)':>11}"
          f"{'R2(log)':>10}{'MAE':>9}{'RMSE':>9}{'std ratio':>11}{'r':>8}{'max(pred)':>11}")
    print("-" * 114)
    for r in rows:
        print(f"{r['protocol']:>6}{r['n_train']:>9,}{r['k']:>4}"
              f"{('CO' if r['winsorize'] else 'KHONG'):>8}"
              f"{r['r2_real']:>11.4f}{r['r2_log']:>10.4f}{r['mae']:>9.4f}"
              f"{r['rmse']:>9.4f}{r['std_ratio']:>11.4f}{r['r']:>8.4f}"
              f"{r['pred_max']:>11.2f}")
    print("=" * 114)

    print("\nDELTA R^2 do winsorize (KHONG winsorize - CO winsorize):")
    for n_train, k in [(50000, 15), (1000, 4)]:
        with_w = next(r for r in rows if r["n_train"] == n_train and r["winsorize"])
        no_w = next(r for r in rows if r["n_train"] == n_train and not r["winsorize"])
        print(f"  n_train={n_train:>6,} k={k:>2}: "
              f"R2_real {with_w['r2_real']:+.4f} -> {no_w['r2_real']:+.4f} "
              f"(delta {no_w['r2_real'] - with_w['r2_real']:+.4f}) | "
              f"R2_log {with_w['r2_log']:+.4f} -> {no_w['r2_log']:+.4f} "
              f"(delta {no_w['r2_log'] - with_w['r2_log']:+.4f})")
        print(f"      max(y_pred): {with_w['pred_max']:.2f} -> {no_w['pred_max']:.2f} "
              f"(y_true max = {y_test_raw.max():.2f}, "
              f"clip threshold = {with_w['clip_threshold']:.4f})")

    csv_path = os.path.join(TRAINING_EVA_RESULT, "ablation_winsorize.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(os.path.join(TRAINING_EVA_RESULT, "ablation_winsorize.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\n[DONE] saved -> {csv_path}")
    print("[DONE] Zero QuApp cloud calls.")


if __name__ == "__main__":
    main()
