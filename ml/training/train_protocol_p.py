# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: Train and PERSIST the Protocol-P production baseline.
#
#          Protocol P: log1p -> expm1. NO winsorize, NO MinMax on the target.
#          Leakage-free feature pool, n_train = 50,000, k = 15.
#          Scored on the shared 10,000-row test set.
#
#          This is the classical baseline number for the deck. The model and
#          every fitted transform are saved to disk so downstream artifacts
#          (make_demo_plots.py) can reuse them WITHOUT retraining.
#
#          Reproduces cell 2 of ablation_winsorize.py exactly (same seeds,
#          same MI ranking, same sample).
#
#          Local only - ZERO QuApp cloud calls.
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
import json
import os

import joblib
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from algorithms.xgboost import xgboost
from evaluation.metrics import mae, mape, r2, rmse
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
from utils.path import TRAINING_EVA_RESULT

# ==========================================================================
# PARAMETERS
# ==========================================================================
N_TRAIN = 50000
K_FEATURES = 15
BUNDLE_PATH = os.path.join(TRAINING_EVA_RESULT, "protocol_p_xgboost.joblib")
METRICS_PATH = os.path.join(TRAINING_EVA_RESULT, "protocol_p_metrics.json")


# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================
def main() -> None:
    os.makedirs(TRAINING_EVA_RESULT, exist_ok=True)
    print(f"[SETUP] Protocol P baseline: n_train={N_TRAIN:,}, k={K_FEATURES}, "
          "log1p -> expm1, NO winsorize")

    X_test_pool, y_test_raw = build_test_set()

    train_df = food_train_df.sample(n=N_TRAIN, random_state=RANDOM_STATE)
    X_pool = train_df.drop(columns=[DEMAND]).to_numpy()[:, POOL_IDX]
    y_raw = train_df[DEMAND].to_numpy()

    ranking = mi_ranking(X_pool, y_raw)
    keep = np.sort(ranking[:K_FEATURES])
    selected = [POOL_NAMES[i] for i in keep]

    x_scaler = MinMaxScaler().fit(X_pool[:, keep])
    X_tr = x_scaler.transform(X_pool[:, keep])
    X_test = x_scaler.transform(X_test_pool[:, keep])

    # Protocol P target treatment: log1p only.
    y_fit = np.log1p(y_raw)

    model = xgboost(X_tr, y_fit)
    if model is None:
        raise RuntimeError("XGBoost training failed.")

    pred_log = np.asarray(model.predict(X_test)).ravel()
    pred_real = np.expm1(pred_log)
    y_test_log = np.log1p(y_test_raw)

    d = diagnostics(y_test_raw, pred_real)
    metrics = {
        "protocol": "P",
        "n_train": N_TRAIN,
        "k": K_FEATURES,
        "r2_real": r2(y_test_raw, pred_real),
        "r2_log": r2(y_test_log, pred_log),
        "mae": mae(y_test_raw, pred_real),
        "rmse": rmse(y_test_raw, pred_real),
        "mape": mape(y_test_raw, pred_real),
        "std_ratio": d["std_ratio"],
        "r": d["r"],
        "pred_max": d["pred_max"],
        "test_n": TEST_N,
        "selected": selected,
    }

    print("=" * 70)
    print("PROTOCOL P - classical production baseline")
    print("=" * 70)
    for key in ("r2_real", "r2_log", "mae", "rmse", "mape", "std_ratio", "r"):
        print(f"  {key:<12} {metrics[key]:.4f}")
    print(f"  {'pred_max':<12} {metrics['pred_max']:.2f}  (y_true max = {y_test_raw.max():.2f})")
    print("=" * 70)

    joblib.dump(
        {
            "protocol": "P",
            "model": model,
            "x_scaler": x_scaler,
            "keep_idx_in_pool": keep,
            "pool_idx": POOL_IDX,
            "pool_names": POOL_NAMES,
            "selected": selected,
            "n_train": N_TRAIN,
            "k": K_FEATURES,
            "target_treatment": "log1p -> expm1 (no winsorize, no minmax)",
        },
        BUNDLE_PATH,
    )
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"[SAVE] model bundle -> {BUNDLE_PATH}")
    print(f"[SAVE] metrics      -> {METRICS_PATH}")
    print("[DONE] Zero QuApp cloud calls.")


if __name__ == "__main__":
    main()
