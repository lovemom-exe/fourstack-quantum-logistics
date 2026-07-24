# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: LOCAL, credit-free VQR-vs-XGBoost feasibility comparison.
#
#          Runs entirely on qiskit's local StatevectorEstimator (NOT the cloud
#          QuappEstimator), so it costs ZERO QuApp credit. Remaining credit is
#          reserved for quapp_proof_run.py.
#
#          FIX APPLIED (model output range bug):
#          A VQR's output is a Pauli expectation value bounded to [-1, 1].
#          Training directly on log1p(sale_amount) put the target well outside
#          that range, so the model could not express it and collapsed to a
#          near-constant prediction (R^2 = -0.0378). The target is now mapped
#          into the model's representable range:
#              winsorize(train) -> log1p -> MinMaxScaler(-1, 1)  [fit on train]
#          and inverted at prediction time:
#              inverse_transform -> expm1 -> metrics in real units.
#
#          EVALUATION PROTOCOL (honest): train / validation / test. Restarts
#          are selected on VALIDATION only. The test set is touched exactly
#          once, at the very end, after the configuration is frozen.
#
#          This is a matched feasibility comparison on a small subset. It is
#          NOT the 50k-row production XGBoost number (R^2 = 0.3938).
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
import os
import time

import joblib
import numpy as np
import pandas as pd

from qiskit.circuit.library import real_amplitudes
from qiskit.primitives import StatevectorEstimator
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from algorithms.vqr import vqr
from algorithms.xgboost import xgboost
from evaluation.metrics import mae, mape, r2, rmse
from training.ml_train import DEMAND, create_food_sample, food_train_df
from utils.path import TRAINING_EVA_RESULT

# ==========================================================================
# PARAMETERS  (all seeds are explicit and printed for reproducibility)
# ==========================================================================
N_SAMPLE = 2000             # full run; auto-reduced to N_SAMPLE_REDUCED if slow
N_SAMPLE_REDUCED = 1000
K_FEATURES = 4              # 4 qubits
FEATURE_MAP_REPS = 2
ANSATZ_REPS = 3
MAXITER = 300               # COBYLA budget (was 60 - objective still descending)
RESTART_SEEDS = [0, 1, 2]   # 3 random restarts
VAL_FRACTION = 0.2
WINSOR_PERCENTILE = 99      # applied to TRAIN target only
RANDOM_STATE = 42

# Timing gate
SMOKE_N = 200
SMOKE_MAXITER = 20
TIME_BUDGET_SECONDS = 30 * 60

# Leakage-free feature pool ONLY: calendar cyclical + weekend + promo/holiday
# + weather. All same-day stock/stockout columns and their interactions are
# excluded, since those describe the same day being predicted.
LEAKAGE_FREE_FEATURES = [
    "day_sin",
    "day_cos",
    "day_sin_month",
    "day_cos_month",
    "month_sin",
    "month_cos",
    "is_weekend",
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
    "bad_weather",
]

ALL_FEATURE_COLUMNS = food_train_df.drop(columns=[DEMAND]).columns.tolist()
N_ANSATZ_PARAMS = real_amplitudes(
    num_qubits=K_FEATURES, reps=ANSATZ_REPS
).num_parameters


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
def _prepare(n_sample: int, verbose: bool = True) -> dict:
    """Build the train/validation/test subset with the full target treatment.

    Everything (feature selection, feature scaling, winsorization, target
    scaling) is fit on TRAIN ONLY and applied to validation/test.
    """
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = create_food_sample(
        n_sample=n_sample, random_state=RANDOM_STATE
    )

    # --- restrict to the leakage-free feature pool -------------------------
    keep_idx = [
        i for i, c in enumerate(ALL_FEATURE_COLUMNS) if c in LEAKAGE_FREE_FEATURES
    ]
    keep_names = [ALL_FEATURE_COLUMNS[i] for i in keep_idx]
    X_train_raw = X_train_raw[:, keep_idx]
    X_test_raw = X_test_raw[:, keep_idx]

    # --- train / validation split (test comes from the eval file) ----------
    X_tr, X_val, y_tr_raw, y_val_raw = train_test_split(
        X_train_raw,
        y_train_raw,
        test_size=VAL_FRACTION,
        random_state=RANDOM_STATE,
    )

    # --- feature selection: fit on train only ------------------------------
    selector = SelectKBest(score_func=mutual_info_regression, k=K_FEATURES)
    X_tr_sel = selector.fit_transform(X_tr, y_tr_raw)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test_raw)
    selected = [keep_names[i] for i in selector.get_support(indices=True)]

    # --- feature scaling to [0, 1] (angle encoding): fit on train only -----
    x_scaler = MinMaxScaler()
    X_tr_s = x_scaler.fit_transform(X_tr_sel)
    X_val_s = x_scaler.transform(X_val_sel)
    X_test_s = x_scaler.transform(X_test_sel)

    # --- TARGET: winsorize TRAIN only, then log1p, then scale to [-1, 1] ---
    clip_threshold = float(np.percentile(y_tr_raw, WINSOR_PERCENTILE))
    n_clipped = int(np.sum(y_tr_raw > clip_threshold))
    y_tr_w = np.clip(y_tr_raw, None, clip_threshold)

    y_tr_log = np.log1p(y_tr_w)
    y_val_log = np.log1p(y_val_raw)   # validation target NOT winsorized
    y_test_log = np.log1p(y_test_raw)  # test target NEVER touched

    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_tr_scaled = y_scaler.fit_transform(y_tr_log.reshape(-1, 1)).ravel()
    y_val_scaled = y_scaler.transform(y_val_log.reshape(-1, 1)).ravel()

    if verbose:
        print(f"[DATA] train={len(y_tr_raw)} val={len(y_val_raw)} test={len(y_test_raw)}")
        print(f"[DATA] leakage-free pool = {len(keep_names)} cols -> selected k={K_FEATURES}: {selected}")
        print(
            f"[WINSOR] train target clipped at p{WINSOR_PERCENTILE} = "
            f"{clip_threshold:.4f}; rows affected = {n_clipped} "
            f"({100.0 * n_clipped / len(y_tr_raw):.2f}% of train). "
            "Validation and test targets untouched."
        )
        print(
            f"[TARGET] log1p -> MinMaxScaler(-1, 1) fit on train; "
            f"train log range = [{y_tr_log.min():.4f}, {y_tr_log.max():.4f}]"
        )

    return {
        "X_tr": X_tr_s,
        "X_val": X_val_s,
        "X_test": X_test_s,
        "y_tr_scaled": y_tr_scaled,
        "y_val_scaled": y_val_scaled,
        "y_test_raw": y_test_raw,
        "y_test_log": y_test_log,
        "y_scaler": y_scaler,
        "x_scaler": x_scaler,
        "selector": selector,
        "pool_idx": keep_idx,
        "pool_names": keep_names,
        "clip_threshold": clip_threshold,
        "n_clipped": n_clipped,
        "selected": selected,
    }


def _fit_vqr(X_tr, y_tr_scaled, seed: int, maxiter: int, verbose: bool = True):
    """Fit one VQR restart on the LOCAL statevector estimator.

    Note on batching: qiskit-machine-learning's EstimatorQNN builds ONE pub per
    observable containing the full (n_samples, n_params) array and issues a
    single estimator.run(...) per objective evaluation - there is no per-sample
    loop in this path (that per-sample loop is what made the cloud estimator
    infeasible).
    """
    rng = np.random.default_rng(seed)
    initial_point = rng.uniform(-np.pi, np.pi, size=N_ANSATZ_PARAMS)

    state = {"n": 0, "last": float("nan")}

    def _cb(_weights: np.ndarray, obj: float) -> None:
        state["n"] += 1
        state["last"] = float(obj)
        if verbose and (state["n"] % 50 == 0 or state["n"] == 1):
            print(f"        [seed {seed}] iter {state['n']:>4}  objective={obj:.6f}")

    model = vqr(
        X_tr,
        y_tr_scaled,
        k=K_FEATURES,
        estimator=StatevectorEstimator(),
        feature_map_reps=FEATURE_MAP_REPS,
        ansatz_reps=ANSATZ_REPS,
        maxiter=maxiter,
        initial_point=initial_point,
        callback=_cb,
    )
    if model is None:
        raise RuntimeError("VQR training returned None.")
    return model, state["last"], state["n"]


def _to_real_units(pred_scaled: np.ndarray, y_scaler: MinMaxScaler):
    """scaled -> log space -> real units."""
    pred_log = y_scaler.inverse_transform(
        np.asarray(pred_scaled).reshape(-1, 1)
    ).ravel()
    return pred_log, np.expm1(pred_log)


def _evaluate(name: str, n: int, y_true_real, y_true_log, pred_log, pred_real) -> dict:
    return {
        "model": name,
        "n": n,
        "k": K_FEATURES,
        "r2_real": r2(y_true_real, pred_real),
        "r2_log": r2(y_true_log, pred_log),
        "mae": mae(y_true_real, pred_real),
        "rmse": rmse(y_true_real, pred_real),
        "mape": mape(y_true_real, pred_real),
    }


def _print_table(rows: list[dict]) -> None:
    print("=" * 92)
    print("MATCHED FEASIBILITY COMPARISON - local StatevectorEstimator, zero QuApp credit")
    print("(separate from the 50k-row production XGBoost baseline, R^2 = 0.3938)")
    print("=" * 92)
    print(
        f"{'model':<10}{'n':>6}{'k':>4}{'R2 (real)':>12}{'R2 (log)':>11}"
        f"{'MAE':>11}{'RMSE':>11}{'MAPE%':>12}"
    )
    print("-" * 92)
    for r in rows:
        print(
            f"{r['model']:<10}{r['n']:>6}{r['k']:>4}{r['r2_real']:>12.4f}"
            f"{r['r2_log']:>11.4f}{r['mae']:>11.4f}{r['rmse']:>11.4f}{r['mape']:>12.2f}"
        )
    print("=" * 92)


def _persist(model, data: dict, seed: int, k: int) -> tuple[str, str]:
    """Persist the selected VQR so it can be re-scored later WITHOUT retraining.

    Saves the trained ansatz weights plus every transform that was fit on train
    (SelectKBest, feature MinMaxScaler, target MinMaxScaler) - the exact set
    needed to apply the frozen model to a new test set.
    """
    os.makedirs(TRAINING_EVA_RESULT, exist_ok=True)
    weights_path = os.path.join(TRAINING_EVA_RESULT, f"vqr_weights_k{k}.npy")
    bundle_path = os.path.join(TRAINING_EVA_RESULT, f"vqr_artifacts_k{k}.joblib")

    np.save(weights_path, np.asarray(model.weights))
    joblib.dump(
        {
            "k": k,
            "seed": seed,
            "weights": np.asarray(model.weights),
            "selector": data["selector"],
            "x_scaler": data["x_scaler"],
            "y_scaler": data["y_scaler"],
            "pool_idx": data["pool_idx"],
            "pool_names": data["pool_names"],
            "selected": data["selected"],
            "clip_threshold": data["clip_threshold"],
            "feature_map_reps": FEATURE_MAP_REPS,
            "ansatz_reps": ANSATZ_REPS,
            "leakage_free_features": LEAKAGE_FREE_FEATURES,
        },
        bundle_path,
    )
    return weights_path, bundle_path


# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================
def main() -> None:
    t_start = time.perf_counter()
    print("=" * 92)
    print("[SEEDS] data/split/selection random_state=%d | restart seeds=%s | numpy default_rng per restart"
          % (RANDOM_STATE, RESTART_SEEDS))
    print(f"[SETUP] k={K_FEATURES} qubits, zz_feature_map(reps={FEATURE_MAP_REPS}), "
          f"real_amplitudes(reps={ANSATZ_REPS}) -> {N_ANSATZ_PARAMS} weights, COBYLA(maxiter={MAXITER})")
    print("[SETUP] estimator = StatevectorEstimator (LOCAL) - NO QuApp cloud calls")
    print("=" * 92)

    # ----------------------------------------------------------
    # Timing gate: short smoke run, then extrapolate.
    print(f"[SMOKE] timing run: n={SMOKE_N}, maxiter={SMOKE_MAXITER}, 1 restart ...")
    smoke = _prepare(SMOKE_N, verbose=False)
    t0 = time.perf_counter()
    _fit_vqr(smoke["X_tr"], smoke["y_tr_scaled"], seed=RESTART_SEEDS[0],
             maxiter=SMOKE_MAXITER, verbose=False)
    smoke_elapsed = time.perf_counter() - t0
    smoke_units = SMOKE_MAXITER * len(smoke["y_tr_scaled"])
    per_unit = smoke_elapsed / smoke_units
    print(f"[SMOKE] elapsed = {smoke_elapsed:.2f}s for {smoke_units} sample-evaluations "
          f"({per_unit * 1000:.4f} ms each)")

    def _estimate(n: int) -> float:
        n_train = int(n * (1 - VAL_FRACTION))
        return per_unit * MAXITER * n_train * len(RESTART_SEEDS)

    est_full = _estimate(N_SAMPLE)
    print(f"[SMOKE] extrapolated full run (n={N_SAMPLE}, {len(RESTART_SEEDS)} restarts) "
          f"= {est_full / 60:.1f} min")

    n_final = N_SAMPLE
    if est_full > TIME_BUDGET_SECONDS:
        n_final = N_SAMPLE_REDUCED
        print(f"[SMOKE] exceeds {TIME_BUDGET_SECONDS / 60:.0f} min budget -> REDUCING n to "
              f"{n_final} (estimated {_estimate(n_final) / 60:.1f} min)")
    else:
        print(f"[SMOKE] within {TIME_BUDGET_SECONDS / 60:.0f} min budget -> keeping n={n_final}")

    # ----------------------------------------------------------
    # Full run.
    print("-" * 92)
    data = _prepare(n_final, verbose=True)
    print("-" * 92)

    print(f"[VQR] {len(RESTART_SEEDS)} random restarts, COBYLA(maxiter={MAXITER}) ...")
    restarts = []
    for seed in RESTART_SEEDS:
        t_r = time.perf_counter()
        model, final_obj, n_evals = _fit_vqr(
            data["X_tr"], data["y_tr_scaled"], seed=seed, maxiter=MAXITER
        )
        # Restart selection uses VALIDATION only - test is not touched here.
        val_pred = np.asarray(model.predict(data["X_val"])).ravel()
        val_mse = float(np.mean((val_pred - data["y_val_scaled"]) ** 2))
        restarts.append(
            {"seed": seed, "model": model, "final_obj": final_obj,
             "n_evals": n_evals, "val_mse": val_mse}
        )
        print(f"     seed={seed}: final train objective={final_obj:.6f} "
              f"({n_evals} evals) | validation MSE={val_mse:.6f} "
              f"| {time.perf_counter() - t_r:.1f}s")

    best = min(restarts, key=lambda r: r["val_mse"])
    print(f"[VQR] SELECTED seed={best['seed']} on validation MSE={best['val_mse']:.6f} "
          "(selection used validation only)")

    w_path, b_path = _persist(best["model"], data, best["seed"], K_FEATURES)
    print(f"[SAVE] weights   -> {w_path}")
    print(f"[SAVE] artifacts -> {b_path}")

    # ----------------------------------------------------------
    # Configuration is now FROZEN. Touch the test set exactly once.
    print("[TEST] configuration frozen - evaluating on test set ONCE.")
    vqr_pred_log, vqr_pred_real = _to_real_units(
        np.asarray(best["model"].predict(data["X_test"])).ravel(), data["y_scaler"]
    )
    vqr_metrics = _evaluate(
        "VQR", n_final, data["y_test_raw"], data["y_test_log"], vqr_pred_log, vqr_pred_real
    )

    # Matched XGBoost: same subset, same splits, same target treatment.
    xgb_model = xgboost(data["X_tr"], data["y_tr_scaled"])
    if xgb_model is None:
        raise RuntimeError("XGBoost training returned None (invalid input types).")
    xgb_pred_log, xgb_pred_real = _to_real_units(
        np.asarray(xgb_model.predict(data["X_test"])).ravel(), data["y_scaler"]
    )
    xgb_metrics = _evaluate(
        "XGBoost", n_final, data["y_test_raw"], data["y_test_log"], xgb_pred_log, xgb_pred_real
    )

    # ----------------------------------------------------------
    _print_table([vqr_metrics, xgb_metrics])

    os.makedirs(TRAINING_EVA_RESULT, exist_ok=True)
    csv_path = os.path.join(TRAINING_EVA_RESULT, "vqr_local_predictions.csv")
    pd.DataFrame(
        {
            "y_true": data["y_test_raw"],
            "y_true_log": data["y_test_log"],
            "vqr_pred": vqr_pred_real,
            "vqr_pred_log": vqr_pred_log,
            "xgboost_pred": xgb_pred_real,
            "xgboost_pred_log": xgb_pred_log,
        }
    ).to_csv(csv_path, index=False)

    elapsed = time.perf_counter() - t_start
    print(f"[DONE] Predictions saved to {csv_path}")
    print(f"[DONE] Restart objectives: "
          f"{ {r['seed']: round(r['final_obj'], 6) for r in restarts} }")
    print(f"[DONE] Winsor threshold={data['clip_threshold']:.4f}, "
          f"rows affected={data['n_clipped']}")
    print(f"[DONE] Total elapsed = {elapsed / 60:.2f} min")
    print("[DONE] Test set evaluated exactly ONCE. Zero QuApp cloud calls.")


if __name__ == "__main__":
    main()
