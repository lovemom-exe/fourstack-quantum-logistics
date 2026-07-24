# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: PART C - VQR at k = 4, 6, 8 qubits, scored on the SHARED 10k test set.
#
#          Everything except k is frozen to the Part-A configuration:
#          zz_feature_map(k, reps=2), real_amplitudes(k, reps=3),
#          COBYLA(maxiter=300), 3 seeded restarts selected on VALIDATION,
#          n_sample=1000 (-> 800 train / 200 val), and the same target
#          treatment: winsorize(train, p99) -> log1p -> MinMaxScaler(-1,1)
#          fit on train, inverted at prediction time.
#
#          k=4 is retrained here because the original k=4 model was never
#          persisted and was lost when that process exited. Every model is now
#          saved immediately after selection so this cannot recur.
#
#          Feature selection uses a SEEDED mutual_info_regression so the whole
#          Part B / Part C family is reproducible and mutually comparable. The
#          original (lost) k=4 run used an unseeded scorer, so its selected
#          feature set may differ slightly from the k=4 retrained here.
#
#          Runs entirely on the local StatevectorEstimator - ZERO cloud calls.
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
import json
import os
import time
from functools import partial

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qiskit.circuit.library import real_amplitudes
from qiskit.primitives import StatevectorEstimator
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from algorithms.vqr import vqr
from evaluation.metrics import mae, mape, r2, rmse
from training.eval_ksweep import (
    AXIS,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    POOL_IDX,
    POOL_NAMES,
    SERIES,
    SURFACE,
    TEST_N,
    VQR_COLOR,
    _style_axes,
    build_test_set,
    diagnostics,
    fit_target_transform,
)
from training.ml_train import create_food_sample
from training.train_vqr_local import (
    ANSATZ_REPS,
    FEATURE_MAP_REPS,
    MAXITER,
    RANDOM_STATE,
    RESTART_SEEDS,
    VAL_FRACTION,
)
from utils.path import TRAINING_EVA_RESULT

# ==========================================================================
# PARAMETERS
# ==========================================================================
K_GRID_VQR = [4, 6, 8]
N_SAMPLE = 1000              # -> 800 train / 200 validation
PROBE_ROWS = 200             # rows used only for the per-k runtime estimate
RESULTS_JSON = os.path.join(TRAINING_EVA_RESULT, "vqr_ksweep_results.json")
RESULTS_CSV = os.path.join(TRAINING_EVA_RESULT, "vqr_ksweep_results.csv")


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
def prepare(k: int, X_test_pool: np.ndarray) -> dict:
    """Build the frozen train/val split for a given k and map the shared test set."""
    X_train_raw, y_train_raw, _, _ = create_food_sample(
        n_sample=N_SAMPLE, random_state=RANDOM_STATE
    )
    X_train_pool = X_train_raw[:, POOL_IDX]

    X_tr_pool, X_val_pool, y_tr_raw, y_val_raw = train_test_split(
        X_train_pool, y_train_raw, test_size=VAL_FRACTION, random_state=RANDOM_STATE
    )

    scorer = partial(mutual_info_regression, random_state=RANDOM_STATE)
    selector = SelectKBest(score_func=scorer, k=k).fit(X_tr_pool, y_tr_raw)
    selected = [POOL_NAMES[i] for i in selector.get_support(indices=True)]

    x_scaler = MinMaxScaler().fit(selector.transform(X_tr_pool))
    X_tr = x_scaler.transform(selector.transform(X_tr_pool))
    X_val = x_scaler.transform(selector.transform(X_val_pool))
    X_test = x_scaler.transform(selector.transform(X_test_pool))

    y_scaler, y_tr_scaled, clip, n_clipped = fit_target_transform(y_tr_raw)
    y_val_scaled = y_scaler.transform(
        np.log1p(y_val_raw).reshape(-1, 1)
    ).ravel()

    return {
        "k": k,
        "X_tr": X_tr, "X_val": X_val, "X_test": X_test,
        "y_tr_scaled": y_tr_scaled, "y_val_scaled": y_val_scaled,
        "selector": selector, "x_scaler": x_scaler, "y_scaler": y_scaler,
        "clip": clip, "n_clipped": n_clipped, "selected": selected,
    }


def fit_once(X_tr, y_tr_scaled, k: int, seed: int, maxiter: int, verbose: bool = True):
    """One seeded VQR restart on the LOCAL statevector estimator."""
    n_params = real_amplitudes(num_qubits=k, reps=ANSATZ_REPS).num_parameters
    rng = np.random.default_rng(seed)
    initial_point = rng.uniform(-np.pi, np.pi, size=n_params)

    state = {"n": 0, "last": float("nan")}

    def _cb(_w: np.ndarray, obj: float) -> None:
        state["n"] += 1
        state["last"] = float(obj)
        if verbose and (state["n"] % 100 == 0 or state["n"] == 1):
            print(f"        [k={k} seed={seed}] iter {state['n']:>4} obj={obj:.6f}")

    model = vqr(
        X_tr, y_tr_scaled, k=k,
        estimator=StatevectorEstimator(),
        feature_map_reps=FEATURE_MAP_REPS,
        ansatz_reps=ANSATZ_REPS,
        maxiter=maxiter,
        initial_point=initial_point,
        callback=_cb,
    )
    if model is None:
        raise RuntimeError(f"VQR training returned None for k={k}.")
    return model, state["last"], state["n"]


def estimate_runtime(data: dict, k: int) -> float:
    """Short probe -> honest ETA for the full 3-restart run at this k."""
    n_params = real_amplitudes(num_qubits=k, reps=ANSATZ_REPS).num_parameters
    probe_maxiter = n_params + 22  # comfortably above COBYLA's num_vars+2 minimum
    Xp = data["X_tr"][:PROBE_ROWS]
    yp = data["y_tr_scaled"][:PROBE_ROWS]

    t0 = time.perf_counter()
    _, _, n_evals = fit_once(Xp, yp, k, RESTART_SEEDS[0], probe_maxiter, verbose=False)
    elapsed = time.perf_counter() - t0

    per_sample_eval = elapsed / max(n_evals * len(yp), 1)
    return per_sample_eval * MAXITER * len(data["y_tr_scaled"]) * len(RESTART_SEEDS)


def persist(model, data: dict, seed: int, k: int) -> str:
    """Save weights + every fitted transform immediately after selection."""
    os.makedirs(TRAINING_EVA_RESULT, exist_ok=True)
    np.save(os.path.join(TRAINING_EVA_RESULT, f"vqr_weights_k{k}.npy"),
            np.asarray(model.weights))
    bundle = os.path.join(TRAINING_EVA_RESULT, f"vqr_artifacts_k{k}.joblib")
    joblib.dump(
        {
            "k": k, "seed": seed, "weights": np.asarray(model.weights),
            "selector": data["selector"], "x_scaler": data["x_scaler"],
            "y_scaler": data["y_scaler"], "selected": data["selected"],
            "clip_threshold": data["clip"], "pool_names": POOL_NAMES,
            "feature_map_reps": FEATURE_MAP_REPS, "ansatz_reps": ANSATZ_REPS,
        },
        bundle,
    )
    return bundle


def run_k(k: int, X_test_pool: np.ndarray, y_test_raw: np.ndarray) -> dict:
    print("=" * 100)
    print(f"[k={k}] VQR: {k} qubits, real_amplitudes(reps={ANSATZ_REPS}) -> "
          f"{real_amplitudes(num_qubits=k, reps=ANSATZ_REPS).num_parameters} weights")

    data = prepare(k, X_test_pool)
    print(f"[k={k}] selected features: {data['selected']}")
    print(f"[k={k}] winsor p99={data['clip']:.4f}, rows affected={data['n_clipped']}")

    eta = estimate_runtime(data, k)
    print(f"[k={k}] ETA for {len(RESTART_SEEDS)} restarts x maxiter={MAXITER}: "
          f"{eta / 60:.1f} min")

    restarts = []
    for seed in RESTART_SEEDS:
        t0 = time.perf_counter()
        model, final_obj, n_evals = fit_once(
            data["X_tr"], data["y_tr_scaled"], k, seed, MAXITER
        )
        val_pred = np.asarray(model.predict(data["X_val"])).ravel()
        val_mse = float(np.mean((val_pred - data["y_val_scaled"]) ** 2))
        hit_cap = n_evals >= MAXITER
        restarts.append({"seed": seed, "model": model, "final_obj": final_obj,
                         "n_evals": n_evals, "val_mse": val_mse, "hit_cap": hit_cap})
        print(f"     seed={seed}: obj={final_obj:.6f} evals={n_evals}"
              f"{' [HIT 300 CAP]' if hit_cap else ''} val_mse={val_mse:.6f} "
              f"({time.perf_counter() - t0:.0f}s)")

    best = min(restarts, key=lambda r: r["val_mse"])
    print(f"[k={k}] SELECTED seed={best['seed']} on validation MSE={best['val_mse']:.6f}")

    bundle = persist(best["model"], data, best["seed"], k)
    print(f"[k={k}] artifacts -> {bundle}")

    # ---- score on the SHARED 10k test set (transforms applied, never refit) ----
    pred_scaled = np.asarray(best["model"].predict(data["X_test"])).ravel()
    pred_log = data["y_scaler"].inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    pred_real = np.expm1(pred_log)
    y_test_log = np.log1p(y_test_raw)

    d_real = diagnostics(y_test_raw, pred_real)
    d_log = diagnostics(y_test_log, pred_log)

    row = {
        "model": "VQR", "n_train": len(data["y_tr_scaled"]), "k": k,
        "r2_real": r2(y_test_raw, pred_real), "r2_log": r2(y_test_log, pred_log),
        "mae": mae(y_test_raw, pred_real), "rmse": rmse(y_test_raw, pred_real),
        "mape": mape(y_test_raw, pred_real),
        "std_ratio_real": d_real["std_ratio"], "r_real": d_real["r"],
        "pred_min_real": d_real["pred_min"], "pred_max_real": d_real["pred_max"],
        "std_ratio_log": d_log["std_ratio"], "r_log": d_log["r"],
        "pred_min_log": d_log["pred_min"], "pred_max_log": d_log["pred_max"],
        "selected_seed": best["seed"],
        "restart_objectives": {r["seed"]: round(r["final_obj"], 6) for r in restarts},
        "restart_evals": {r["seed"]: r["n_evals"] for r in restarts},
        "hit_cap_any": any(r["hit_cap"] for r in restarts),
        "selected": ";".join(data["selected"]),
    }
    print(f"[k={k}] TEST(10k): R2_real={row['r2_real']:+.4f} R2_log={row['r2_log']:+.4f} "
          f"std_ratio={row['std_ratio_real']:.3f} r={row['r_real']:.3f}")
    return row


# ==========================================================================
# PLOTTING
# ==========================================================================
def plot_vs_xgb(vqr_rows: list[dict], metric: str, ylabel: str, title: str,
                filename: str) -> str:
    xgb = pd.read_csv(os.path.join(TRAINING_EVA_RESULT, "ksweep_results.csv"))
    xgb1k = xgb[xgb["n_train"] == 1000].sort_values("k")

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    _style_axes(ax, title, ylabel)

    ax.plot(xgb1k["k"], xgb1k[metric], color=SERIES[1000], linewidth=2, marker="o",
            markersize=8, label="XGBoost n_train=1,000", zorder=3)
    ax.annotate("XGBoost", (xgb1k["k"].iloc[-1], xgb1k[metric].iloc[-1]),
                textcoords="offset points", xytext=(8, 0), color=INK_SECONDARY,
                fontsize=9, va="center")

    pts = sorted(vqr_rows, key=lambda r: r["k"])
    xs = [p["k"] for p in pts]
    ys = [p[metric] for p in pts]
    ax.plot(xs, ys, color=VQR_COLOR, linewidth=2, marker="s", markersize=8,
            label="VQR n_train=800", zorder=3)
    ax.annotate("VQR", (xs[-1], ys[-1]), textcoords="offset points", xytext=(8, 0),
                color=INK_SECONDARY, fontsize=9, va="center")

    ax.axhline(0, color=AXIS, linewidth=1, linestyle="--", zorder=1)
    ax.set_xticks(sorted(set(list(xgb1k["k"]) + xs)))
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
    t_start = time.perf_counter()
    print(f"[SETUP] VQR k-sweep {K_GRID_VQR}, maxiter={MAXITER}, seeds={RESTART_SEEDS}")
    print(f"[SETUP] scored on the shared {TEST_N:,}-row test set")
    print("[SETUP] estimator = StatevectorEstimator (LOCAL) - ZERO QuApp cloud calls")

    X_test_pool, y_test_raw = build_test_set()

    rows: list[dict] = []
    for k in K_GRID_VQR:
        rows.append(run_k(k, X_test_pool, y_test_raw))
        # incremental save so a long run never loses completed work
        with open(RESULTS_JSON, "w") as fh:
            json.dump(rows, fh, indent=2)
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
        print(f"[SAVE] progress written ({len(rows)}/{len(K_GRID_VQR)} done)")

    print("=" * 100)
    print(f"PART C - VQR vs qubit count, shared {TEST_N:,}-row test set")
    print("=" * 100)
    print(f"{'k':>3}{'R2 (real)':>12}{'R2 (log)':>11}{'MAE':>10}{'RMSE':>10}"
          f"{'std ratio':>11}{'r':>8}{'evals(cap?)':>14}")
    print("-" * 100)
    for r in rows:
        cap = "YES" if r["hit_cap_any"] else "no"
        evals = ",".join(str(v) for v in r["restart_evals"].values())
        print(f"{r['k']:>3}{r['r2_real']:>12.4f}{r['r2_log']:>11.4f}{r['mae']:>10.4f}"
              f"{r['rmse']:>10.4f}{r['std_ratio_real']:>11.4f}{r['r_real']:>8.4f}"
              f"{evals + ' (' + cap + ')':>14}")
    print("=" * 100)

    p1 = plot_vs_xgb(rows, "r2_real", "R² (real units)",
                     f"VQR vs XGBoost — R² (real) by k, shared {TEST_N:,}-row test set",
                     "vqr_vs_xgb_r2_real.png")
    p2 = plot_vs_xgb(rows, "r2_log", "R² (log space)",
                     f"VQR vs XGBoost — R² (log) by k, shared {TEST_N:,}-row test set",
                     "vqr_vs_xgb_r2_log.png")

    print(f"[DONE] results : {RESULTS_CSV}")
    print(f"[DONE] plot    : {p1}")
    print(f"[DONE] plot    : {p2}")
    print(f"[DONE] total elapsed = {(time.perf_counter() - t_start) / 60:.2f} min")
    print("[DONE] Zero QuApp cloud calls.")


if __name__ == "__main__":
    main()
