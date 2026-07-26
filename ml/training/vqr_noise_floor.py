# ==========================================================================
# Author: Nguyen Minh Hoang
# Purpose: Do SAN NHIEU cua giao thuc VQR cu (precision mac dinh 0.015625).
#
#          Cau hoi can tra loi: xu huong R^2 theo k do duoc o muc 5 co that
#          khong, hay chi la dao dong giua cac lan chay?
#
#          Cach do: lap lai NGUYEN VEN quy trinh cua mot k (3 restart co seed ->
#          chon theo validation -> cham tren test set chung 10k) N lan, voi
#          DUNG cau hinh cu. Moi thu deu giong het nhau giua cac lan; khac biet
#          quan sat duoc chinh la nhieu.
#
#          So sanh bien do nay voi khoang bien thien cua duong cong k=4..7
#          (0.075 o R^2 real). Neu hai cai cung co thi "xu huong" khong co y
#          nghia.
#
#          Chay hoan toan cuc bo - ZERO QuApp cloud calls.
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from evaluation.metrics import mae, r2, rmse
from training.eval_ksweep import TEST_N, build_test_set, diagnostics
from training.train_vqr_ksweep import fit_once, prepare
from training.train_vqr_local import MAXITER, RESTART_SEEDS
from utils.path import TRAINING_EVA_RESULT

# ==========================================================================
# PARAMETERS
# ==========================================================================
NOISE_K = 4                  # k re nhat -> nhieu lan lap nhat trong cung ngan sach
N_REPEATS = 3
NOISY_PRECISION = 0.015625   # DUNG mac dinh cu cua EstimatorQNN

RESULTS_JSON = os.path.join(TRAINING_EVA_RESULT, "vqr_noise_floor.json")
RESULTS_CSV = os.path.join(TRAINING_EVA_RESULT, "vqr_noise_floor.csv")

# khoang bien thien cua duong cong k=4..7 duoi giao thuc cu, de doi chieu
MEASURED_TREND_SPAN_REAL = 0.0752   # -0.1059 -> -0.0307
MEASURED_TREND_SPAN_LOG = 0.1751    # -0.4683 -> -0.2932


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
def one_full_repeat(rep: int, data: dict, y_test_raw: np.ndarray) -> dict:
    """Mot lan chay tron ven: 3 restart -> chon theo validation -> cham test."""
    restarts = []
    for seed in RESTART_SEEDS:
        t0 = time.perf_counter()
        model, final_obj, n_evals = fit_once(
            data["X_tr"], data["y_tr_scaled"], NOISE_K, seed, MAXITER,
            verbose=False, precision=NOISY_PRECISION,
        )
        val_pred = np.asarray(model.predict(data["X_val"])).ravel()
        val_mse = float(np.mean((val_pred - data["y_val_scaled"]) ** 2))
        restarts.append({"seed": seed, "model": model, "obj": final_obj,
                         "n_evals": n_evals, "val_mse": val_mse})
        print(f"     [rep {rep}] seed={seed}: obj={final_obj:.6f} evals={n_evals} "
              f"val_mse={val_mse:.6f} ({time.perf_counter() - t0:.0f}s)", flush=True)

    best = min(restarts, key=lambda r: r["val_mse"])

    pred_scaled = np.asarray(best["model"].predict(data["X_test"])).ravel()
    pred_log = data["y_scaler"].inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    pred_real = np.expm1(pred_log)
    y_test_log = np.log1p(y_test_raw)

    row = {
        "repeat": rep, "k": NOISE_K, "precision": NOISY_PRECISION, "maxiter": MAXITER,
        "r2_real": r2(y_test_raw, pred_real), "r2_log": r2(y_test_log, pred_log),
        "mae": mae(y_test_raw, pred_real), "rmse": rmse(y_test_raw, pred_real),
        "std_ratio_real": diagnostics(y_test_raw, pred_real)["std_ratio"],
        "r_real": diagnostics(y_test_raw, pred_real)["r"],
        "selected_seed": best["seed"],
        "restart_evals": {r["seed"]: r["n_evals"] for r in restarts},
        "val_mse_selected": best["val_mse"],
    }
    print(f"  [rep {rep}] -> R2_real={row['r2_real']:+.4f} "
          f"R2_log={row['r2_log']:+.4f} (seed {best['seed']})", flush=True)
    return row


def summarise(rows: list[dict]) -> None:
    def spread(col: str) -> tuple[float, float, float]:
        v = np.array([r[col] for r in rows], dtype=float)
        return float(v.min()), float(v.max()), float(v.std(ddof=1))

    print("=" * 92)
    print(f"SÀN NHIỄU — giao thức CŨ (precision={NOISY_PRECISION}), k={NOISE_K}, "
          f"{len(rows)} lần lặp")
    print("=" * 92)
    print(f"{'lần':>5}{'R² (real)':>13}{'R² (log)':>12}{'seed chọn':>12}{'evals':>18}")
    print("-" * 92)
    for r in rows:
        ev = ",".join(str(v) for v in r["restart_evals"].values())
        print(f"{r['repeat']:>5}{r['r2_real']:>13.4f}{r['r2_log']:>12.4f}"
              f"{r['selected_seed']:>12}{ev:>18}")
    print("-" * 92)

    for col, span, label in (("r2_real", MEASURED_TREND_SPAN_REAL, "R² (real)"),
                             ("r2_log", MEASURED_TREND_SPAN_LOG, "R² (log)")):
        lo, hi, sd = spread(col)
        rng = hi - lo
        verdict = ("XU HƯỚNG KHÔNG ĐÁNG TIN — nhiễu cùng cỡ hoặc lớn hơn"
                   if rng >= 0.5 * span else
                   "xu hướng lớn hơn nhiễu rõ rệt")
        print(f"{label}: min={lo:+.4f} max={hi:+.4f} biên độ={rng:.4f} "
              f"std={sd:.4f}")
        print(f"    so với khoảng biến thiên k=4..7 ({span:.4f}): "
              f"nhiễu/xu hướng = {rng / span:.2f}x -> {verdict}")
    print("=" * 92)


# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================
def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    print(f"[SETUP] lặp {N_REPEATS} lần cấu hình CŨ tại k={NOISE_K}, "
          f"maxiter={MAXITER}, precision={NOISY_PRECISION}")
    print(f"[SETUP] chấm trên test set chung {TEST_N:,} dòng")
    print("[SETUP] mục đích: biết dao động giữa các lần chạy có lớn bằng "
          "'xu hướng' theo k hay không")

    X_test_pool, y_test_raw = build_test_set()
    data = prepare(NOISE_K, X_test_pool)

    rows = []
    for rep in range(1, N_REPEATS + 1):
        rows.append(one_full_repeat(rep, data, y_test_raw))
        with open(RESULTS_JSON, "w") as fh:
            json.dump(rows, fh, indent=2)
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    summarise(rows)
    print(f"[DONE] {RESULTS_CSV}")


if __name__ == "__main__":
    main()
