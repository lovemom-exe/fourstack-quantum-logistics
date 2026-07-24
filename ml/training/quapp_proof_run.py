# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: PROOF-OF-EXECUTION on real QuApp hardware (minimal credit).
#
#          This is NOT training and has NO optimization loop. It submits a
#          FIXED set of exactly N_CIRCUITS (=3) tiny 2-qubit circuits to the
#          QuApp backend, one at a time, and records the returned job IDs,
#          measurement histograms, and computed ZZ expectation values. The
#          goal is a citation-worthy artifact proving the pipeline really ran
#          real circuits on the device.
#
#          Each circuit is  zz_feature_map(2, reps=1) -> real_amplitudes(2,
#          reps=1)  bound to a distinct featured test-data row (2 features ->
#          2 qubits), which matches the estimator's ZZ observable so it does
#          not hit QuappEstimator.expectation's NotImplementedError.
#
#          Safety: N_CIRCUITS is fixed, there is exactly one loop over that
#          list, shots is left at the QuappEstimator default, and the total
#          number of credit-consuming jobs is printed at the end.
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import dotenv_values, find_dotenv

from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit.qasm2 import dumps
from qiskit.quantum_info import SparsePauliOp
from sklearn.preprocessing import MinMaxScaler

from utils.path import FRESH_RETAIL_EVAL_PATH_FEATURE, TRAINING_EVA_RESULT

# ==========================================================================
# PARAMETERS  (hard safety limits)
# ==========================================================================
N_CIRCUITS = 3                      # NEVER submit more than this
FEATURE_COLUMNS = ["avg_temperature", "avg_humidity"]  # 2 features -> 2 qubits
ROW_INDICES = [10, 200, 450]        # 3 distinct featured test rows
SCALER_FIT_ROWS = 500               # small sample used only to fit the scaler
# Fixed (non-trained) ansatz weights for real_amplitudes(2, reps=1) -> 4 params.
FIXED_ANSATZ_WEIGHTS = np.array([0.1, 0.2, 0.3, 0.4])

OBSERVABLE = SparsePauliOp("ZZ")
OUTPUT_JSON = os.path.join(TRAINING_EVA_RESULT, "quapp_proof.json")


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
def _require_token() -> str:
    """Read the ACCESS_TOKEN the same way QuappEstimator does; fail loudly."""
    env_path = find_dotenv()
    config = dotenv_values(env_path)
    token = config.get("ACCESS_TOKEN")
    if not token:
        raise SystemExit(
            "[FATAL] No ACCESS_TOKEN found.\n"
            f"        .env searched at: {env_path or '<no .env file located>'}\n"
            "        Deliverable 2 submits REAL circuits to the QuApp backend and "
            "needs the same ACCESS_TOKEN that QuappEstimator uses.\n"
            "        Create an .env (e.g. ml/.env) containing ACCESS_TOKEN=<token> "
            "and re-run. No jobs were submitted."
        )
    return token


def _load_feature_rows() -> np.ndarray:
    """Load N_CIRCUITS scaled 2-feature rows from the featured test data.

    Only the small eval CSV is read (the 996 MB train file is never touched).
    A MinMaxScaler is fit on a small sample so the two features land in [0, 1],
    a sensible angle range for the feature map.
    """
    df = pd.read_csv(
        FRESH_RETAIL_EVAL_PATH_FEATURE,
        usecols=FEATURE_COLUMNS,
        nrows=SCALER_FIT_ROWS,
    )
    features = df[FEATURE_COLUMNS].to_numpy(dtype=float)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    rows = scaled[ROW_INDICES]
    assert rows.shape == (N_CIRCUITS, len(FEATURE_COLUMNS))
    return rows


def _build_bound_qasm(feature_row: np.ndarray) -> str:
    """zz_feature_map(2,1) -> real_amplitudes(2,1), fully bound, as QASM2."""
    feature_map = zz_feature_map(feature_dimension=len(FEATURE_COLUMNS), reps=1)
    ansatz = real_amplitudes(num_qubits=len(FEATURE_COLUMNS), reps=1)
    circuit = feature_map.compose(ansatz)

    binding = {}
    for param, value in zip(feature_map.parameters, feature_row):
        binding[param] = float(value)
    for param, value in zip(ansatz.parameters, FIXED_ANSATZ_WEIGHTS):
        binding[param] = float(value)

    bound = circuit.assign_parameters(binding)
    return dumps(bound)


# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================
def main() -> None:
    _require_token()

    # Imported here so the loud token check above runs first.
    from algorithms.quapp_estimator import QuappEstimator

    estimator = QuappEstimator()
    feature_rows = _load_feature_rows()

    # Dry-run summary (non-interactive) so the credit spend is obvious up front.
    print(
        f"[DRY-RUN] About to submit {N_CIRCUITS} jobs to device_id="
        f"{estimator.device_id}, shots={estimator.shots} each. "
        "No training, no optimization loop."
    )
    print(f"[DRY-RUN] Observable = {OBSERVABLE.to_list()[0][0]}, "
          f"features = {FEATURE_COLUMNS}, rows = {ROW_INDICES}")

    results = []
    jobs_submitted = 0

    # The ONLY loop: exactly N_CIRCUITS fixed circuits, one job each.
    for idx, feature_row in enumerate(feature_rows):
        qasm = _build_bound_qasm(feature_row)

        job_id = estimator._invoke(qasm)      # reuse existing HTTP invoke
        jobs_submitted += 1
        print(f"[{idx + 1}/{N_CIRCUITS}] submitted job_id={job_id}")

        job = estimator._wait_job(job_id)     # reuse existing polling
        counts = job["histogram"]
        expectation = QuappEstimator.expectation(counts, OBSERVABLE)

        print(f"        counts={counts}")
        print(f"        <ZZ> = {expectation:+.6f}")

        results.append(
            {
                "index": idx,
                "job_id": job_id,
                "feature_columns": FEATURE_COLUMNS,
                "feature_row": [float(v) for v in feature_row],
                "counts": counts,
                "zz_expectation": expectation,
            }
        )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device_id": estimator.device_id,
        "shots": estimator.shots,
        "observable": OBSERVABLE.to_list()[0][0],
        "n_circuits": N_CIRCUITS,
        "total_jobs_submitted": jobs_submitted,
        "results": results,
    }

    os.makedirs(TRAINING_EVA_RESULT, exist_ok=True)
    with open(OUTPUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2)

    print("=" * 60)
    print(f"[DONE] Real circuits executed on QuApp device {estimator.device_id}.")
    print(f"[DONE] Job IDs: {[r['job_id'] for r in results]}")
    print(f"[DONE] Results saved to {OUTPUT_JSON}")
    print(f"[CREDIT] TOTAL credit-consuming jobs submitted: {jobs_submitted}")
    print("=" * 60)


if __name__ == "__main__":
    main()
