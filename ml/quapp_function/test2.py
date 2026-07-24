import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

_cache = {}

# H2 STO-3G Hamiltonian at r=0.735 Angstrom (Jordan-Wigner)
_H2_COEFFS = {
    "II": -0.8105479805373266,
    "ZI": +0.1721839326191232,
    "IZ": -0.2257534922240237,
    "ZZ": +0.1229330365024460,
    "XX": +0.1686890435184897,
    "YY": +0.1686890435184897,
}
_NUCLEAR_REPULSION = 0.7151043390810673

_PAULIS = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _real_amplitudes(n_qubits: int, reps: int, params: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for _ in range(reps):
        for q in range(n_qubits):
            qc.ry(params[idx], q)
            idx += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    for q in range(n_qubits):
        qc.ry(params[idx], q)
        idx += 1
    return qc


def _pauli_expectation(qc_base: QuantumCircuit, pauli: str, sim: AerSimulator) -> float:
    if pauli == "II":
        return 1.0
    n = qc_base.num_qubits
    qc = qc_base.copy()
    for i, p in enumerate(reversed(pauli)):
        if p == "X":
            qc.h(i)
        elif p == "Y":
            qc.sdg(i)
            qc.h(i)
    qc.measure_all()
    result = sim.run(transpile(qc, sim), shots=8192).result()
    counts = result.get_counts()
    total = sum(counts.values())
    exp = 0.0
    for bitstr, cnt in counts.items():
        parity = sum(int(b) for b in bitstr) % 2
        exp += (1 - 2 * parity) * cnt / total
    return exp


def _energy(params: np.ndarray, reps: int, sim: AerSimulator, history: list) -> float:
    qc = _real_amplitudes(2, reps, params)
    e = sum(
        coeff * _pauli_expectation(qc, pauli, sim)
        for pauli, coeff in _H2_COEFFS.items()
    )
    history.append(float(e))
    return e


def processing(invocation_input: dict) -> QuantumCircuit:
    qasm = invocation_input.get("qasm")
    if qasm is None:
        raise ValueError("'qasm' is required.")

    return QuantumCircuit.from_qasm_str(qasm)


def post_processing(job_result) -> dict:
    try:
        counts = job_result.get_counts()
    except Exception:
        counts = None

    return {
        "counts": counts,
        "success": job_result.success,
        "backend": job_result.backend_name,
        "shots": job_result.results[0].shots,
    }
