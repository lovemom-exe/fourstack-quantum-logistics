# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: My own  Estimator
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
from qiskit.qasm2 import dumps


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
class QuappEstimator:
    def __init__(self):
        pass

    def run(self, pubs, precision=None):
        results = []

        for circuit, observable, params in pubs:
            bound = circuit.assign_parameters(params[0])
            qasm = dumps(bound)


# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================
