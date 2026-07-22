# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
import numpy as np
import inspect
from qiskit.primitives import PrimitiveJob

# ==========================================================================
# PARAMETERS
# ==========================================================================

# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================

print(inspect.signature(PrimitiveJob))
print(inspect.getsource(PrimitiveJob.__init__))
# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================
