# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: My own  Estimator
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
from qiskit.qasm2 import dumps

import time
import httpx

from utils.path import ACCESS_TOKEN_PATH
from utils.api_util import (
    function_name,
    project_id,
    device_id,
    shots,
    workspace_id,
)

# ==========================================================================
# PARAMETERS
# ==========================================================================
with open(ACCESS_TOKEN_PATH, "r") as access_token:
    ACCESS_TOKEN = access_token.read().strip()
func_name = function_name
prj_id = project_id
dev_id = device_id
shots = shots


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================
class QuappEstimator:
    def __init__(
        self,
        token: str = ACCESS_TOKEN,
        project_id: str = prj_id,
        tenant_id: str = workspace_id,
        function_name: str = func_name,
        device_id: int = dev_id,
        shots: int = shots,
        base_url: str = "https://functions.quapp.cloud",
    ):
        if token == "":
            raise ValueError("[!] Missing access token")

        self.function_name = function_name
        self.device_id = device_id
        self.shots = shots

        self.client = httpx.Client(
            base_url=base_url,
            timeout=120,
            headers={
                "Authorization": f"Bearer {token}",
                "X-Project-Id": project_id,
                "X-Tenant-Id": tenant_id,
            },
        )

    def run(self, pubs, precision=None):
        result = []

        for circuit, observable, params in pubs:
            bound = circuit.assign_parameters(params[0])
            qasm = dumps(bound)

            # Invoke Function
            jobId = self._invoke(qasm)

            # Job
            job = self._wait_job(jobId)
            counts = job["histogram"]

            ev = self.expectation(counts, observable)
            result.append(ev)
        return result

    # ----------------------------------------------------------
    # API
    # ----------------------------------------------------------
    def _invoke(self, qasm: str) -> str:
        json_data = {
            "functionName": self.function_name,
            "deviceId": self.device_id,
            "description": "estimator",
            "shots": self.shots,
            "input": {"qasm": qasm},
        }
        response = self.client.post(
            "/api/v1/third-party/function/invoke", json=json_data
        )
        print(response.status_code)
        print(response.json())
        response.raise_for_status()
        if "data" not in response.json():
            raise RuntimeError(response.json())

        return response.json()["data"]

    def _wait_job(self, jobId: str) -> dict:
        """Run the job in cloud

        Args:
            jobId (str): job ID

        Raises:
            RuntimeError: raise when the status return ERROR

        Returns:
            dict: the "data" in json format
        """
        while True:
            response = self.client.get(f"api/v1/third-party/jobs/{jobId}/detail")
            response.raise_for_status()

            data = response.json()["data"]
            status = data["status"]

            if status == "DONE":
                return data
            if status == "ERROR":
                raise RuntimeError(data)

            time.sleep(1)

    # ----------------------------------------------------------
    # Quantum Part
    # ----------------------------------------------------------
    @staticmethod
    def expectation(counts: dict, observable) -> float:
        pauli = observable.to_list()[0][0]

        if pauli != "ZZ":
            raise NotImplementedError(f"Observable {pauli} is not supported yet.")

        total = sum(counts.values())

        value = 0.0

        for bitstring, count in counts.items():

            parity = bitstring.count("1") % 2

            eigenvalue = 1 if parity == 0 else -1

            value += eigenvalue * count

        return value / total


# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================
def main():
    # print(ACCESS_TOKEN)
    pass


if __name__ == "__main__":
    main()
