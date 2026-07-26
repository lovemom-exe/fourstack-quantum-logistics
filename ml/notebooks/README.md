# VQR training notebook

The main training file is:

```text
ml/notebooks/train_vqr_colab.ipynb
```

To run it:

1. Open `train_vqr_colab.ipynb` in Google Colab.
2. In the first configuration cell, set either the repository URL/directory or enable Google Drive and set the Drive project path.
3. Run the dependency-installation cell. Its versions match `ml/uv.lock`.
4. Run the remaining cells from top to bottom.
5. Start with `FAST_MODE = True`, 100 training rows, shallow circuits, and low optimizer evaluations.
6. Review `ml/results/vqr_experiments.csv`, the validation charts, failures, loss curves, and the Ridge comparison.
7. Increase samples or optimizer evaluations only after identifying the most promising small configurations. Higher feature counts mean more simulated qubits and can become very expensive.
8. Download or otherwise preserve `ml/results/` and `ml/models/perishable_vqr/` before the Colab runtime is discarded. If the repository is on Google Drive, confirm that these paths are inside the mounted project.

The test split is not used during configuration selection. The final notebook cells save the winning VQR with Qiskit's dill API, save its preprocessing objects and schemas, clear the in-memory model reference, and perform one clean reload/inference check suitable for later FastAPI integration.
