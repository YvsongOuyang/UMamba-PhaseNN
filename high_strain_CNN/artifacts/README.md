# Artifact policy

`artifacts/` separates experiment outputs from source code.

- `training/pytorch_autophasenn/` contains lightweight configs, histories,
  logs, manifests, PIDs, and TensorBoard events that may be tracked in Git.
- `evaluations/autophasenn_pytorch/` contains the existing PyTorch evaluation
  summaries, per-sample tables, and threshold sweeps and may be tracked.
- `evaluations/simulation_tensorflow/` contains lightweight reports from the
  official TensorFlow H5 evaluated on reproduced paper-style simulations.
- `models/`, `parity/`, `simulation/`, and `visualizations/` are generated or
  large local outputs and are ignored by Git.
- Future official-model results should use the distinct
  `evaluations/autophasenn_tensorflow/` namespace.

Do not rewrite paths embedded in historical manifests: they identify the
server-side checkpoints and data that produced each result.
