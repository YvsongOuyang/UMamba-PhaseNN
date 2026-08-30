# TensorFlow reference implementation

`train_upstream.py` is the author TensorFlow 2.10.1 implementation vendored
from upstream commit `43d31f3`. It defines the published network, `.npz`
loader, WCA loss, and training loop.

Keep upstream behavior isolated in this directory. AutoPhaseNN samples are
adapted by `tools/export_autophasenn_samples.py`, then passed to the shared
`simulation/evaluate_paper_model.py` entry point for official-H5 inference,
evaluation and visualization. The adapter does not modify `train_upstream.py`.

The official H5 model is a local artifact at
`artifacts/models/model_paper.h5` and is intentionally not tracked in this
parent repository.

`simulation/evaluate_paper_model.py` loads that H5 directly for full-factorial
simulation evaluation, support-threshold calibration, held-out metrics, and
representative 2D/3D visualizations. It does not use converted PyTorch weights.
