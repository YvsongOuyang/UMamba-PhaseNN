# Retired continuum simulator

This is a recoverable archive, not the current author-source implementation.
Files were moved on 2026-08-30 without changing their bytes, including pending
local edits. Original paths, sizes, and SHA-256 hashes are in `manifest.json`.
Eight files occupy 87,025 bytes. Moving them on the same disk frees no space.

## Why these files were retired

- `simulation/config.py`, `generator.py`, and `generate_dataset.py`: the early
  independently implemented voxel/continuum simulator, including assumed
  parameter ranges and support-phase-span scaling. Use the active
  `simulation.generate_author_dataset` command for the supplied-author route.
- `simulation/__init__.py`: the previous package exports, retained for rollback.
- `configs/`: the two sampling profiles for that retired simulator.
- `tests/test_simulation.py.txt`: the complete pre-cleanup test file. The text
  suffix prevents accidental discovery. Tests for active NPZ, FFT, support
  thresholds, and prediction caches remain in the active `tests/` directory.
- `docs/PROJECT_STRUCTURE.before_cleanup.md`: historical commands and workflow
  descriptions, retained to interpret earlier experiments.

Active AutoPhaseNN export still needs the original NPZ record and writer. Those
two definitions were extracted unchanged into `simulation/sample_io.py`; active
code does not import anything from this archive.

## Preservation boundary

The supplied external author sources, official TensorFlow model/code, PyTorch
models/training, experimental-data workflow, evaluation outputs, and all
datasets/checkpoints were left in place. Historical reports keep their original
paths and protocols; do not relabel them as author-source results.

This snapshot is not a separately maintained runnable package. To revisit the
old simulator, restore the original paths from the manifest in a separate
checkout. Do not overwrite the current package exports or tests in place.

Small archived source files can be versioned. Do not add datasets, predictions,
weights, virtual environments, or caches to Git under this directory.
