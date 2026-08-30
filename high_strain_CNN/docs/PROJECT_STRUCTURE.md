# Project structure

## Active code

```text
high_strain_CNN/
  tensorflow_reference/    Unmodified upstream TensorFlow training reference
  pytorch_autophasenn/     Shared PyTorch models/trainer; separate memmap/author readers
  simulation/
    author_generator.py          Supplied-author geometry and perturbation calls
    generate_author_dataset.py   Pure NPZ dataset generation
    evaluate_author_code.py      Stream generation through the official H5
    run_paper_model.py           Shared single-sample TF/PyTorch inference
    evaluate_paper_model.py      Existing-NPZ evaluation and threshold calibration
    visualization.py            Shared simulation figures
    sample_io.py                 Shared standard/compact NPZ phase reader and export writer
  experimental/           Official measured-data inference, without truth metrics
  tools/                  Conversion, parity checks, AutoPhaseNN export/validation
  configs/                AutoPhaseNN data schema
  requirements/           Separate backend environments
  vendor/                 Unchanged author source/potentials with SHA-256 inventory
  tests/                  Active scientific and data-handling regression tests
  docs/                   Current map and historical paper/source audit
  artifacts/              Results separated by dataset and backend
  archive/                Recoverable retired code; never imported by active code
```

## Routes

1. Supplied-author simulation: bundled `vendor/codes_for_BCDI_dataset_creation` ->
   `author_generator` -> NPZ -> official H5 inference -> WCA and figures.
2. AutoPhaseNN/PyTorch: original memmaps -> `pytorch_autophasenn.data` ->
   training/checkpoint -> evaluation/reconstruction -> visualization.
3. AutoPhaseNN/official TensorFlow: `tools.export_autophasenn_samples` -> NPZ ->
   `simulation.evaluate_paper_model`. This is not the author-simulation dataset.
4. Measured data: `experimental.run_official_data` -> reciprocal phase -> inverse
   FFT -> figures. Without ground truth this route does not report WCA or IoU.

The default PyTorch dataset reader remains the AutoPhaseNN memmap adapter.
`--data-format author_npz` selects `pytorch_autophasenn/author_data.py`, which
reads standard or compact fixed author samples. It does not simulate particles
online. See `COMPACT_AUTHOR_DATA.md` for generation, split, and worker commands.

## Supplied-author generation

Run from `high_strain_CNN/`. The source directory defaults to the bundled copy:

```bash
python -m simulation.generate_author_dataset \
  --output-dir /path/to/datasets/author_source_v2 \
  --profile paper --category-sampling random \
  --num-samples 900 --seed 20260830 \
  --oversampling-policy record --no-save-extras
```

This command writes fixed `I`, `phi`, and metadata arrays. Omitting
`--no-save-extras` also saves `object`, `support`, and `I_clean`; it is not the
compact format proposed below. No TensorFlow import or training is required.

`record` preserves and flags all source draws, including oversampling violations.
The default `error` stops on a violation. Neither mode resizes or retries a draw.
Formal paper-condition training requires a separately agreed selection policy.
Shape and phase probabilities and observations per particle remain explicit
reproduction assumptions. The Windows scattering backend is FFT/NUFFT, not
native PyNX. See `PAPER_PIPELINE_AUDIT_20260828.md` for numerical boundaries.

The latest completed source-call benchmark is
`artifacts/evaluations/simulation_tensorflow/author_generator_author_calls_v2_record_seed20260830_n900/`.
Earlier `author_generator_paper_*` results used retired sampling rules; preserve
their labels rather than combining their statistics with the latest run.

## Artifact policy

```text
artifacts/
  generation/<run-name>/          Generation console/log/PID/config, not sample arrays
  training/pytorch_autophasenn/    Lightweight run manifests and histories
  training/pytorch_simulation/    Author-dataset PyTorch run manifests/histories
  evaluations/
    autophasenn_pytorch/          AutoPhaseNN-trained PyTorch results
    autophasenn_tensorflow/       Official H5 tested on AutoPhaseNN samples
    simulation_tensorflow/       Official H5 tested on synthetic samples
    experimental_tensorflow/     Official H5 tested on measured data
  models/                        Downloaded/converted weights, local only
  parity/                        Reproducible numerical references, local only
  simulation/                    Generated datasets/caches, local only
  upstream_data/                 Downloaded experimental data, local only
  visualizations/                Rendered figures, local only
```

Server checkpoints remain under the configured external checkpoint root.
Never delete AutoPhaseNN source data, best/resumable checkpoints, or the latest
900-sample reference to make a code cleanup look like a disk cleanup. Inspect
large server files first; moving them on the same filesystem saves no space.

## Server storage: 90 GB available

Measured compressed sizes from the latest 900 samples, excluding checkpoints:

| Proposed contents | Estimated size for 102,000 samples | Status |
| --- | ---: | --- |
| `I + phi + metadata` | 107.6 GB | Existing lean NPZ output |
| Above plus `object + support` | 117.8 GB | Size estimate; extras currently also include `I_clean` |
| All current extras including `I_clean` | 217.2 GB | Existing full NPZ output |
| `I + object + support + metadata`, derive `phi` by FFT | About 19 GB | Implemented via `--storage compact` / `--data-format author_npz` |

The last estimate relies on the measured sparsity of these complex64 objects;
it is not a guarantee for other distributions or complex128 storage. Keep the
already sampled noisy `I` fixed; do not redraw Poisson noise during loading.
Derive the clean reciprocal phase using the source convention:

```python
phi = np.angle(np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(obj))))
```

The current stored complex64 object has already been rounded relative to the
source calculation. Deriving a label is numerically close, not bitwise identical
to the existing stored `phi`. Regression tests compare the two label paths and
their WCA losses; native Linux GPU generation still needs a server smoke test.

An all-900-sample read-only check found a maximum intensity-weighted circular
phase MAE of 1.12e-7 rad and a maximum WCA between the two label arrays of
4.20e-14. This is label agreement, not a neural-network performance result.
No regenerated label was byte-identical; the maximum phase error among voxels
with nonzero measured intensity was 4.58e-5 rad. Mean FFT/angle time was 0.0104 s
per sample, excluding file loading and decompression.

A local CPU check of one sample from each of nine categories (`OMP_NUM_THREADS=2`)
measured about 0.62-0.65 s per observation after geometry creation, versus about
0.011 s for an FFT-derived label, excluding disk/decompression time. Wulff
geometry construction took 2.8-4.0 s in this small selection; cache/reuse a
particle for its observations. These are not server throughput measurements.

## Reproducible online generation

Seed-only regeneration is possible, but the fixed dataset manifest must include
sample ID, particle ID, particle seed, observation seed, chosen shape/phase,
rotation and oversampling policies, source/config hashes, backend, and library
versions. Resolve category choices before workers start. Do not derive the
sample seed from epoch, worker ID, or loading order when a fixed dataset is
intended. Different seeds are not a mathematical guarantee of distinct arrays.

The current source uses global NumPy/Python random states. Concurrent threads
must not share those states; a future online loader should use separate worker
processes and bounded particle caches. Frozen validation/test sets must be split
by particle, not by observation. Seed repetition reproduced noisy intensities
exactly in nine checks, but multithreaded NUFFT introduced reciprocal-phase
differences up to 2.4e-7 rad. Cross-version/backend bitwise equality is not claimed.

## Retired files

The independent continuum generator and its guessed-distribution configs are
preserved in `archive/20260830_legacy_continuum/`. Its manifest records eight
original files and their hashes. The original mixed test file is stored with a
`.py.txt` suffix; active I/O, reconstruction, threshold, and cache checks remain
under `tests/`. Source cleanup did not delete or move experiment data/weights.

Keep historical audit reports unchanged as provenance. Their old generator
commands refer to the archived route; use the current entry points above for
new author-source experiments.
