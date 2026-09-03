# Compact author dataset

## Stored and reconstructed fields

The new `--storage compact` option changes storage, not the author's geometry,
strain, orientation, oversampling, or Poisson sampling calls.

| Field | Stored type | Meaning |
| --- | --- | --- |
| `I` | float32 | Fixed Poisson-noisy **linear intensity**, not amplitude |
| `object` | complex64 | Clean complex real-space object before Poisson noise |
| `support` | bool | Author-generated real-space truth mask |
| `metadata_json` | JSON string | Seeds, particle/observation IDs, categories, parameters, backend, format |

`phi` and `I_clean` are omitted. Loading reconstructs the wrapped label with the
author's convention, then subtracts its central-voxel phase for training:

```python
phi = np.angle(np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(object))))
phi = phi - phi[32, 32, 32]
x = np.log1p(I)
x = (x - x.min()) / (x.max() - x.min())
```

There is **no additional center-of-mass correction**, amplitude squaring,
noise redraw, random augmentation, particle regeneration, or epoch-dependent
seed here. WCA uses `x` as its weight, as in the existing training loss.
Complex64 object rounding means numerical, not byte-for-byte, label agreement.
The previous 900-sample audit measured maximum intensity-weighted circular
label MAE 1.12e-7 rad and label-to-label WCA 4.20e-14. These are not model scores.

The measured compression estimate for 95,000/4,000/3,000 samples is about 19 GB,
excluding checkpoints, filesystem overhead, and evaluation prediction caches.
Compression is distribution-dependent. Keeping only best/last checkpoints via
`--save-every 0` avoids accumulating periodic copies on the 90 GB server budget.

## Generate on Linux

Run from `high_strain_CNN/`. Use a fresh output directory. The manifest is
written after successful completion; do not train on an incomplete directory.
The generator refuses to overwrite existing samples. An interrupted paper-profile
run can be resumed explicitly using its saved configuration, as described below.

The original author code and gold potential resources are bundled under
`vendor/codes_for_BCDI_dataset_creation/` and selected by default. After pulling
the repository, no separate source upload or `AUTHOR_CODE_DIR` variable is
needed. `--author-code-dir` is only needed to override the bundled copy.

Use an environment containing the author's scientific dependencies (NumPy,
SciPy, ASE, matplotlib, scikit-image, scikit-learn) and a CUDA-enabled PyNX with
PyCUDA and a working CUDA compiler; include `ipywidgets` for the source's
interactive utility imports. Refer to the [official PyNX installation
instructions](https://pynx.esrf.fr/en/latest/install.html). TensorFlow and
PyTorch are not imported by the generator. The compatibility environment is
listed in `requirements/author-simulation.txt`; it also includes TensorFlow
for the separate official-model evaluation workflow.

```bash
DATA_DIR="/data_ssd/oyys/high_strain_cnn/dataset"
RUN_NAME="author_compact_seed20260830_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$PWD/artifacts/generation/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

nohup env CUDA_VISIBLE_DEVICES=0 python -u -m simulation.generate_author_dataset \
  --output-dir "${DATA_DIR}" --log-dir "${RUN_DIR}" \
  --storage compact --scattering-backend pynx_cuda \
  --split-counts 95000 4000 3000 \
  --seed 20260830 --oversampling-policy record \
  --workers 4 --print-freq 50 \
  > "${RUN_DIR}/console.log" 2>&1 < /dev/null &

PID=$!
echo "${PID}" > "${RUN_DIR}/generate.pid"
echo "Submitted PID=${PID}; data=${DATA_DIR}; logs=${RUN_DIR}"
tail -f "${RUN_DIR}/console.log"
```

`--output-dir` holds the sample NPZs, `dataset_manifest.json`, and a small
`.generation.lock` file preventing concurrent writers. The training reader needs
the manifest alongside the data. `--log-dir` holds `generation.log`
and `config.json`; the shell command above places `console.log` and
`generate.pid` there too. Without `--log-dir`, a timestamped directory under
the subproject's `artifacts/generation/` is selected automatically. Existing
logs in older dataset directories are not moved or deleted.

For an initial smoke test use `--split-counts 48 12 12` and a different output
directory. Choose GPUs through the CUDA environment/device visibility settings
of the installed PyNX/PyCUDA runtime. The source passes `gpu_name=""`, which
requests all visible GPUs. This is an **offline** job: avoid competing for the
same GPU while training.

`pynx_cuda` calls the real author-requested `Fhkl_thread(language="cuda")` and
real Thomson factor ([PyNX scattering API](https://pynx.esrf.fr/en/latest/modules/scattering/index.html)).
Startup checks PyCUDA devices and a native kernel call;
failure is an error, not an adapter fallback. Geometry, strain, Poisson noise,
and compression still use CPU; it is not an all-GPU generator. PyNX and the
compatibility FFT/NUFFT backend are not claimed bitwise equivalent or equally
fast. Native CUDA execution must be verified on the server.

Use `--scattering-backend compat` for the existing CPU FFT/NUFFT implementation.
The default remains `compat`; merely installing CUDA does not select it.
The default storage remains `standard` for backward compatibility: always
pass `--storage compact` for the compact dataset.

`--oversampling-policy record` preserves and flags the source draws even when
their measured oversampling is <=2. It does not filter, rescale, or retry them.
The alternative/default `error` stops on such a draw. This preserves the
previous source-call experiment policy; it is not a claim that all samples
satisfy the paper's oversampling condition.

### GPU execution and bounded concurrency

The supplied author notebook generates one observation, not a batched dataset.
Its `Create_diffraction` calls `Fhkl_thread(..., language="cuda", gpu_name="")`.
Neither that wrapper nor the documented PyNX scattering API exposes a dataset
`batch_size`. PyNX's `nbCPUthread` applies to its CPU backend, not GPU batches.
`sizeQ=64` is spatial resolution and `nstep` is reciprocal sampling; neither is
a batch setting. Do not change them to increase throughput.

Our CPU compatibility route uses FFT/NUFFT rather than PyNX's direct atomic sum;
similar CPU/GPU end-to-end times do not by themselves imply CUDA is inactive.
Both routes still perform geometry, grid rotation, amplitude/phase processing,
noise and compression on the CPU. In a local CPU profile, constructing the first
Wulff particle took about 4.0 seconds while its observation took about 0.8 seconds,
including about 0.33 seconds for the source's point-by-point grid rotation. These
are one-sample local timings, not a server CUDA benchmark.

`--workers` defaults to **1**. Try **2 or 4**, using the same seed and counts in
separate small test directories before selecting the faster setting. Each worker
is a spawned process with its own source RNGs and native CUDA context. This can
overlap CPU preparation and processing, but a single GPU may serialize concurrent
calls; increasing worker count or VRAM use is not a guarantee of higher throughput.
Do not start 50 CUDA processes as a substitute for a batch-size setting.

Each task retains the original three observations per particle. Shape/phase
choices are drawn centrally in the original order. Particle and observation seeds
and split boundaries do not depend on scheduling. The queue is bounded to twice
the worker count; workers write their own samples and send only metadata back.
`--worker-threads 1` limits numeric CPU threads in each spawned worker, avoiding
nested thread oversubscription. The serial route preserves its existing thread
environment. Author source files and scientific function calls are unchanged.

`--print-freq 50` controls **logging only**. Progress shows new/reused counts,
wall-clock throughput and an HH:MM:SS ETA. Mean worker times separate `geometry`,
`q_grid`, `scattering`, `reconstruction`, `amplitude`, `phase`, `noise_and_labels`,
`write` and `hash`. Geometry is charged once per particle, not once per observation.
Parallel stage times overlap; do not sum them to estimate wall-clock elapsed time.
`scattering` includes the source wrapper and transfers, not just CUDA kernel time.
Startup and old-file validation can inflate the initial ETA.

### Resume an interrupted generation run

**Stop the previous generator and wait for it (and its workers) to exit first.**
Never run the old serial script and the new script against the same output folder.
New versions hold an OS lock; legacy versions did not. Resume also checks the
legacy run's recorded Linux PID. A completed manifest is never overwritten.

Use a **new** log directory and pass the **previous** run's `config.json`. All
dataset settings, including the output directory, seed, backend, categories and
split counts, are inherited. Conflicting dataset overrides are rejected. Only
execution/logging options such as worker count should change:

```bash
OLD_RUN="$PWD/artifacts/generation/author_compact_seed20260830_20260830_235306"
RUN_DIR="$PWD/artifacts/generation/author_compact_resume_w4_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"

nohup env CUDA_VISIBLE_DEVICES=0 python -u -m simulation.generate_author_dataset \
  --resume-from "${OLD_RUN}/config.json" \
  --log-dir "${RUN_DIR}" --workers 4 --print-freq 50 \
  > "${RUN_DIR}/console.log" 2>&1 < /dev/null &

PID=$!
echo "${PID}" > "${RUN_DIR}/generate.pid"
echo "Submitted PID=${PID}; logs=${RUN_DIR}"
```

Existing samples are checked against the deterministic schedule (seeds, particle
IDs, shapes, phases, rotation, oversampling and backend), array schemas/values and
ZIP CRCs. They are reused without overwriting or redrawing their noise. Missing
observations, including partial particles, are generated with their original seeds.
Malformed files stop resume with a named error; they are not silently deleted.
After confirming an old interrupted file is incomplete, move only that file out
of the dataset and retry. New writes use a temporary file followed by an atomic
rename, so interruption does not publish a half-written sample.

New run configs record author-source hashes and the generator protocol. Older
configs without source hashes produce a provenance warning: metadata and file
integrity are checked, but historical source identity cannot be verified from
those configs. Keep the same software environment and visible GPU when resuming;
matching seeds does not promise bitwise results across library/hardware changes.
The final manifest is sorted by sample index, regardless of completion order.

One shape and one phase family are selected per observation, with a shape
reused for the particle's observations. Category sampling defaults to random,
not forced 3x3 balancing. Source phase-function random distributions are
unchanged. Three observations per particle is still an explicit reproduction
assumption. At a split boundary the final particle can have fewer observations;
the next split always starts a new particle seed. Train/val/test geometry is
disjoint, and the training reader checks this across the entire manifest.

## Train and measure loading

The synced server run `author_compact_seed20260830_20260831_001855` completed
102000 observations and logged a final manifest on 2026-08-31 at 04:00:58.
Its original train/val/test counts are 95000/4000/3000. Counting unique warning
indices in its complete generation log gives the following expected filtered
counts; the training reader verifies the actual manifest on the server:

| Split | Original | Excluded at <=2 on any axis | Eligible |
| --- | ---: | ---: | ---: |
| Train | 95000 | 14469 | 80531 |
| Validation | 4000 | 617 | 3383 |
| Test | 3000 | 476 | 2524 |

Use the existing PyTorch training environment. Example, from scratch with the
existing default reduced model and unchanged optimizer/loss/scheduler defaults:

```bash
python -u -m pytorch_autophasenn.train \
  --data-format author_npz \
  --data-dir /data_ssd/oyys/high_strain_cnn/dataset \
  --author-min-oversampling 2 \
  --run-name author_compact_reduced_scratch \
  --num-workers 4 --save-every 0
```

No `--pretrained`/`--resume` means scratch; no `--fp16` means normal float32.
Model variants and their learning-rate defaults are unchanged. Select a
different existing variant explicitly when running an ablation.

Train/val counts come from `dataset_manifest.json`; optional
`--num-samples-train`/`--num-samples-val` select a **fixed prefix** of those splits
(after any filter). Omitting them keeps the full eligible pools.
Author-data validation keeps its final partial batch. Training retains the
existing `drop_last=True` behavior; its split must contain at least one batch.

### Persist the filtered split lists

All NPZs live in one directory. `dataset_manifest.json` records each filename
and its `train`, `val` or `test` split; there are no physical split directories.
To persist the strict >2 selection on the server, stop any reader/training job
first, then run this once from `high_strain_CNN/`:

```bash
python -m tools.filter_author_manifest \
  --data-dir /data_ssd/oyys/high_strain_cnn/dataset
```

The default cutoff is 2 on **every** axis. Add `--dry-run` to validate and show
counts without updating the manifest. This only edits metadata: it does not
read array contents, resimulate, move, copy or delete any NPZ. It does check file
existence and the entire original manifest for duplicate entries and particle
leakage before filtering. Exact-boundary values of 2 are excluded.

The original manifest is kept byte-for-byte as `dataset_manifest.unfiltered.json`.
The active manifest loses excluded `samples` entries and updates `num_samples`,
`splits` and `category_counts`. Retained filenames, ordering, split assignments,
sample metadata and hashes are unchanged. `index_filter` records the cutoff,
backup hash, original/excluded split counts and excluded filenames per split.
Generation timing and execution fields still describe the original generation.
Updates use the existing dataset writer lock and an atomic file replacement;
repeating the same command checks the backup and makes no further changes.

The existing `AuthorNPZPhaseDataset` automatically reads the revised train/val/test
lists without `--author-min-oversampling`. Supplying that option again is harmless.
Run manifests also retain the persistent filter's provenance. Do not limit
`--num-samples-train` or `--max-batches-per-epoch` for a full eligible-pool epoch.
This does not replenish removed examples or mix any held-out particle into training.

Tools that simply glob all `sample_*.npz` still see excluded files. In particular,
the older standalone TensorFlow threshold-sweep script does not use these split
lists; do not use its directory scan as the filtered dataset's held-out test.
Held-out readers must select `split="test"` from this active manifest.

### Optional read-time filter

`--author-min-oversampling 2` requires all three recorded oversampling values to
be strictly greater than 2. This is an optional **index filter**, disabled by
default to preserve prior runs. It does not rewrite NPZs or the source manifest,
rescale particles, redraw noise, or look at model performance. Missing/invalid
oversampling metadata causes an explicit error when filtering is enabled.
The same rule is applied to train and validation. The shared dataset class also
supports `split="test", min_oversampling=2` for a held-out evaluation reader.

Splits remain particle-disjoint: leakage checks inspect the entire unfiltered
manifest before an index can be used. Run metadata records the source manifest
hash, threshold, original/eligible/excluded counts for every split and a hash of
the selected filenames. Startup logs print these counts. This does not replenish
excluded observations: a 4000-observation validation split may yield fewer eligible
observations. Keep the test split out of model selection; filling splits back to
paper counts is separate work, not required for a compute-limited pilot.

### Compute-limited training

The existing `reduced` model caps the bottleneck at 1024 channels and has
39,160,897 parameters. It can read compact author data without model changes.
The separate `reduced_bn_no_outer_skip` variant is not implicitly selected.

Prefer full passes through the entire eligible training pool, reshuffling each
epoch, rather than permanently restricting training to the first 25000 observations.
Leave `--max-batches-per-epoch` at its default 0. At batch size 16, 80531 eligible
observations give 5033 training batches per full epoch (the final three are dropped).
The example below uses 20 full epochs as an initial compute budget, not a claim
that 20 epochs are enough to converge.

For shorter feedback cycles only, the optional limit of 1563 batches processes
25008 observations per short epoch. Short epochs reshuffle the full pool and
can overlap across epochs, so four short rounds do not guarantee full coverage.
They do not save training compute at equal total updates. Compare optimizer
steps or processed observations, not epoch numbers alone, with the paper.

The existing `--max-batches-per-epoch` caps **both** train and validation. With
at most 4000 validation observations and batch size 16, validation needs at most
250 batches, so the 1563 limit still evaluates the complete fixed eligible
validation split every epoch. A smaller limit or batch size may truncate validation;
startup now reports actual batch budgets and warns about that case.

Example background pilot, from `high_strain_CNN/` in the PyTorch environment:

```bash
RUN_NAME="author_reduced_osgt2_full_bs16_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$PWD/artifacts/training/pytorch_simulation/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

nohup env CUDA_VISIBLE_DEVICES=0 python -u -m pytorch_autophasenn.train \
  --data-format author_npz \
  --data-dir /data_ssd/oyys/high_strain_cnn/dataset \
  --author-min-oversampling 2 \
  --model-variant reduced --run-name "${RUN_NAME}" \
  --epochs 20 --batch-size 16 \
  --save-every 0 \
  > "${RUN_DIR}/console.log" 2>&1 < /dev/null &

PID=$!
echo "${PID}" > "${RUN_DIR}/train.pid"
echo "Submitted PID=${PID}; logs=${RUN_DIR}/console.log"
```

This starts from scratch, with default float32, four loader workers, Adam at
1e-4 and ReduceLROnPlateau (factor 0.5, patience 5, minimum LR 1e-6). The scheduler
uses fixed-validation WCA after each full epoch. Changing to short rounds also
changes scheduler and validation frequency. Defaults retain best/last checkpoints under
`/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/high_strain_cnn/<run-name>`.

For a controlled global-context ablation against the completed
`reduced_bn_no_outer_skip` run, use the same filtered manifest, 30 full epochs,
batch size 16, seed 42, Adam at 1e-3, and the same plateau scheduler. The Mamba
run must omit both `--pretrained` and `--resume`:

```bash
RUN_NAME="author_bn_no_outer_skip_mamba8_scratch_bs16_lr1e-3_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$PWD/artifacts/training/pytorch_simulation/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

nohup env CUDA_VISIBLE_DEVICES=0 python -u -m pytorch_autophasenn.train \
  --data-format author_npz \
  --data-dir /data_ssd/oyys/high_strain_cnn/dataset \
  --model-variant reduced_bn_no_outer_skip_mamba8 \
  --epochs 30 \
  --run-name "${RUN_NAME}" \
  --save-every 0 \
  > "${RUN_DIR}/console.log" 2>&1 < /dev/null &

PID=$!
echo "${PID}" > "${RUN_DIR}/train.pid"
echo "Submitted PID=${PID}; logs=${RUN_DIR}/console.log"
```

At the same seed, the Mamba model's CNN parameters are initialized identically
to the base ablation. Its zero-initialized residual gate initially preserves
the base mapping; all parameters are nevertheless placed in one Adam optimizer
and trained jointly from scratch. TensorBoard records `model/mamba_gate` each
epoch. Compare the two models at 30 epochs using validation/test WCA, the same
real-space metrics and threshold sweep, parameter count, peak GPU memory, and
wall time per optimizer step.
If the measured training-only throughput is 25000 observations in 15 minutes,
a filtered full epoch costs about 48 minutes, and 20 full epochs cost about
16 training hours, plus validation/loading/checkpoint overhead. The previous
60-short-round budget is approximately 18.63 full epochs. Actual filtered pool
size and server timings determine the real budget.

The default is **4 worker processes per loader**, with one Torch compute thread
per worker and `--prefetch-factor 2`. Workers only decompress NPZs and do a CPU
FFT. They never load the generator or initialize CUDA. Spawn isolates workers
from the training CUDA context; `persistent_workers=True` avoids restarting
them each epoch. Train and val have separate pools: 4 are active at a time,
although both pools can remain resident. Prefetch is bounded, not dataset-wide
caching; at batch 16, eight prefetched input/label batches are about 256 MiB
plus worker, decompression, FFT, active-batch, and pinned-memory overhead.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -u -m tools.benchmark_author_loader \
  --data-dir /data_ssd/oyys/high_strain_cnn/dataset \
  --num-samples 1024 --num-workers 4
```

The first pass includes process startup; the second has warm workers and may
benefit from the filesystem cache. Compare `--num-workers 0` or `2` on the
server, rather than extrapolating Windows timings. For 95,000 training samples
in 30-45 minutes, loading needs roughly 53-35 samples/s. In real training,
`data_wait` reports host time waiting for batches, including first-epoch worker
startup; it is not a CUDA-kernel idle-time profiler. Low steady-state wait is
the useful check. Time spent loading can overlap GPU training through prefetch.

## Local verification

Filtered-training verification (2026-08-31): 104 project tests passed, including
strict per-axis filtering before subset selection, missing-metadata rejection,
particle leakage checks even for excluded observations, four-worker loading and
reshuffled short training rounds with complete fixed validation. A full `reduced`
model also completed one real 64-cubed compact training/validation sample on CPU
with four loader workers, normal float32 and the filter enabled. Best/last
checkpoints reloaded strictly with 39,160,897 finite model parameters and optimizer
state. This smoke test verifies execution, not model convergence or server speed.

Parallel generation verification (Windows CPU compatibility backend, 2026-08-31):
12 actual author-generated observations covering all nine shape/phase categories
took about 13.9 seconds with one worker and 7.2 seconds with four workers, including
worker/module startup. Intensity and support arrays matched exactly; complex objects
matched at `rtol=atol=2e-6`; physical metadata matched after excluding timers. Resuming
with two workers preserved nine existing files and reproduced three missing samples'
intensities exactly. This small CPU test is not a native CUDA speedup claim.

Local verification (Windows CPU, 2026-08-30): 12 newly generated compact samples
completed the generation-to-training route, including a full reduced model
training/validation batch with four workers and normal-precision checkpoints.
On 900 compact copies of the existing reference samples (166,843,747 bytes),
batch size 16 gave these loader-only timings:

| Workers | First pass | Warm second pass | Warm samples/s |
| --- | ---: | ---: | ---: |
| 0 | 17.255 s | 18.230 s | 49.37 |
| 4 | 10.655 s | 5.020 s | 179.28 |

These include decompression, FFT, preprocessing, collation, and worker transfer,
but no GPU training/H2D copy. Warm cache and local hardware limit extrapolation.
The official H5 also ran on compact reference sample 0: standard and compact
labels both gave WCA 0.4720180035 with exactly equal model input. This is a
single-sample storage-regression check, not a new model-performance experiment.
Temporary benchmark copies and smoke-test model weights are removed afterward;
the original reference dataset/models remain untouched.

## File ownership and compatibility

- `simulation/author_generator.py`: native/compat backend selection and writers.
- `vendor/codes_for_BCDI_dataset_creation/`: unchanged author source and resources.
- `simulation/generate_author_dataset.py`: generation CLI, split schedule, manifest.
- `simulation/generation_execution.py`: deterministic jobs, bounded workers, validated reuse, atomic writes and dataset lock.
- `simulation/sample_io.py`: one shared legacy/compact phase reader.
- `pytorch_autophasenn/author_data.py`: author dataset index and source preprocessing.
- `pytorch_autophasenn/train.py`: selects dataset, worker setup, existing trainer.
- `tools/benchmark_author_loader.py`: model-free server throughput check.
- `tests/test_compact_author_data.py`: schema, label/loss, split and worker tests.
- `tests/test_generation_execution.py`: original category order, serial/spawn parity, partial resume and write safety.

Author training records go to `artifacts/training/pytorch_simulation/<run-name>`;
AutoPhaseNN records keep their existing directory. Large checkpoints keep the
existing external checkpoint root. The data manifest hash, source hashes,
backend, split counts, preprocessing, and training arguments are recorded with
each run. The reader does not rehash every NPZ at every epoch; NPZ CRC checks
detect corrupt compressed members, while dataset immutability remains required.

The shared single-sample TF/PyTorch inference and TensorFlow dataset evaluator
can read both formats. Original `tensorflow_reference/train_upstream.py` remains
untouched and still expects stored `phi`; it cannot directly train on compact
NPZs. New compact training uses the PyTorch entry point above. The existing
AutoPhaseNN memmap reader, model definitions, WCA loss, and reconstruction
physics are unchanged.
