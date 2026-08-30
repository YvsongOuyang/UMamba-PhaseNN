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
The generator currently refuses to overwrite or resume existing samples.

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
CUDA_VISIBLE_DEVICES=0 python -u -m simulation.generate_author_dataset \
  --output-dir /data_ssd/oyys/high_strain_cnn/dataset \
  --storage compact --scattering-backend pynx_cuda \
  --split-counts 95000 4000 3000 \
  --seed 20260830 --oversampling-policy record
```

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

One shape and one phase family are selected per observation, with a shape
reused for the particle's observations. Category sampling defaults to random,
not forced 3x3 balancing. Source phase-function random distributions are
unchanged. Three observations per particle is still an explicit reproduction
assumption. At a split boundary the final particle can have fewer observations;
the next split always starts a new particle seed. Train/val/test geometry is
disjoint, and the training reader checks this across the entire manifest.

## Train and measure loading

Use the existing PyTorch training environment. Example, from scratch with the
existing default reduced model and unchanged optimizer/loss/scheduler defaults:

```bash
python -u -m pytorch_autophasenn.train \
  --data-format author_npz \
  --data-dir /data_ssd/oyys/high_strain_cnn/dataset \
  --run-name author_compact_reduced_scratch \
  --num-workers 4 --save-every 0
```

No `--pretrained`/`--resume` means scratch; no `--fp16` means normal float32.
Model variants and their learning-rate defaults are unchanged. Select a
different existing variant explicitly when running an ablation.

Train/val counts come from `dataset_manifest.json`; optional
`--num-samples-train`/`--num-samples-val` limit those splits for smoke tests.
Author-data validation keeps its final partial batch. Training retains the
existing `drop_last=True` behavior; its split must contain at least one batch.

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
- `simulation/sample_io.py`: one shared legacy/compact phase reader.
- `pytorch_autophasenn/author_data.py`: author dataset index and source preprocessing.
- `pytorch_autophasenn/train.py`: selects dataset, worker setup, existing trainer.
- `tools/benchmark_author_loader.py`: model-free server throughput check.
- `tests/test_compact_author_data.py`: schema, label/loss, split and worker tests.

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
