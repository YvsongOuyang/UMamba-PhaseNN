# Project structure

This project contains related but distinct implementations and experiments.
Keeping the model backend, dataset, and artifact type visible in each path is
important because their outputs are not interchangeable.

## Source directories

```text
high_strain_CNN/
  tensorflow_reference/       Author TensorFlow reference implementation
  pytorch_autophasenn/        PyTorch port and AutoPhaseNN workflow
  simulation/                 Shared synthetic-data generation and inference
  experimental/               Official experimental-data inference only
  tools/                      Conversion, parity, and data-validation commands
  configs/                    Versioned data and simulation configuration
  requirements/               Backend-specific dependency files
  tests/                      Fast source-level tests
  artifacts/                  Training, evaluation, model, and figure outputs
```

### TensorFlow reference

`tensorflow_reference/train_upstream.py` is the vendored author training code.
It expects author-style `.npz` files containing `I` and `phi`, and it defines
the published TensorFlow architecture and WCA training objective. Keep this
file close to upstream so that it remains a trustworthy reference.

The official `model_paper.h5` belongs in `artifacts/models/`. AutoPhaseNN subsets
can now be exported by `tools/export_autophasenn_samples.py` and evaluated by the
shared official-H5 evaluator, without modifying the upstream TensorFlow file.

### PyTorch and AutoPhaseNN

`pytorch_autophasenn/` is one complete workflow:

```text
AutoPhaseNN memmaps
  -> data.py
  -> model.py + losses.py
  -> train.py
  -> checkpoint
  -> evaluate.py
  -> reconstruction.py
  -> shared AutoPhaseNN post-processing and metrics
  -> visualize.py
```

Evaluation records from this workflow are stored under
`artifacts/evaluations/autophasenn_pytorch/`; they are not evaluations of the
official TensorFlow H5 model. Official-H5 simulation results have their own
`artifacts/evaluations/simulation_tensorflow/` namespace.

### Simulation

`simulation/` is backend-neutral. It generates author-compatible `.npz`
samples and can send the same sample through either the official TensorFlow H5
model or the converted PyTorch `published` model:

```text
configs/simulation_paper.json
  -> simulation/generate_dataset.py
  -> sample_*.npz
  -> simulation/run_paper_model.py
  -> reciprocal phase, complex real-space reconstruction, metrics, figures
```

This path is also the right place to compare TensorFlow and PyTorch on exactly
the same input before introducing AutoPhaseNN-specific post-processing.

`simulation/evaluate_paper_model.py` provides the dataset-level official-H5
workflow. Balanced generation uses the full shape/phase Cartesian product; one
half calibrates support threshold and the other half reports held-out metrics.
Large predictions and figures stay ignored, while lightweight reports live in
`artifacts/evaluations/simulation_tensorflow/`.

The supplied-author-code benchmark has a generation core in
`simulation/author_generator.py`, an evaluation entry point in
`simulation/evaluate_author_code.py`, and writes to
`artifacts/evaluations/simulation_tensorflow/author_generator/`. It directly
uses the supplied atomic particle and perturbation modules. On Windows, the
unavailable PyNX `Fhkl_thread` call uses an equivalent FCC FFT for unrotated
grids and FINUFFT type-1 evaluation of the same atomic sum for rotated grids.
Both paths are verified against direct atomic summation. This route remains
distinct from both our configurable continuum simulator and the no-ground-truth
experimental-data route.

The author-code generator and evaluator default to random category selection.
Balanced categories are only an opt-in diagnostic, not a claim about the
original training distribution. The evaluator streams generation and inference,
keeps only a batch of volumes plus the current particle in memory, and flushes
the CSV after every batch. Its report includes grouped WCA, a particle-cluster
bootstrap interval, and all distribution assumptions. NPZ inputs and raw NPY
predictions remain local under the run directory; lightweight reports and a WCA
overview are trackable. It does not rank or filter samples by prediction error.

Install the extra author-source dependencies, then point the adapter at the
supplied source directory. The command follows the notebook's effective Wulff
plus random-strain path and records hashes of every supplied Python/notebook
source file:

```bash
python -m pip install -r requirements/author-simulation.txt

python -m simulation.evaluate_author_code \
  --author-code-dir D:/code/PYTHON/codes_for_BCDI_dataset_creation \
  --model artifacts/models/model_paper.h5 \
  --output-dir artifacts/evaluations/simulation_tensorflow/author_generator_notebook_seed20260829_n24 \
  --profile notebook \
  --num-samples 24 --batch-size 2 --seed 20260829 \
  --visualize-samples 3 --support-threshold 0.1 --device cpu
```

The paper-category profile treats shape and phase as two independent
categorical choices. Each observation contains exactly one shape (Wulff,
Winterbottom, or random planar cuts) and one phase family (double Gaussian,
double cosine, or Gaussian-correlated random). The two functions inside a
Gaussian/cosine field are added to each other; the three phase families are not
stacked. Reciprocal orientation and oversampling vary per observation, and
Poisson sampling is always last. With three observations per particle, the
balanced nine-sample pure-generation command covers the full Cartesian
product without loading TensorFlow:

```bash
python -m simulation.generate_author_dataset \
  --author-code-dir D:/code/PYTHON/codes_for_BCDI_dataset_creation \
  --output-dir artifacts/simulation/author_generator_author_calls_v2_seed20260829_n9 \
  --profile paper --category-sampling balanced \
  --num-samples 9 --observations-per-particle 3 \
  --seed 20260829 --save-extras
```

Use the evaluation command when the same generation run should immediately be
passed through the official H5:

```bash
python -m simulation.evaluate_author_code \
  --author-code-dir D:/code/PYTHON/codes_for_BCDI_dataset_creation \
  --model artifacts/models/model_paper.h5 \
  --output-dir artifacts/evaluations/simulation_tensorflow/author_generator_author_calls_v2_balanced_seed20260829_n9 \
  --profile paper --category-sampling balanced \
  --num-samples 9 --observations-per-particle 3 \
  --batch-size 1 --seed 20260829 --visualize-samples 3 --device cpu
```

Paper-profile NPZ files additionally contain amplitude-threshold `support`,
complex `object`, and clean `I_clean` arrays. Their metadata records the
particle/observation identity, source phase function, rotation matrix,
measured per-axis oversampling, and the source-default Poisson sampling policy
(the source does not return the sampled scale). Phase generation is tagged
`author_function_defaults_v1`: parameters are sampled inside the supplied
functions without support-span rescaling. The source does not return unwrapped
phase, so no `object_phase` truth or exact support-span statistic is fabricated.
The older `author_generator_paper_*` evaluation folders used the removed
support-span scaling and remain historical records, not source-default results.

`generator_protocol=author_calls_v2` additionally removes the old custom random
cutting implementation and geometry-dependent nstep lower bound. Source shape,
q-grid rotation, amplitude, phase and Poisson functions now run without parameter
overrides. An oversampling violation stops the run without rescaling or retries.
Earlier `source_defaults` smoke results predate these extra source-call fixes.

The default oversampling policy is `error`. To evaluate all source draws,
explicitly use `--oversampling-policy record`: retain and flag observations
whose measured oversampling is not greater than two on every axis. Their
WCA remains in the overall result; separate subset statistics are diagnostic.
The source code itself has no such rejection rule. The first strict 900-sample
attempt stopped after 12 observations and is marked by `run_status.json`;
the full rerun uses `author_generator_author_calls_v2_record_seed20260830_n900`.

The seven named ingredients split into three shape alternatives, three phase
alternatives, and rotation shared by all nine pairs. The `paper` profile enables
rotation for each observation by default; the `notebook` profile preserves its
disabled rotation. Evaluation reports include a full 3-by-3 count matrix even
when random sampling leaves some pairs empty. The balanced commands above are
branch diagnostics, not claims about the author's original category proportions.
Windows still uses the documented FFT/NUFFT compatibility backend, not native
PyNX, and in-memory coordinates instead of eight-decimal LMP serialization.

#### Source-informed reproduction candidate

`configs/simulation_paper.json` retains the original sampling profile.
`configs/simulation_reproduction_v2.json` is the first source-informed candidate:
measured post-rotation oversampling, an author-style Gaussian convolution field,
explicit provisional parameter ranges, and a consistent reciprocal-origin shift.
It still uses voxelized continuum objects and a discrete NumPy FFT, not the
unpublished atomic PyNX/LAMMPS pipeline. See `PAPER_PIPELINE_AUDIT_20260828.md`,
sections 10-11, for sources, assumptions, and first-run results.

Run from `high_strain_CNN/` with a compatible NumPy/SciPy environment; evaluation
also requires TensorFlow and `artifacts/models/model_paper.h5`:

```bash
python -m simulation.generate_dataset \
  --config configs/simulation_reproduction_v2.json \
  --output-dir artifacts/simulation/reproduction_v2_seed20260828 \
  --num-samples 72 --seed 20260828 --balanced-categories --save-extras

python -m simulation.evaluate_paper_model \
  --dataset-dir artifacts/simulation/reproduction_v2_seed20260828 \
  --output-dir artifacts/evaluations/simulation_tensorflow/reproduction_v2_seed20260828 \
  --cache-dir artifacts/simulation/tensorflow_prediction_cache/reproduction_v2_seed20260828 \
  --visualization-dir artifacts/visualizations/simulation_tensorflow/reproduction_v2_seed20260828 \
  --device cpu --batch-size 2
```

Use a new output name for each parameter experiment. The evaluator now requires
clean `object` and `support` arrays, and checks SHA-256 identities of the model,
samples, and cached predictions. Old filename-only caches need fresh inference.
Raw phase spans are measured before reciprocal recentering; the matching linear
phase ramp, effective span, centroid residual, and wrapped edge energy are recorded
separately. The WCA definition and official neural network are unchanged.

#### Paired unstrained control

Use the same generator entry point to remove real-space phase from an existing
simulation dataset, without redrawing geometry or changing individual photon
peaks. Source samples must include clean extras and a dataset manifest. The
control retains filenames (and therefore the calibration/evaluation split),
automatically saves clean extras, and records reference hashes. Do not combine
this mode with `--config` or `--balanced-categories`; all geometric parameters
come from the references. Omitting `--num-samples` uses every reference sample.

```bash
python -m simulation.generate_dataset \
  --unstrained-from artifacts/simulation/reproduction_v2_seed20260828 \
  --output-dir artifacts/simulation/reproduction_v2_unstrained_paired_seed20260828 \
  --seed 20260828

python -m simulation.evaluate_paper_model \
  --dataset-dir artifacts/simulation/reproduction_v2_unstrained_paired_seed20260828 \
  --output-dir artifacts/evaluations/simulation_tensorflow/reproduction_v2_unstrained_paired_seed20260828 \
  --cache-dir artifacts/simulation/tensorflow_prediction_cache/reproduction_v2_unstrained_paired_seed20260828 \
  --visualization-dir artifacts/visualizations/simulation_tensorflow/reproduction_v2_unstrained_paired_seed20260828 \
  --device cpu --batch-size 2
```

This is a diagnostic, not the paper's high-strain training distribution. Only
real-space phase is set to zero; reciprocal-space phase is still calculated by
the FFT. Diffraction and Poisson noise are recomputed, not copied. Fixed photon
peaks do not imply fixed total photon counts when the intensity pattern changes.
The control uses the same centering policy and records any reference-frame phase
ramp; all 72 samples in this run required zero reciprocal shift and retained
exactly zero real-space phase. Results are documented in audit section 12.

### Tools

- `convert_keras_weights.py`: TensorFlow H5 to PyTorch checkpoint.
- `export_tensorflow_reference.py`: deterministic TensorFlow tensors.
- `verify_pytorch_parity.py`: numerical comparison with converted PyTorch.
- `validate_data.py`: AutoPhaseNN memmap schema and finite-value validation.
- `export_autophasenn_samples.py`: seeded AutoPhaseNN subsets in author-loader NPZ format.

## Artifact directories

```text
artifacts/
  training/pytorch_autophasenn/         tracked lightweight run records
  evaluations/autophasenn_pytorch/      tracked PyTorch evaluation tables/logs
  evaluations/simulation_tensorflow/    tracked official-H5 simulation reports
  evaluations/autophasenn_tensorflow/   tracked official-H5 AutoPhaseNN reports
  models/                               local H5/PT weights, ignored
  parity/                               generated parity tensors, ignored
  simulation/                           generated datasets/results, ignored
  visualizations/simulation_tensorflow/ generated official-H5 figures, ignored
  visualizations/autophasenn_pytorch/   generated PyTorch figures, ignored
  visualizations/autophasenn_tensorflow/ generated official-H5 AutoPhaseNN figures
```

### Official experimental data

`experimental/run_official_data.py` consumes the author's Git-LFS files from
`matteomasto/high_strain_CNN/exp_data`. The paper identifies `data1.npy` and
`data2.npy` by their exact shapes as Particle 1 and Particle 2. Store local
copies under the ignored `artifacts/upstream_data/official_exp_data/` directory,
then run:

```bash
python -m experimental.run_official_data \
  --data-dir artifacts/upstream_data/official_exp_data \
  --model artifacts/models/model_paper.h5 \
  --output-dir artifacts/evaluations/experimental_tensorflow/official_exp_data \
  --files data1.npy data2.npy --device cpu
```

The runner centers each measured intensity at its raw-intensity center of mass,
resamples it once to `64 x 64 x 64`, and applies the official `log1p` plus
per-volume min-max preprocessing. It combines measured modulus with predicted
reciprocal phase, applies the centered inverse FFT, and re-interpolates the
complex object to the paper's original experimental grid. Since no true phase
or object is supplied, this workflow never reports WCA, support IoU, NRMSE, or
phase MAE. The support threshold is visualization-only.

Large checkpoints remain in the configured external checkpoint root. A run
manifest stores their paths so the lightweight Git record remains traceable.
Historical manifests and logs retain their original absolute server paths.

## Official H5 on AutoPhaseNN

Use the small export adapter and the existing evaluator; there is no duplicate
TensorFlow training or visualization implementation. Run from `high_strain_CNN/`:

```bash
python -m tools.export_autophasenn_samples \
  --data-dir E:/dataset/autophaseNN \
  --output-dir artifacts/simulation/autophasenn_val_subset32_seed20260828 \
  --num-samples 32 --seed 20260828

python -m simulation.evaluate_paper_model \
  --dataset-dir artifacts/simulation/autophasenn_val_subset32_seed20260828 \
  --output-dir artifacts/evaluations/autophasenn_tensorflow/official_subset32_seed20260828 \
  --cache-dir artifacts/parity/autophasenn_tensorflow/official_subset32_seed20260828 \
  --visualization-dir artifacts/visualizations/autophasenn_tensorflow/official_subset32_seed20260828 \
  --device cpu --batch-size 2
```

The export step needs the existing PyTorch data adapter; actual model inference
uses only the official TensorFlow H5. Data files are raw memmaps despite their
`.npy` suffixes; shapes, counts and dtypes come from `configs/autophasenn_data.json`.
The source files are read-only. The exporter records exact random indices and
per-sample source hashes, squares measured modulus exactly once, and adds no
noise or peak rescaling. Particle phase is retained, not set to zero.

The existing fractional amplitude-COM phase-ramp correction is applied to the
clean Fourier field. Clean real-space truth is reconstructed from its clean
modulus and corrected phase, not from noisy measurements. Ground-truth support
is `abs(centered clean object) > 0.1 * max`, independent of the selected prediction
threshold. Fourier sub-voxel centering can introduce edge ringing; its convention
is recorded and is distinct from the integer-only simulation centering.

Default evaluation uses the first 16 exported samples for threshold calibration
and the other 16 for metrics. Ordering follows the seeded random draw, not source
index sorting. The cached float32 prediction tensor lives in the local parity
artifact namespace for this run; it is not itself a TensorFlow/PyTorch parity
test. Use `--reuse-predictions` only with the matching dataset/model/cache.

This is an out-of-distribution test of the official model, not an evaluation of
the AutoPhaseNN-trained PyTorch variant. See audit section 13 for the initial
subset results and limitations.

## Recheck the historical 0.5694 category

The generator can fix a shape and phase family while drawing fresh sample-level
parameters from an existing config. This reproduces the historical sample's
Wulff plus double-Gaussian category without duplicating the particle:

```bash
python -m simulation.generate_dataset \
  --config configs/simulation_paper.json \
  --output-dir artifacts/simulation/paper_config_wulff_double_gaussian_seed20260829 \
  --num-samples 24 --seed 20260829 \
  --shape-type wulff --phase-type double_gaussian --save-extras

python -m simulation.evaluate_paper_model \
  --dataset-dir artifacts/simulation/paper_config_wulff_double_gaussian_seed20260829 \
  --output-dir artifacts/evaluations/simulation_tensorflow/paper_config_wulff_double_gaussian_seed20260829 \
  --cache-dir artifacts/simulation/tensorflow_prediction_cache/paper_config_wulff_double_gaussian_seed20260829 \
  --visualization-dir artifacts/visualizations/simulation_tensorflow/paper_config_wulff_double_gaussian_seed20260829 \
  --device cpu --batch-size 2 --visualize-samples 3
```

The first half is used only to calibrate the support threshold; metrics use the
held-out second half. Phase WCA is independent of the support threshold. See
audit section 14 and `reference_sample_comparison.json` for the reference-sample
comparison and validation against the unchanged official TensorFlow loss.

## Naming rule

Use `<dataset>_<backend>` in artifact namespaces, and include the model variant
and important evaluation condition in the run name. Do not place generated
weights, datasets, or images beside source files.
