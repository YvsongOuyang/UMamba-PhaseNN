# high_strain_CNN

A supervised 3D CNN for reciprocal-space phase retrieval from highly strained
Bragg coherent diffraction patterns. This checkout keeps the original
TensorFlow 2.10.1 implementation and adds a numerically verified PyTorch port,
a resource-reduced PyTorch training variant, and an AutoPhaseNN memmap adapter.

The model predicts the **reciprocal-space phase**. It does not directly predict
the real-space amplitude and phase used by AutoPhaseNN. Combining the measured
diffraction modulus with the predicted reciprocal phase and applying an inverse
FFT gives a real-space initialization that can be refined by an iterative phase
retrieval algorithm.

## Implementations

- `tensorflow_reference/`: original TensorFlow training code from upstream and
  the home for future official-model AutoPhaseNN adapters.
- `pytorch_autophasenn/model.py`: reduced and published
  `HighStrainPhaseUNet` variants.
- `pytorch_autophasenn/losses.py`: equivalent weighted circular-average (WCA) loss,
  including global-phase and conjugate/twin ambiguity handling.
- `pytorch_autophasenn/data.py`: AutoPhaseNN raw-memmap adapter.
- `pytorch_autophasenn/reconstruction.py`: combines measured diffraction modulus and
  predicted reciprocal phase, then reconstructs a complex real-space object.
- `pytorch_autophasenn/train.py`, `evaluate.py`, and `visualize.py`: the full
  PyTorch workflow on AutoPhaseNN data.
- `simulation/`: configurable paper-style Wulff/Winterbottom/random-particle
  simulation, official TensorFlow inference, reconstruction, and 2D/3D plots.
- `tools/`: H5 conversion, TensorFlow/PyTorch parity, and data validation.
- `artifacts/`: backend- and dataset-labelled experiment records and outputs.

See [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) for the dependency
graph, artifact policy, and planned TensorFlow evaluation layout.

## PyTorch architecture

`HighStrainPhaseUNet` supports three variants:

| Variant | Encoder scales | Bottleneck | Parameters | Purpose |
|---|---:|---:|---:|---|
| `reduced` | 5 | 1024 | 39,160,897 | Default AutoPhaseNN-data training |
| `reduced_bn_no_outer_skip` | 5 | 1024 | 39,121,665 | BatchNorm and no full-resolution skip ablation |
| `published` | 6 | 2048 | 143,759,937 | Original-weight conversion and parity |

The reduced model removes the complete deepest encoder-decoder scale. Its
bottom path is `4^3 x 512 -> pool -> 2^3 x 512 -> 2^3 x 1024 -> 4^3 x 512`.
The result is concatenated with the compressed `4^3 x 256` skip, preserving the
published decoder input of `4^3 x 768` from that point onward.

All variants retain the two multi-dilation input blocks, compressed U-Net
skips, LeakyReLU slope 0.2, TensorFlow `SAME` padding, transpose-convolution
voxel alignment, and one-channel `64 x 64 x 64` reciprocal-phase output.

The `reduced_bn_no_outer_skip` variant is based only on `reduced`. It removes
the outermost `64^3` encoder-to-decoder skip, including its dedicated
`conv3d_12` compressor, so `conv3d_19` receives the decoder's 32 channels
directly instead of a 48-channel concatenation. Every remaining hidden
convolution and transpose convolution follows
`Conv/ConvTranspose -> BatchNorm3d -> LeakyReLU`; the final one-channel
`conv3d_20` output is intentionally left unnormalized. BatchNorm uses
`eps=1e-3` and PyTorch `momentum=0.01`, equivalent to Keras running-statistics
momentum `0.99`. The data pipeline, WCA loss, output semantics, and real-space
reconstruction are unchanged.

PyTorch tensors use `NCDHW`; TensorFlow tensors use `NDHWC`. This layout change
does not change the logical tensor dimensions.

## Install

The training port supports Python 3.10 and PyTorch 2.x:

```bash
python -m pip install -r requirements/pytorch.txt
```

For the exact environment used by the numerical parity and end-to-end smoke
tests, install the lock file in a clean Python 3.10 environment:

```bash
python -m pip install -r requirements/pytorch-lock.txt
```

TensorFlow is needed only to regenerate the numerical reference. Because the
published model uses TensorFlow 2.10.1, use a separate Python 3.10 environment:

```bash
python -m pip install -r requirements/tensorflow-parity.txt
```

## Version and data management

`VERSION` contains the PyTorch port version. The shared
`configs/autophasenn_data.json` file is the source of truth for:

- dataset and schema version;
- root directory and split filenames;
- sample counts, volume shape, and dtypes;
- diffraction/real-space semantics;
- input and target preprocessing.

Training and evaluation load their defaults from this file while still
allowing explicit command-line overrides. Validate the external data before a
new training run:

```bash
python -u -m tools.validate_data \
  --output artifacts/training/pytorch_autophasenn/data_validation.json
```

This checks the exact expected byte size of all four raw memmaps and samples
the first, middle, and last regions for finite values. Diffraction samples are
also checked for nonnegative modulus. Add `--sha256` only when a full content
fingerprint is required; hashing roughly 90 GB of data can take several
minutes.

Every training run writes:

```text
artifacts/training/pytorch_autophasenn/<run-name>/config.json
artifacts/training/pytorch_autophasenn/<run-name>/run_manifest.json
artifacts/training/pytorch_autophasenn/<run-name>/history.json
artifacts/training/pytorch_autophasenn/<run-name>/tensorboard/
```

`run_manifest.json` records the project version, Git commit/dirty state, Python, NumPy,
PyTorch, CUDA, cuDNN, GPU, resolved data paths, expected and actual file sizes,
data version, and all training arguments. The same manifest is embedded in
every checkpoint. Evaluation results record both the checkpoint version and
the current evaluator version and warn when they differ.

## Convert the published weights

Place the official `model_paper.h5` under `artifacts/models/`, then run:

```bash
python -m tools.convert_keras_weights
```

The converter validates the complete set of 27 parameterized layers before it
writes the checkpoint.

## Numerical equivalence

Generate a deterministic TensorFlow reference:

```bash
python -m tools.export_tensorflow_reference \
  --output artifacts/parity/tensorflow_layers.npz \
  --include-intermediates
```

Compare every parameterized layer and the final output:

```bash
python -m tools.verify_pytorch_parity \
  --reference artifacts/parity/tensorflow_layers.npz \
  --max-abs-tolerance 1e-3
```

The checked-in port was validated on the same deterministic `64^3` input with:

- final maximum absolute error: `1.1253357e-4`;
- final mean absolute error: `1.2890492e-5`;
- final relative maximum error: `3.1677958e-6`;
- all 27 parameterized layers below the `1e-3` threshold;
- TensorFlow/PyTorch WCA loss maximum absolute error: `0` on a separate random
  three-sample comparison.

These checks establish numerical equivalence of the `published` variant within
normal float32 backend differences. They do not apply to the intentionally
smaller `reduced` variant. Training a new model on a different dataset cannot
guarantee identical paper metrics because the data distribution, hardware
kernels, shuffling, and optimization trajectory are different.

## Paper-style simulated data

The article states that its synthetic objects combine Wulff, Winterbottom, and
randomly plane-cut particle shapes with one of three real-space phase fields:
two Gaussians, two cosines, or a 3D Gaussian-correlated random profile. It also
specifies a `64 x 64 x 64` reciprocal grid, oversampling above two, random
orientation, phase variation from `2pi` to `5pi`, and Poisson noise. The public
repository does **not** include the PyNX simulation program or the complete
sampling distributions used for those variables.

`configs/simulation_paper.json` therefore separates the paper-fixed conditions
from configurable replication assumptions. The NumPy FFT evaluates the same
kinematic far-field transform on a regular cubic grid; it is not presented as
the missing original PyNX source. Every sample records its exact random shape,
phase, rotation, photon level, centering shift, and assumption notes in
`metadata_json`, while retaining the official loader's required keys:

```text
I       float32 [64, 64, 64]  noisy linear diffraction intensity
phi     float32 [64, 64, 64]  wrapped reciprocal-space phase
```

Generate a small full-factorial diagnostic set with all optional truth arrays.
`--balanced-categories` cycles through every one of the three-shape by
three-phase combinations:

```bash
python -m simulation.generate_dataset \
  --config configs/simulation_paper.json \
  --output-dir artifacts/simulation/dataset \
  --num-samples 9 \
  --balanced-categories \
  --save-extras
```

Omit `--save-extras` when generating a larger training set. Those lean files
still work directly with the original `load_and_process_file()` function in
`tensorflow_reference/train_upstream.py`; extras are only needed for direct real-space accuracy analysis and
visualization.

For the strongest fidelity check, run the author's TensorFlow 2.10.1 H5 model
rather than the converted network. The command saves the exact NDHWC input,
raw TensorFlow phase, twin-aligned phase, reconstructed complex object, metrics,
and both center-slice and 3D figures:

```bash
python -m simulation.run_paper_model \
  --sample artifacts/simulation/dataset/sample_00000.npz \
  --backend tensorflow \
  --model artifacts/models/model_paper.h5 \
  --output-dir artifacts/simulation/official_tensorflow
```

The runner applies the documented `log1p` and per-volume min-max input
preprocessing. It reconstructs with
`sqrt(I) * exp(1j * predicted_phase)` followed by the centered inverse FFT. For
simulation evaluation only, the known target phase selects the direct or
conjugate/twin-equivalent solution before real-space metrics are computed; the
raw model output is always saved separately.

The same entry point can consume the converted checkpoint:

```bash
python -m simulation.run_paper_model \
  --sample artifacts/simulation/dataset/sample_00000.npz \
  --backend pytorch \
  --model artifacts/models/model_paper_pytorch.pt \
  --device cuda \
  --output-dir artifacts/simulation/converted_pytorch
```

The published TensorFlow model contains `143,759,937` trainable parameters. A
numerical test of the simulator is available without loading that large model:

```bash
python -m unittest tests.test_simulation
```

### Evaluate the official model on reproduced simulations

For threshold calibration and held-out evaluation, generate 72 samples. This
gives 36 calibration and 36 evaluation samples, with every shape/phase pair
appearing exactly four times in each half:

```bash
python -m simulation.generate_dataset \
  --output-dir artifacts/simulation/paper_evaluation \
  --num-samples 72 \
  --seed 20260828 \
  --balanced-categories \
  --save-extras \
  --overwrite
```

Run the official TensorFlow H5 once and perform the broad support-threshold
scan. Model predictions are cached under the ignored `artifacts/simulation/`
tree so later threshold scans do not repeat inference:

```bash
python -m simulation.evaluate_paper_model \
  --dataset-dir artifacts/simulation/paper_evaluation \
  --batch-size 1
```

Refine the near-optimal interval while reusing those predictions:

```bash
python -m simulation.evaluate_paper_model \
  --dataset-dir artifacts/simulation/paper_evaluation \
  --output-dir artifacts/evaluations/simulation_tensorflow/official_published_calibrated \
  --visualization-dir artifacts/visualizations/simulation_tensorflow/official_published_calibrated \
  --thresholds 0.125 0.13 0.135 0.14 0.145 0.15 0.155 0.16 0.165 0.17 0.175 \
  --batch-size 1 \
  --reuse-predictions
```

Threshold selection uses only the calibration half. It first identifies the
IoU plateau within `0.001` of the highest mean support IoU, then selects the
threshold whose predicted/true support-volume ratio is closest to one. The
reported final metrics use only the held-out half and the exact boolean support
saved by the simulator. The evaluator writes JSON, CSV, Markdown, and log files
under `artifacts/evaluations/simulation_tensorflow/` and creates representative
2D/3D figures under `artifacts/visualizations/simulation_tensorflow/`.

## AutoPhaseNN data mapping

The PyTorch entry point uses the same raw memmap files as
`autophasenn_training_pipeline`:

```text
/data_ssd/oyys/autophasenn/
  train_diff.npy   float32 diffraction modulus, [25000, 64, 64, 64]
  train_real.npy   complex64 real-space object, [25000, 64, 64, 64]
  val_diff.npy     float32 diffraction modulus, [5000, 64, 64, 64]
  val_real.npy     complex64 real-space object, [5000, 64, 64, 64]
```

Despite the `.npy` suffix, these files are opened as raw `numpy.memmap` arrays,
matching the existing AutoPhaseNN loader. For each sample the adapter performs:

1. square the stored diffraction modulus to recover intensity;
2. apply `log1p` by default;
3. min-max normalize the volume to `[0, 1]`;
4. measure the real-space amplitude center of mass and remove its corresponding
   linear reciprocal-space phase ramp;
5. Fourier-transform the complex real-space object with the AutoPhaseNN shift
   convention;
6. subtract the reciprocal center phase and use the result as the target;
7. optimize the original WCA loss with the normalized input as its weights.

This is the direct semantic mapping between the two datasets: the original
repository stores intensity and reciprocal phase, while AutoPhaseNN stores
diffraction modulus and the corresponding complex real-space object.

## Train on AutoPhaseNN data

The defaults already point to the paths and sample counts above. Training uses
the 39.2M-parameter `reduced` variant unless explicitly overridden:

```bash
python -u -m pytorch_autophasenn.train --run-name high_strain_autophase_scratch
```

Train the BatchNorm/no-outer-skip ablation from scratch with the otherwise
unchanged defaults. This variant starts at `1e-3`; the existing variants keep
their `1e-4` default:

```bash
python -u -m pytorch_autophasenn.train \
  --model-variant reduced_bn_no_outer_skip \
  --run-name high_strain_reduced_bn_no_outer_skip_scratch
```

Background training with a timestamped, self-describing run directory:

```bash
RUN_NAME="high_strain_reduced_centered_scratch_bs16_lr1e-4_plateau_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$PWD/artifacts/training/pytorch_autophasenn/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

nohup env CUDA_VISIBLE_DEVICES=0 python -u -m pytorch_autophasenn.train \
  --run-name "${RUN_NAME}" \
  --epochs 240 \
  > "${RUN_DIR}/console.log" 2>&1 < /dev/null &

PID=$!
echo "${PID}" > "${RUN_DIR}/train.pid"
```

Fine-tune the converted published weights:

```bash
python -u -m pytorch_autophasenn.train \
  --model-variant published \
  --run-name high_strain_autophase_finetune \
  --pretrained artifacts/models/model_paper_pytorch.pt
```

Published checkpoints cannot be loaded into the reduced model because the
deepest tensor shapes differ. Evaluation detects the variant automatically
from the checkpoint.

Training keeps the original Adam settings (`beta1=0.9`, `beta2=0.999`,
`epsilon=1e-7`), float32, batch size 16, and the WCA objective. The `reduced`
and `published` variants start at the paper's `1e-4`; the BatchNorm ablation
starts at `1e-3` to test whether normalization supports faster optimization.
All variants use the AutoPhaseNN `ReduceLROnPlateau` defaults: factor `0.5`,
patience `5`, and minimum learning rate `1e-6`. The scheduler is the only
learning-rate-policy departure from the paper's constant rate. The adapted
reduced-model training horizon defaults to 240 epochs; the published upstream
training used 60 epochs. Reduce the batch size if the target GPU runs out of
memory.

Small run records and TensorBoard events are written under
`artifacts/training/pytorch_autophasenn/<run-name>`.
Large checkpoints are written under:

```text
/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/high_strain_cnn/<run-name>
```

Batch progress logs report elapsed time and ETA for the current train or
validation stage. Epoch summaries report epoch time, total run time, remaining
training ETA, and the estimated local completion timestamp.

Use `--resume <checkpoint>` to restore the model, optimizer, scaler, history,
and epoch state.

## Reconstruct and compare with AutoPhaseNN

The phase network predicts only reciprocal-space phase. The evaluator combines
that prediction with the original, unnormalized measured diffraction modulus:

```text
predicted_spectrum = measured_modulus * exp(1j * predicted_phase)
predicted_object = fftshift(ifftn(ifftshift(predicted_spectrum)))
```

It then extracts real-space amplitude, phase, and support and directly reuses
the official AutoPhaseNN phase unwrapping, phase-offset removal, center-of-mass
alignment, support threshold, and metric functions. The real-space amplitude,
phase, SSIM, and support metrics therefore use the same scale and definitions
as `autophasenn_training_pipeline/evaluate.py`.

Evaluate a trained checkpoint on the full validation set:

```bash
python -u -m pytorch_autophasenn.evaluate \
  --checkpoint /data_ssd/oyys/autophasenn/autophasenn_pipeline_output/high_strain_cnn/your_run/checkpoint_best.pt \
  --data-dir /data_ssd/oyys/autophasenn
```

Run the planned support-threshold diagnostic in the same evaluation pass:

```bash
python -u -m pytorch_autophasenn.evaluate \
  --checkpoint /data_ssd/oyys/autophasenn/autophasenn_pipeline_output/high_strain_cnn/your_run/checkpoint_best.pt \
  --threshold 0.3 \
  --threshold-sweep 0.05 0.1 0.15 0.2 0.25 0.3 0.35
```

The evaluator unwraps each phase volume only once, then applies every requested
threshold. The primary `--threshold` value remains the headline result for
that model. The sweep reports the threshold with the best mean
support IoU and the threshold whose mean support-volume ratio is closest to one;
both are validation diagnostics rather than replacements for the primary result.

Unless `--output-dir` is supplied, results are stored like the AutoPhaseNN
pipeline under
`artifacts/evaluations/autophasenn_pytorch/evaluate_<model-variant>/`.

The standard result files match the AutoPhaseNN evaluation bundle:

```text
evaluation_results.json
evaluation_samples.csv
evaluation_summary.md
evaluation.log
```

When `--threshold-sweep` is enabled, it additionally writes:

```text
evaluation_threshold_sweep.csv
evaluation_threshold_sweep_samples.csv
```

With the two save flags, the evaluator also writes raw memmaps compatible with
the AutoPhaseNN storage convention:

```text
predicted_realspace.npy          complex64 [N, 64, 64, 64]
predicted_reciprocal_phase.npy   float32   [N, 64, 64, 64]
```

For 5000 samples these optional files require about 9.77 GiB and 4.88 GiB,
respectively. Omit the save flags when only aggregate and per-sample metrics are
needed.

The published WCA loss explicitly treats reciprocal phase and its negation as
a conjugate/twin ambiguity. Evaluation defaults to `--ambiguity-mode
twin_aligned`, which uses the ground truth only to select the physically
equivalent sign before computing real-space metrics. This produces the fairest
quality comparison but is an evaluation-only oracle operation. Use:

```bash
python -u -m pytorch_autophasenn.evaluate \
  --checkpoint path/to/checkpoint_best.pt \
  --ambiguity-mode raw
```

to evaluate the uncorrected network output.

Reciprocal-space modulus metrics are expected to be nearly zero for this model:
the reconstruction reuses the measured modulus by construction. They confirm
FFT consistency but do not measure learned quality. Compare `phase_wca` and the
real-space amplitude, phase, SSIM, and support metrics when judging this model
against AutoPhaseNN.

## Visualize reconstructed objects

The visualization entry point uses the same post-processing and result naming
convention as AutoPhaseNN:

```bash
python -u -m pytorch_autophasenn.visualize \
  --checkpoint /data_ssd/oyys/autophasenn/autophasenn_pipeline_output/high_strain_cnn/your_run/checkpoint_best.pt \
  --data-dir /data_ssd/oyys/autophasenn \
  --num-samples 3
```

Unless overridden, figures and metadata are written to
`artifacts/visualizations/autophasenn_pytorch/vision_<model-variant>/`. The 2D overview separates learned reciprocal
phase quality from real-space reconstruction quality. The reprojected modulus
panels are labeled explicitly because they reuse the measured modulus and are
only an FFT consistency check.

The default run writes the same complete ten-image bundle as AutoPhaseNN:

```text
visualization_2d.png
visualization_3d.png
visualization_shift_comparison_3d.png
visualization_error_3d.png
visualization_reciprocal_2d.png
visualization_reciprocal_3d.png
visualization_amplitude_3d.png
visualization_phase_3d.png
visualization_diffraction_3d.png
visualization_diffraction_phase_3d.png
```

The four five-panel 3D volumes compare target, reconstruction before center
shift, reconstruction after center shift, the shift difference, and the final
target difference. Each output can still be disabled explicitly by passing
`none` to its corresponding `--output-...` option.

## Original project

The upstream project and pretrained model are available at
<https://github.com/matteomasto/high_strain_CNN>.

This source tree was vendored from upstream commit `43d31f3`. Large upstream
Git LFS data and model files are intentionally excluded from the parent
UMamba-PhaseNN repository and can be downloaded from the upstream project when
needed.
