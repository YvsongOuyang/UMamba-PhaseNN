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

- `train.py`: original TensorFlow training code from the upstream repository.
- `pytorch_port/model.py`: reduced and published `HighStrainPhaseUNet` variants.
- `pytorch_port/losses.py`: equivalent weighted circular-average (WCA) loss,
  including global-phase and conjugate/twin ambiguity handling.
- `pytorch_port/data.py`: AutoPhaseNN raw-memmap adapter.
- `pytorch_port/reconstruction.py`: combines measured diffraction modulus and
  predicted reciprocal phase, then reconstructs a complex real-space object.
- `train_pytorch.py`: PyTorch training entry point for the AutoPhaseNN files.
- `evaluate_autophase.py`: reconstructs real space and evaluates with the same
  post-processing and metric functions as `autophasenn_training_pipeline`.
- `visualize_postprocessed.py`: creates matching reciprocal-space and
  real-space 2D/3D diagnostic figures from a HighStrain checkpoint.
- `convert_keras_weights.py`: converts `model_paper.h5` to a standard PyTorch
  checkpoint.
- `export_tensorflow_reference.py` and `verify_pytorch_parity.py`: reproducible
  TensorFlow/PyTorch numerical parity check.

## PyTorch architecture

`HighStrainPhaseUNet` supports two variants:

| Variant | Encoder scales | Bottleneck | Parameters | Purpose |
|---|---:|---:|---:|---|
| `reduced` | 5 | 1024 | 39,160,897 | Default AutoPhaseNN-data training |
| `published` | 6 | 2048 | 143,759,937 | Original-weight conversion and parity |

The reduced model removes the complete deepest encoder-decoder scale. Its
bottom path is `4^3 x 512 -> pool -> 2^3 x 512 -> 2^3 x 1024 -> 4^3 x 512`.
The result is concatenated with the compressed `4^3 x 256` skip, preserving the
published decoder input of `4^3 x 768` from that point onward.

Both variants retain the two multi-dilation input blocks, compressed U-Net
skips, LeakyReLU slope 0.2, TensorFlow `SAME` padding, transpose-convolution
voxel alignment, and one-channel `64 x 64 x 64` reciprocal-phase output.

PyTorch tensors use `NCDHW`; TensorFlow tensors use `NDHWC`. This layout change
does not change the logical tensor dimensions.

## Install

The training port supports Python 3.10 and PyTorch 2.x:

```bash
python -m pip install -r requirements-pytorch.txt
```

For the exact environment used by the numerical parity and end-to-end smoke
tests, install the lock file in a clean Python 3.10 environment:

```bash
python -m pip install -r requirements-pytorch-lock.txt
```

TensorFlow is needed only to regenerate the numerical reference. Because the
published model uses TensorFlow 2.10.1, use a separate Python 3.10 environment:

```bash
python -m pip install -r requirements-tensorflow-parity.txt
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
python -u validate_data.py \
  --output runs/data_validation.json
```

This checks the exact expected byte size of all four raw memmaps and samples
the first, middle, and last regions for finite values. Diffraction samples are
also checked for nonnegative modulus. Add `--sha256` only when a full content
fingerprint is required; hashing roughly 90 GB of data can take several
minutes.

Every training run writes:

```text
runs/<run-name>/config.json
runs/<run-name>/run_manifest.json
runs/<run-name>/history.json
runs/<run-name>/tensorboard/
```

`run_manifest.json` records the project version, Git commit/dirty state, Python, NumPy,
PyTorch, CUDA, cuDNN, GPU, resolved data paths, expected and actual file sizes,
data version, and all training arguments. The same manifest is embedded in
every checkpoint. Evaluation results record both the checkpoint version and
the current evaluator version and warn when they differ.

## Convert the published weights

`model_paper.h5` is managed by Git LFS. After cloning, make sure it is a real
575 MB H5 file rather than a small LFS pointer, then run:

```bash
python convert_keras_weights.py \
  --keras-h5 model_paper.h5 \
  --output model_paper_pytorch.pt
```

The converter validates the complete set of 27 parameterized layers before it
writes the checkpoint.

## Numerical equivalence

Generate a deterministic TensorFlow reference:

```bash
python export_tensorflow_reference.py \
  --keras-h5 model_paper.h5 \
  --output parity_output/tensorflow_layers.npz \
  --include-intermediates
```

Compare every parameterized layer and the final output:

```bash
python verify_pytorch_parity.py \
  --checkpoint model_paper_pytorch.pt \
  --reference parity_output/tensorflow_layers.npz \
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
4. Fourier-transform the complex real-space object with the AutoPhaseNN shift
   convention;
5. use the centered reciprocal phase as the target;
6. optimize the original WCA loss with the normalized input as its weights.

This is the direct semantic mapping between the two datasets: the original
repository stores intensity and reciprocal phase, while AutoPhaseNN stores
diffraction modulus and the corresponding complex real-space object.

## Train on AutoPhaseNN data

The defaults already point to the paths and sample counts above. Training uses
the 39.2M-parameter `reduced` variant unless explicitly overridden:

```bash
python -u train_pytorch.py --run-name high_strain_autophase_scratch
```

Background training with a timestamped, self-describing run directory:

```bash
RUN_NAME="high_strain_reduced_scratch_bs16_lr1e-4_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$PWD/runs/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

nohup env CUDA_VISIBLE_DEVICES=0 python -u train_pytorch.py \
  --run-name "${RUN_NAME}" \
  > "${RUN_DIR}/console.log" 2>&1 < /dev/null &

PID=$!
echo "${PID}" > "${RUN_DIR}/train.pid"
```

Fine-tune the converted published weights:

```bash
python -u train_pytorch.py \
  --model-variant published \
  --run-name high_strain_autophase_finetune \
  --pretrained model_paper_pytorch.pt
```

Published checkpoints cannot be loaded into the reduced model because the
deepest tensor shapes differ. Evaluation detects the variant automatically
from the checkpoint.

Important defaults matching the original recipe are 60 epochs, Adam with
learning rate `1e-4`, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-7`, constant
learning rate, float32, batch size 16, and the WCA objective. Reduce the batch
size if the target GPU runs out of memory. Use `--lr-scheduler plateau` only for
an intentional departure from the original recipe.

Small run records and TensorBoard events are written under `runs/<run-name>`.
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
python -u evaluate_autophase.py \
  --checkpoint /data_ssd/oyys/autophasenn/autophasenn_pipeline_output/high_strain_cnn/your_run/checkpoint_best.pt \
  --data-dir /data_ssd/oyys/autophasenn
```

Unless `--output-dir` is supplied, results are stored like the AutoPhaseNN
pipeline under `evaluate/evaluate_<model-variant>/`, for example
`evaluate/evaluate_reduced/`.

The standard result files match the AutoPhaseNN evaluation bundle:

```text
evaluation_results.json
evaluation_samples.csv
evaluation_summary.md
evaluation.log
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
python -u evaluate_autophase.py \
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
python -u visualize_postprocessed.py \
  --checkpoint /data_ssd/oyys/autophasenn/autophasenn_pipeline_output/high_strain_cnn/your_run/checkpoint_best.pt \
  --data-dir /data_ssd/oyys/autophasenn \
  --num-samples 3
```

Unless overridden, figures and metadata are written to
`vision/vision_<model-variant>/`. The 2D overview separates learned reciprocal
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
