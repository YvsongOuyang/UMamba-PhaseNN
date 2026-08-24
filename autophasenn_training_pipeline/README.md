# AutoPhaseNN PyTorch Training Pipeline

This folder is a cleaned, standalone PyTorch training/evaluation flow for the
TF-compatible AutoPhaseNN model that can load the converted checkpoint:

```text
PyTorch/cohere-trained_model_tf_compatible.pth
```

## Files

```text
dataset.py                  Memmap data loading with pipeline preprocessing
losses.py                   Project loss functions and scale alignment
model_tf_compatible.py      PyTorch model matching the converted TF2 checkpoint
model_residual.py           ResidualAutoPhaseNN architecture variant
model_amplitude_skip.py     AmplitudeSkipAutoPhaseNN architecture variant
model_decoder_cross_skip.py DecoderCrossSkipAutoPhaseNN architecture variant
model_decoder_cross_concat.py DecoderCrossConcatAutoPhaseNN architecture variant
model_mamba_skip.py         AutoPhaseNNBiPVMSkip architecture variant
model_factory.py            Model selection and pretrained initialization
train.py                    Full train / fine-tune entry point
evaluate.py                 Checkpoint evaluation and loss report
visualize_postprocessed.py  TF test_network_unsup-style visualization
```

## ResidualAutoPhaseNN Variant

`ResidualAutoPhaseNN` preserves the baseline six-output forward contract,
amplitude/phase decoder channel sequences, and four encoder max-pooling
operations. Every two-convolution encoder, bottleneck, and decoder module is
replaced by a 3D residual block; a `1 x 1 x 1` projection is used on the shortcut
when its input and output channel counts differ.

Select it in the existing training, evaluation, and visualization workflows
with:

```text
--model-variant residual
```

Residual checkpoints are structurally different from TF-compatible baseline
checkpoints. Start a new residual run with `--from-scratch`, and pass
`--model-variant residual` again when resuming, evaluating, or visualizing it.

```bash
python autophasenn_training_pipeline/train.py \
  --model-variant residual \
  --from-scratch
```

## Amplitude Skip Variant

`AmplitudeSkipAutoPhaseNN` adds standard U-Net concatenation skips only at
`8 x 8 x 8` and `16 x 16 x 16`. Each pre-pooling encoder feature is passed to
the amplitude decoder at the matching scale. The phase decoder remains exactly
the baseline path and receives only the shared bottleneck. All output
activations, support construction, forward physics, and losses remain unchanged.

Initialize it from a baseline checkpoint with:

```bash
python autophasenn_training_pipeline/train.py \
  --model-variant amplitude_skip \
  --pretrained /path/to/baseline/checkpoint_best.pt
```

The two expanded amplitude convolutions copy their baseline channels exactly
and initialize only the added skip-channel kernels to zero. The phase decoder
parameters are copied without any shape change. `--pretrained` starts a fresh
optimizer for fine-tuning. Use `--resume` only with a checkpoint already created
by the `amplitude_skip` variant.

For a fair quick comparison, fine-tune both `baseline` and `amplitude_skip` from
the same baseline checkpoint with identical data order, seed, optimizer,
learning rate, scheduler, and epoch count. Evaluate each resulting best
checkpoint with the same settings and its matching `--model-variant`.

## Decoder Cross-Skip Variant

`DecoderCrossSkipAutoPhaseNN` keeps the baseline encoder and both decoder
blocks unchanged, then exchanges amplitude and phase decoder features after the
`8 x 8 x 8` and `16 x 16 x 16` blocks. Each exchange uses independent
`1 x 1 x 1` convolutions and simultaneous bidirectional residual updates:

```text
A'   = A   + alpha_amp   * phase_to_amp(Phi)
Phi' = Phi + alpha_phase * amp_to_phase(A)
```

Both updates use the pre-update `A` and `Phi`. All four scalar `alpha`
parameters start at zero, so loading a baseline checkpoint initially reproduces
the baseline outputs exactly. No gate, attention, activation, additional loss,
or intermediate physics supervision is added.

The optional staged fine-tuning schedule is controlled by two epoch counts:

```bash
python autophasenn_training_pipeline/train.py \
  --model-variant decoder_cross_skip \
  --pretrained /path/to/baseline/checkpoint_best.pt \
  --cross-skip-only-epochs 5 \
  --decoder-finetune-epochs 10 \
  --epochs 100 \
  --lr 1e-4
```

This trains only the two cross-skip modules in epochs 1-5, both decoders plus
the cross-skips in epochs 6-15, and the full network from epoch 16 onward.
Frozen BatchNorm running statistics remain fixed. Set both stage counts to zero
to fine-tune the full network immediately. The four learned `alpha` values are
written to the console and TensorBoard after every epoch.

## Decoder Cross-Concat Variant

`DecoderCrossConcatAutoPhaseNN` is built directly from the baseline and adds
conventional channel concatenation only between the amplitude and phase
decoders at `8 x 8 x 8` and `16 x 16 x 16`. The encoder has no skip connection.
At each scale, both branch inputs are built simultaneously:

```text
amplitude path: concat(A, Phi)
phase path:     concat(Phi, A)
```

The concatenated tensors enter the next original decoder block. Only that
block's first `3 x 3 x 3` convolution is widened; baseline branch kernels are
copied exactly and new cross-channel kernels start at zero. The initialized
model therefore reproduces baseline outputs without `alpha`, extra fusion
blocks, attention, gates, or new losses. Fine-tune the full model directly at a
low learning rate.

## Bi-PVM Mamba Skip Variant

`AutoPhaseNNBiPVMSkip` preserves the baseline encoder, bottleneck, both decoder
paths, output activations, support construction, forward physics, and training
loss. It adds independent encoder-to-decoder Bi-PVM bridges at only the
`8 x 8 x 8` and `16 x 16 x 16` scales. Amplitude and phase use four completely
independent bridges and fusion blocks.

Each bridge projects its encoder feature to 32 channels, applies residual
depthwise 3D local mixing, splits the flattened sequence into four 8-channel
groups, and applies independent forward and reverse Mamba operators per group.
The bridge output is concatenated with the matching decoder feature and fused
by one `3 x 3 x 3` convolution with the baseline LeakyReLU and BatchNorm
configuration. The Mamba parameters are fixed to `d_model=8`, `d_state=16`,
`d_conv=4`, and `expand=2` for this experiment.

The variant uses the repository's existing `mamba-ssm` dependency and is
selected with `--model-variant mamba_skip`. Train it from random initialization
with the same configuration as the baseline comparison:

```bash
python autophasenn_training_pipeline/train.py \
  --model-variant mamba_skip \
  --from-scratch
```

Training runs are kept inside this subproject by default. If `--run-name` is
omitted, the script builds one from the timestamp, model variant,
initialization, loss, batch size, learning rate, optimizer, scheduler,
threshold, and seed:

```text
autophasenn_training_pipeline/runs/<run-name>/
  config.json
  run_info.json
  history.json
  tensorboard/
```

Large checkpoint files are stored separately on the SSD by default:

```text
/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/<run-name>/
  checkpoint_last.pt
  checkpoint_best.pt
  checkpoint_epoch_XXXX.pt
```

Use `--run-name NAME` for a stable experiment label. `--output-dir` and
`--checkpoint-dir` remain available when artifacts must be stored elsewhere.
To inspect all runs:

```bash
tensorboard --logdir autophasenn_training_pipeline/runs --port 6006
```

## Data Format

The dataset now uses the same memmap file layout as the root `data_loader.py`:

```text
train_diff.npy / val_diff.npy: float32 diffraction modulus, shape (N, 64, 64, 64)
train_real.npy / val_real.npy: complex64 real-space object, shape (N, 64, 64, 64)
```

After loading, `dataset.py` keeps this pipeline's preprocessing contract:
diffraction, amplitude, and phase are returned as float32 channel-first tensors;
optional `--scale-i` normalization is applied to diffraction; missing real-space
data produces zero amplitude/phase tensors.

## Standard Loss And Metrics

The paper uses a reciprocal-space amplitude/modulus loss:

```text
Loss = sum(|sqrt(Ie) - sqrt(Im)|) / N^3
```

In this codebase the diffraction tensors are already `abs(FFT)`, i.e.
`sqrt(intensity)`, so the standard training loss is simply per-voxel MAE between
predicted and measured diffraction modulus:

```text
--loss-type paper_mae
```

This is the default and is implemented with the PyTorch standard wrapper:

```python
torch.nn.L1Loss(reduction="mean")
```

For modulus tensors this is mathematically identical to the paper loss. Using
the standard wrapper also avoids tiny differences from custom reduction order.

Recommended metrics reported by `evaluate.py`:

```text
paper_modulus_mae    Paper Eq. (1), training loss for modulus tensors
chi2_modulus         Paper Eq. (2), reciprocal-space chi2 metric
real_amp_ssim        Paper Fig. 2, local-window 3D amplitude SSIM
r_factor_free        Amplitude R-factor on held-out reciprocal voxels
llk_free             Poisson deviance on held-out reciprocal voxels
chi2_free            Paper Eq. (2) restricted to held-out reciprocal voxels
relative_l1_modulus  Sum |pred-true| / sum |true|
pearson_corr         Shape correlation in reciprocal space
```

The free R-factor metrics accept an experiment-defined mask through
`--free-mask`. When no mask is supplied, evaluation creates a deterministic 5%
diagnostic mask using `--free-fraction` and `--free-seed`. The generated mask is
clearly identified in the report because it is not automatically excluded from
the model's training objective and therefore is not numerically comparable to
the paper's original free-pixel experiment.

Legacy names are still accepted for old command lines, but are no longer the
recommended defaults:

```text
l1, paper -> paper_mae, implemented by torch.nn.L1Loss
mse       -> implemented by torch.nn.MSELoss
sq        -> chi2_modulus
mae       -> relative_l1_modulus
pcc       -> pearson loss
comb      -> 0.5 * (chi2_modulus + pearson_loss)
```

Training can optionally scale-align predicted diffraction before computing its
loss:

```text
--scale-align-loss
```

That applies:

```text
Y_pred = Y_pred * sum(Y_true) / sum(Y_pred)
```

before computing the diffraction loss. The standalone evaluator intentionally
does not apply this transformation: reciprocal-space metrics use the raw model
`pred_diff` output.

## Smoke Test

From the repository root on Windows PowerShell:

```powershell
$PY = ".\.conda_autophase_tfpt\python.exe"
& $PY "autophasenn_training_pipeline\train.py" `
  --data-dir "data\aicdi_sample\memmap" `
  --data-train-diff "train_diff.npy" `
  --data-train-real "train_real.npy" `
  --data-val-diff "val_diff.npy" `
  --data-val-real "val_real.npy" `
  --num-samples-train 3 `
  --num-samples-val 3 `
  --output-dir "outputs\pt_pipeline_dryrun" `
  --pretrained "PyTorch\cohere-trained_model_tf_compatible.pth" `
  --device cpu `
  --batch-size 1 `
  --loss-type paper_mae `
  --unsupervised `
  --dry-run
```

## 100-Sample Fit Check

This runs from random initialization, uses reciprocal-space L1 only, trains and
validates on the same first 100 training samples, and caches those samples in
RAM:

```bash
python autophasenn_training_pipeline/train.py \
  --data-dir /data_ssd/oyys/autophasenn \
  --data-train-diff train_diff.npy \
  --data-train-real train_real.npy \
  --output-dir ./output/autophasenn_overfit100_l1 \
  --from-scratch \
  --loss-type l1 \
  --loss-scope diff \
  --overfit-samples 100 \
  --batch-size 8 \
  --epochs 200 \
  --lr 1e-3 \
  --num-workers 0
```

## Fine-Tune Example

```powershell
$PY = ".\.conda_autophase_tfpt\python.exe"
& $PY "autophasenn_training_pipeline\train.py" `
  --data-dir "data\aicdi_sample\memmap" `
  --data-train-diff "train_diff.npy" `
  --data-train-real "train_real.npy" `
  --data-val-diff "val_diff.npy" `
  --data-val-real "val_real.npy" `
  --num-samples-train 25000 `
  --num-samples-val 5000 `
  --output-dir "outputs\pt_finetune" `
  --pretrained "PyTorch\cohere-trained_model_tf_compatible.pth" `
  --device cuda `
  --batch-size 1 `
  --epochs 20 `
  --loss-type paper_mae `
  --unsupervised `
  --lr 1e-5
```

`--scale-align-loss` is available, but use it intentionally: it rescales the
predicted diffraction before loss calculation and therefore changes the training
objective from the paper's direct MAE.

Use `--device cpu` if no CUDA GPU is available. Full 64^3 3D backprop on CPU is
very slow, so CPU is mostly useful for dry runs or small checks.

## Evaluate

By default `evaluate.py` loads the configured `decoder_cross_concat` checkpoint.
Pass `--checkpoint` and its matching `--model-variant` explicitly when
evaluating another model.
Every generated artifact is written under a model-specific directory by
default:

```text
autophasenn_training_pipeline/evaluate/evaluate_<model-variant>/
```

For example, `--model-variant decoder_cross_concat` writes to
`evaluate/evaluate_decoder_cross_concat/`.

```text
evaluation_results.json   Complete configuration, provenance, statistics, and per-sample metrics
evaluation_samples.csv    One readable row per sample
evaluation_summary.md     Paper-first human-readable summary
evaluation.log            Resolved paths and execution log
```

Reciprocal-space metrics use the raw diffraction arrays without input or
prediction rescaling. Real-space amplitude, phase, and support metrics use the
same SciPy/scikit-image post-processing as `visualize_postprocessed.py`: phase
unwrapping, support masking, mean-phase removal, and center-of-mass shifting.
The evaluator batches masking, center-of-mass shifting, and all metric tensors
on the selected PyTorch device. The exact `skimage.unwrap_phase` step remains on
CPU because PyTorch has no equivalent CUDA operator; independent volumes are
unwrapped concurrently. `--postprocess-workers 0` selects up to eight threads
automatically, while `--postprocess-workers 1` limits CPU use. Set
`--num-workers` above zero to overlap memmap loading with GPU inference.

```powershell
$PY = ".\.conda_autophase_tfpt\python.exe"
& $PY "autophasenn_training_pipeline\evaluate.py" `
  --checkpoint "PyTorch\cohere-trained_model_tf_compatible.pth" `
  --data-dir "data\aicdi_sample\memmap" `
  --data-diff "val_diff.npy" `
  --data-real "val_real.npy" `
  --num-samples 5000 `
  --device cpu `
  --limit 3
```

Use the original experiment holdout mask when available:

```powershell
& $PY "autophasenn_training_pipeline\evaluate.py" `
  --checkpoint "PyTorch\cohere-trained_model_tf_compatible.pth" `
  --data-dir "data\aicdi_sample\memmap" `
  --free-mask "data\aicdi_sample\free_mask.npy" `
  --device cpu
```

Use `--data-real none` for experimental data without ground-truth amplitude and
phase; reciprocal-space and free-mask metrics are still produced.

Use `--threshold-sweep` to evaluate multiple real-space support thresholds from
the model's raw pre-support amplitude in one inference pass. The primary
`--threshold` remains the headline result. A sweep additionally writes
`evaluation_threshold_sweep.csv` with one aggregate row per threshold and
`evaluation_threshold_sweep_samples.csv` with per-sample metrics. Thresholds
selected on validation data are calibration diagnostics, not independent test
results.

```bash
python autophasenn_training_pipeline/evaluate.py \
  --model-variant mamba_skip \
  --checkpoint /path/to/checkpoint_best.pt \
  --threshold 0.1 \
  --threshold-sweep 0.05 0.075 0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4
```

## Visualize

```powershell
$PY = ".\.conda_autophase_tfpt\python.exe"
& $PY "autophasenn_training_pipeline\visualize_postprocessed.py" `
  --checkpoint "PyTorch\cohere-trained_model_tf_compatible.pth" `
  --data-dir "data\aicdi_sample\memmap" `
  --data-diff "val_diff.npy" `
  --data-real "val_real.npy" `
  --dataset-size 5000 `
  --device cpu `
  --num-samples 5 `
  --seed 42
```

The visualizer directly reuses the official post-processing functions from
`evaluate.py`: phase unwrapping, support masking, subtracting mean phase inside
support, and center-of-mass shifting. It writes a normalized 2D center-slice
grid, a true/predicted 3D amplitude-isosurface comparison, and a prediction 3D
comparison before/after the center shift. The 3D surface color is wrapped phase
in radians and the red guides mark the volume center. A separate real-space 3D
error figure shows signed amplitude error (`true - prediction`) and wrapped
phase error on the true/predicted support intersection. Its amplitude-error
surface is controlled by `--amplitude-error-level`.

Four independent sampled-volume 3D figures make amplitude and phase values
explicit instead of combining them on one surface:

```text
visualization_amplitude_3d.png          Real-space amplitude
visualization_phase_3d.png              Real-space wrapped phase
visualization_diffraction_3d.png        Diffraction modulus (log10 normalized)
visualization_diffraction_phase_3d.png  Derived wrapped diffraction phase
```

Every figure uses the same five columns and sign convention:

```text
1. Target
2. Prediction before center shift
3. Prediction after center shift
4. Difference: before - after
5. Difference: after - target
```

The first three columns share one color scale; the two signed-difference columns
share a symmetric scale. Phase differences are wrapped to `[-pi, pi]`. Phase is
shown only inside valid real-space support, and diffraction phase is shown only
where normalized diffraction modulus exceeds `--reciprocal-phase-threshold`.
Phase-difference panels use the intersection of the two valid supports being
compared. The target amplitude/phase is evaluator-centered, while the two
prediction columns intentionally expose the state before and after prediction
centering.
`--max-volume-points`, `--volume-point-size`, and `--volume-alpha` control the
sampled-voxel volume rendering without changing any calculated volume.

The reciprocal-space outputs distinguish diffraction modulus from phase. The
stored `*_diff.npy` tensors and model `pred_diff` output are diffraction modulus
(`abs(FFT)`, or square root of intensity), not diffraction phase. When
real-space ground truth is available, the visualizer derives its Fourier phase
using the same `ifftshift -> fftn -> fftshift` convention as the model and
compares it with the phase derived from the predicted complex object. These are
derived phases: experimental modulus-only input does not contain a measured
diffraction phase. Both complex objects first undergo the evaluator's mean-phase
removal and center-of-mass registration to reduce global and linear phase
ambiguities. Low-modulus voxels are hidden because phase is undefined
there; configure this with `--reciprocal-phase-threshold`, and control the 3D
reciprocal modulus surface with `--reciprocal-surface-level`.
Because a real-space translation changes Fourier phase but not Fourier modulus,
the diffraction-modulus `before - after` panel should be numerically near zero;
the corresponding diffraction-phase panel exposes the expected phase ramp.

Adjust general surface detail with `--surface-step-size` and the camera with
`--view-elevation` and `--view-azimuth`; pass `none` to any optional output path
to disable that image.
All default visualization outputs are written under a model-specific directory
next to `evaluate`:

```text
autophasenn_training_pipeline/vision/vision_<model-variant>/
```

Use `--output-dir` to override the directory for all default image names, or
pass an explicit image path to override only that output. The directory is
created automatically when the first image is saved.
These display operations are not part of training backprop.

## Server Background Training

```bash
cd /home/oyys/code/UMamba-AutoPhaseNN

RUN_NAME="decoder_cross_concat_ft_bs4_lr1e-4_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$PWD/autophasenn_training_pipeline/runs/${RUN_NAME}"
BASELINE_CKPT="/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/autophasenn_retrain_l1/checkpoint_best.pt"

mkdir -p "${RUN_DIR}"

nohup env CUDA_VISIBLE_DEVICES=0 python -u \
  autophasenn_training_pipeline/train.py \
  --model-variant decoder_cross_concat \
  --pretrained "${BASELINE_CKPT}" \
  --run-name "${RUN_NAME}" \
  --data-dir /data_ssd/oyys/autophasenn \
  --data-train-diff train_diff.npy \
  --data-train-real train_real.npy \
  --data-val-diff val_diff.npy \
  --data-val-real val_real.npy \
  --num-samples-train 25000 \
  --num-samples-val 5000 \
  --epochs 100 \
  --batch-size 4 \
  --num-workers 4 \
  --device cuda \
  --loss-type paper_mae \
  --loss-scope diff \
  --optimizer adam \
  --lr 1e-4 \
  --lr-scheduler plateau \
  --threshold 0.1 \
  --save-every 10 \
  --print-freq 50 \
  > "${RUN_DIR}/console.log" 2>&1 < /dev/null &

PID=$!
echo "${PID}" > "${RUN_DIR}/train.pid"

echo "训练已启动"
echo "PID        : ${PID}"
echo "日志       : ${RUN_DIR}/console.log"
echo "运行目录   : ${RUN_DIR}"
```
