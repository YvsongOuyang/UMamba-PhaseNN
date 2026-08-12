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
model_dual_skip.py          DualPairwiseSkipAutoPhaseNN architecture variant
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

## Dual Pairwise Skip Variant

`DualPairwiseSkipAutoPhaseNN` adds standard U-Net concatenation skips only at
`8 x 8 x 8` and `16 x 16 x 16`. Each pre-pooling encoder feature is passed
separately to the amplitude and phase decoder at the matching scale. The two
decoder branches have no direct connection, and all output activations, support
construction, forward physics, and losses remain unchanged.

Initialize it from a baseline checkpoint with:

```bash
python autophasenn_training_pipeline/train.py \
  --model-variant dual_skip \
  --pretrained /path/to/baseline/checkpoint_best.pt
```

The four expanded decoder convolutions copy their baseline channels exactly and
initialize only the added skip-channel kernels to zero. `--pretrained` starts a
fresh optimizer for fine-tuning. Use `--resume` only with a checkpoint already
created by the `dual_skip` variant.

For a fair quick comparison, fine-tune both `baseline` and `dual_skip` from the
same baseline checkpoint with identical data order, seed, optimizer, learning
rate, scheduler, and epoch count. Evaluate each resulting best checkpoint with
the same settings and its matching `--model-variant`.

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

By default `evaluate.py` loads the retraining output checkpoint
`/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/autophasenn_retrain_l1/checkpoint_best.pt`.
Pass `--checkpoint` explicitly when evaluating an older converted checkpoint.
Every generated artifact is written under
`autophasenn_training_pipeline/output/evaluate/` by default:

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

## Visualize

```powershell
$PY = ".\.conda_autophase_tfpt\python.exe"
& $PY "autophasenn_training_pipeline\visualize_postprocessed.py" `
  --checkpoint "PyTorch\cohere-trained_model_tf_compatible.pth" `
  --data-dir "data\aicdi_sample\memmap" `
  --data-diff "val_diff.npy" `
  --data-real "val_real.npy" `
  --dataset-size 5000 `
  --output-png "outputs\pt_pipeline_visualization.png" `
  --device cpu `
  --num-samples 5 `
  --seed 42
```

The visualization intentionally mirrors `TF2/test_network_unsup.py`: phase
unwrap, support masking, subtracting mean phase inside support, center-of-mass
shifting, and center-slice plotting. These display operations are not part of
training backprop.
