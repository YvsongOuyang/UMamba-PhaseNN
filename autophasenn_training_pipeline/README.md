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
train.py                    Full train / fine-tune entry point
evaluate.py                 Checkpoint evaluation and loss report
visualize_postprocessed.py  TF test_network_unsup-style visualization
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
relative_l1_modulus  Sum |pred-true| / sum |true|
pearson_corr         Shape correlation in reciprocal space
```

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

To reproduce the scale-aligned evaluation you used:

```text
--scale-align-loss
```

That applies:

```text
Y_pred = Y_pred * sum(Y_true) / sum(Y_pred)
```

before computing the diffraction loss.

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

```powershell
$PY = ".\.conda_autophase_tfpt\python.exe"
& $PY "autophasenn_training_pipeline\evaluate.py" `
  --checkpoint "PyTorch\cohere-trained_model_tf_compatible.pth" `
  --data-dir "data\aicdi_sample\memmap" `
  --data-diff "val_diff.npy" `
  --data-real "val_real.npy" `
  --num-samples 5000 `
  --output-json "outputs\pt_pipeline_eval.json" `
  --device cpu `
  --limit 3
```

Add `--scale-align-loss` if you want the scale-aligned metric report.

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
