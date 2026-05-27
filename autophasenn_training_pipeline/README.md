# AutoPhaseNN PyTorch Training Pipeline

This folder is a cleaned, standalone PyTorch training/evaluation flow for the
TF-compatible AutoPhaseNN model that can load the converted checkpoint:

```text
PyTorch/cohere-trained_model_tf_compatible.pth
```

## Files

```text
dataset.py                  Data list parsing and .npz/.npy loading
losses.py                   Project loss functions and scale alignment
model_tf_compatible.py      PyTorch model matching the converted TF2 checkpoint
train.py                    Full train / fine-tune entry point
evaluate.py                 Checkpoint evaluation and loss report
visualize_postprocessed.py  TF test_network_unsup-style visualization
```

## Data Format

Preferred sample format is `.npz`:

```text
arr_0: diffraction amplitude, shape (64, 64, 64)
arr_1: complex real-space object, shape (64, 64, 64)
```

The loader also supports `.npy` or `.npz` files that contain one complex
diffraction array. In that case it uses `abs(diffraction)` as input and
computes the real-space target with `ifftn(ifftshift(...))`.

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
& $PY "pytorch_training_pipeline\train.py" `
  --data-dir "data\aicdi_sample\prepared" `
  --data-list "3D_upsamp.txt" `
  --output-dir "outputs\pt_pipeline_dryrun" `
  --pretrained "PyTorch\cohere-trained_model_tf_compatible.pth" `
  --device cpu `
  --batch-size 1 `
  --train-size 3 `
  --loss-type paper_mae `
  --unsupervised `
  --dry-run
```

## Fine-Tune Example

```powershell
$PY = ".\.conda_autophase_tfpt\python.exe"
& $PY "pytorch_training_pipeline\train.py" `
  --data-dir "data\aicdi_sample\prepared" `
  --data-list "3D_upsamp.txt" `
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

```powershell
$PY = ".\.conda_autophase_tfpt\python.exe"
& $PY "pytorch_training_pipeline\evaluate.py" `
  --checkpoint "PyTorch\cohere-trained_model_tf_compatible.pth" `
  --data-dir "data\aicdi_sample\prepared" `
  --data-list "3D_upsamp.txt" `
  --output-json "outputs\pt_pipeline_eval.json" `
  --device cpu `
  --limit 3
```

Add `--scale-align-loss` if you want the scale-aligned metric report.

## Visualize

```powershell
$PY = ".\.conda_autophase_tfpt\python.exe"
& $PY "pytorch_training_pipeline\visualize_postprocessed.py" `
  --checkpoint "PyTorch\cohere-trained_model_tf_compatible.pth" `
  --data-dir "data\aicdi_sample\prepared" `
  --data-list "3D_upsamp.txt" `
  --output-png "outputs\pt_pipeline_visualization.png" `
  --device cpu `
  --num-samples 3
```

The visualization intentionally mirrors `TF2/test_network_unsup.py`: phase
unwrap, support masking, subtracting mean phase inside support, center-of-mass
shifting, and center-slice plotting. These display operations are not part of
training backprop.
