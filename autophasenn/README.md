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

## Important Loss Options

`losses.py` contains the same PyTorch-style functions discussed during testing:

```text
l1, log, sq, mae, paper, pcc, comb, comb2, comb_log
```

By default the custom project losses sum over the batch, matching the original
TF-style definitions. If you want the training loss divided by batch size for
backprop, pass:

```text
--batch-average-loss
```

To reproduce the "scale alignment" evaluation you used:

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
  --loss-type comb `
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
  --loss-type comb `
  --unsupervised `
  --lr 1e-5 `
  --scale-align-loss
```

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

