# BatchNorm Recalibration Experiment

This experiment isolates checkpoint BatchNorm running statistics from all learned
weights. It evaluates the same checkpoint twice:

1. standard `model.eval()` using checkpoint running mean/variance;
2. `model.eval()` after resetting and cumulatively recalibrating only BatchNorm
   running buffers from training-set inputs.

Non-BN modules stay in evaluation mode during calibration. Gradients, optimizer
updates, learned weights, and BatchNorm affine parameters are disabled or untouched.
If `--threshold` is omitted, the checkpoint training threshold is used so BN is the
only changed variable.

Run from the repository root:

```bash
python -u autophasenn_training_pipeline/experiments/bn_recalibration/run.py \
  --checkpoint /data_ssd/oyys/autophasenn/autophasenn_pipeline_output/mamba_skip_scratch_bs4_lr1e-3_20260823_155916/checkpoint_best.pt \
  --model-variant mamba_skip
```

The output directory contains full normal/recalibrated sample metrics, paired
differences, per-layer BN drift, a compact JSON/Markdown report, and the recalibrated
BN buffers. The buffer file is not a full model checkpoint.
