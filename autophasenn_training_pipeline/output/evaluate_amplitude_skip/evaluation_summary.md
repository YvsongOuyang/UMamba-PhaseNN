# AutoPhaseNN Evaluation Summary

## Run

| Item | Value |
|---|---|
| Checkpoint | `/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/amplitude_skip_ft_bs4_lr1e-3_20260816_205650/checkpoint_best.pt` |
| Checkpoint epoch | 59 |
| Model variant | `amplitude_skip` |
| Device | `cuda` |
| PyTorch | `2.2.2` |
| CUDA runtime | 11.8 |
| GPU | NVIDIA GeForce RTX 4090 |
| Samples | 5000 |
| Batch size | 16 |
| Support threshold | 0.1 |
| SSIM window | 7 x 7 x 7 |
| Real-space ground truth | True |
| Real-space post-process | `official_skimage_unwrap_batched_torch` |
| Post-process tensor device | `cuda` |
| Phase unwrap workers | 8 |
| Total evaluation wall time | 372.863 s |
| Mean model inference | 6.45938 ms/sample |
| Model throughput | 154.814 samples/s |

## Paper Metric Coverage

| Metric | Mean | Std | P50 | P95 | Better | Paper usage |
|---|---:|---:|---:|---:|---|---|
| `paper_modulus_mae` | 3.65582 | 0.799115 | 3.46864 | 5.19746 | lower | Eq. (1): MAE between measured and estimated diffraction modulus. |
| `chi2_modulus` | 0.0161675 | 0.015348 | 0.0115789 | 0.0426633 | lower | Eq. (2): reciprocal-space chi-square on diffraction modulus. |
| `real_amp_ssim` | 0.95727 | 0.0316286 | 0.961377 | 0.997345 | higher | Fig. 2: local-window 3D SSIM of real-space amplitude. |
| `r_factor_free` | 0.321165 | 0.0606248 | 0.309915 | 0.43767 | lower | Free R-factor family evaluated on held-out reciprocal voxels. |
| `llk_free` | 269.278 | 348.944 | 164.661 | 800.587 | lower | Supplementary Note 3: free Poisson log-likelihood diagnostic. |
| `chi2_free` | 0.0227632 | 0.0206359 | 0.0164787 | 0.0623455 | lower | Supplementary Note 3: free chi-square diagnostic. |

## Free-Metric Provenance

- Source: `generated_diagnostic_mask`
- Selected voxels: 12906 (4.92325%)
- Interpretation: The generated mask is reproducible but was not automatically excluded during model training. Treat Rfree, LLKfree, and chi2free as diagnostics, not as a numerically comparable reproduction of the paper.

## Mean Metrics by Group

### Reciprocal Primary

| Metric | Mean | Meaning |
|---|---:|---|
| `paper_modulus_mae` | 3.65582 | Primary paper-style far-field modulus L1. Lower is better. |
| `relative_l1_modulus` | 0.31073 | Scale-normalized far-field L1. Lower is better. |
| `chi2_modulus` | 0.0161675 | Paper chi-square style far-field error. Lower is better. |
| `pearson_corr` | 0.991744 | Far-field Pearson correlation. Higher is better. |

### Realspace Primary

| Metric | Mean | Meaning |
|---|---:|---|
| `real_amp_l1` | 0.00576707 | Real-space full-volume amplitude L1. Lower is better, but can be small for sparse objects. |
| `real_amp_ssim` | 0.95727 | Local-window 3D amplitude SSIM reported by the paper. Higher is better. |
| `real_amp_global_ssim` | 0.875694 | Global 3D amplitude SSIM-like score. Higher is better. |
| `real_support_iou` | 0.748613 | Intersection-over-union between predicted and true support. Higher is better. |
| `real_support_dice` | 0.846134 | Dice score between predicted and true support. Higher is better. |
| `real_support_pred_fraction` | 0.0235875 | Predicted support fraction in the 64^3 volume; should be close to true fraction. |
| `real_support_volume_ratio` | 1.00952 | pred_support_fraction / true_support_fraction; ideal is near 1. |
| `real_phase_mae_true_support` | 0.239874 | Wrapped phase MAE on the true support. Lower is better. |

### Reciprocal Diagnostic

| Metric | Mean | Meaning |
|---|---:|---|
| `relative_log_mse` | 0.107203 | Log-domain far-field diagnostic. Lower is better. |
| `pearson_loss` | 0.00825645 | 1 - pearson_corr. Lower is better. |
| `voxel_mse` | 60.8376 | Raw far-field MSE on the current data scale. Lower is better. |
| `voxel_rmse` | 7.18189 | Raw far-field RMSE on the current data scale. Lower is better. |

### Realspace Diagnostic

| Metric | Mean | Meaning |
|---|---:|---|
| `real_amp_mse` | 0.00360913 | Real-space full-volume amplitude MSE. Lower is better. |
| `real_amp_rmse` | 0.0521557 | Real-space full-volume amplitude RMSE. Lower is better. |
| `real_support_true_fraction` | 0.0234351 | True support fraction in the 64^3 volume. |
| `real_phase_mae_intersection` | 0.126634 | Wrapped phase MAE on support intersection. Lower is better. |
| `real_phase_rmse_true_support` | 0.419295 | Wrapped phase RMSE on the true support. Lower is better. |

### Paper Free

| Metric | Mean | Meaning |
|---|---:|---|
| `r_factor_free` | 0.321165 | Amplitude R-factor on held-out reciprocal voxels. Lower is better. |
| `llk_free` | 269.278 | Mean Poisson deviance on held-out reciprocal voxels. Lower is better. |
| `chi2_free` | 0.0227632 | Paper Eq. (2) chi-square restricted to held-out reciprocal voxels. Lower is better. |

### Other

| Metric | Mean | Meaning |
|---|---:|---|
| `inference_ms` | 6.45938 | Additional evaluation diagnostic. |
| `real_amp_rel_l1` | 0.326611 | Real-space amplitude L1 normalized by true amplitude sum. Lower is better. |
| `real_phase_l1_true_support` | 0.239874 | Wrapped phase L1 on the true support. Lower is better. |
| `real_phase_mse_true_support` | 0.241394 | Wrapped phase MSE on the true support. Lower is better. |
| `real_phase_rel_l1_true_support` | 0.347775 | Wrapped phase L1 normalized by target phase magnitude on the true support. Lower is better. |
| `real_support_l1` | 0.00692694 | Binary support mask L1. Lower is better. |
| `real_support_mse` | 0.00692694 | Binary support mask MSE. Lower is better. |
| `real_support_rel_l1` | 0.309181 | Binary support mask L1 normalized by true support volume. Lower is better. |
| `real_support_rmse` | 0.075603 | Binary support mask RMSE. Lower is better. |

## Files

- `evaluation_results.json`: full configuration, provenance, distributions, and per-sample values.
- `evaluation_samples.csv`: one row per evaluated sample.
- `evaluation_summary.md`: this readable summary.
- `evaluation.log`: execution log and resolved paths.
