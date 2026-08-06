# AutoPhaseNN Evaluation Summary

## Run

| Item | Value |
|---|---|
| Checkpoint | `/data_ssd/oyys/autophasenn/autophasenn.pth` |
| Checkpoint epoch | None |
| Device | `cuda` |
| PyTorch | `2.2.2` |
| CUDA runtime | 11.8 |
| GPU | NVIDIA GeForce RTX 4090 |
| Samples | 5000 |
| Batch size | 2 |
| Support threshold | 0.1 |
| SSIM window | 7 x 7 x 7 |
| Real-space ground truth | True |
| Scale-aligned diffraction | False |
| Total evaluation wall time | 49.6079 s |
| Mean model inference | 6.4879 ms/sample |
| Model throughput | 154.133 samples/s |

## Paper Metric Coverage

| Metric | Mean | Std | P50 | P95 | Better | Paper usage |
|---|---:|---:|---:|---:|---|---|
| `paper_modulus_mae` | 7.40553 | 0.611224 | 7.38612 | 8.42964 | lower | Eq. (1): MAE between measured and estimated diffraction modulus. |
| `chi2_modulus` | 0.246147 | 0.107117 | 0.234194 | 0.439198 | lower | Eq. (2): reciprocal-space chi-square on diffraction modulus. |
| `real_amp_ssim` | 0.86511 | 0.0273354 | 0.86714 | 0.906707 | higher | Fig. 2: local-window 3D SSIM of real-space amplitude. |
| `r_factor_free` | 0.640837 | 0.0559284 | 0.637452 | 0.737623 | lower | Free R-factor family evaluated on held-out reciprocal voxels. |
| `llk_free` | 2359.83 | 1084.52 | 2069.59 | 4506.97 | lower | Supplementary Note 3: free Poisson log-likelihood diagnostic. |
| `chi2_free` | 0.268698 | 0.112954 | 0.249311 | 0.480344 | lower | Supplementary Note 3: free chi-square diagnostic. |

## Free-Metric Provenance

- Source: `generated_diagnostic_mask`
- Selected voxels: 12906 (4.92325%)
- Interpretation: The generated mask is reproducible but was not automatically excluded during model training. Treat Rfree, LLKfree, and chi2free as diagnostics, not as a numerically comparable reproduction of the paper.

## Mean Metrics by Group

### Reciprocal Primary

| Metric | Mean | Meaning |
|---|---:|---|
| `paper_modulus_mae` | 7.40553 | Primary paper-style far-field modulus L1. Lower is better. |
| `relative_l1_modulus` | 0.63222 | Scale-normalized far-field L1. Lower is better. |
| `chi2_modulus` | 0.246147 | Paper chi-square style far-field error. Lower is better. |
| `pearson_corr` | 0.953021 | Far-field Pearson correlation. Higher is better. |

### Realspace Primary

| Metric | Mean | Meaning |
|---|---:|---|
| `real_amp_l1` | 0.0295721 | Real-space full-volume amplitude L1. Lower is better, but can be small for sparse objects. |
| `real_amp_ssim` | 0.86511 | Local-window 3D amplitude SSIM reported by the paper. Higher is better. |
| `real_amp_global_ssim` | 0.326748 | Global 3D amplitude SSIM-like score. Higher is better. |
| `real_support_iou` | 0.235735 | Intersection-over-union between predicted and true support. Higher is better. |
| `real_support_dice` | 0.364251 | Dice score between predicted and true support. Higher is better. |
| `real_support_pred_fraction` | 0.0343013 | Predicted support fraction in the 64^3 volume; should be close to true fraction. |
| `real_support_volume_ratio` | 1.51771 | pred_support_fraction / true_support_fraction; ideal is near 1. |
| `real_phase_mae_true_support` | 0.779567 | Wrapped phase MAE on the true support. Lower is better. |

### Reciprocal Diagnostic

| Metric | Mean | Meaning |
|---|---:|---|
| `relative_log_mse` | 0.173908 | Log-domain far-field diagnostic. Lower is better. |
| `pearson_loss` | 0.0469789 | 1 - pearson_corr. Lower is better. |
| `voxel_mse` | 859.646 | Raw far-field MSE on the current data scale. Lower is better. |
| `voxel_rmse` | 28.8794 | Raw far-field RMSE on the current data scale. Lower is better. |

### Realspace Diagnostic

| Metric | Mean | Meaning |
|---|---:|---|
| `real_amp_mse` | 0.0248146 | Real-space full-volume amplitude MSE. Lower is better. |
| `real_amp_rmse` | 0.155704 | Real-space full-volume amplitude RMSE. Lower is better. |
| `real_support_true_fraction` | 0.0234351 | True support fraction in the 64^3 volume. |
| `real_phase_mae_intersection` | 0.456067 | Wrapped phase MAE on support intersection. Lower is better. |
| `real_phase_rmse_true_support` | 1.02372 | Wrapped phase RMSE on the true support. Lower is better. |

### Paper Free

| Metric | Mean | Meaning |
|---|---:|---|
| `r_factor_free` | 0.640837 | Amplitude R-factor on held-out reciprocal voxels. Lower is better. |
| `llk_free` | 2359.83 | Mean Poisson deviance on held-out reciprocal voxels. Lower is better. |
| `chi2_free` | 0.268698 | Paper Eq. (2) chi-square restricted to held-out reciprocal voxels. Lower is better. |

### Other

| Metric | Mean | Meaning |
|---|---:|---|
| `inference_ms` | 6.4879 | Additional evaluation diagnostic. |
| `real_amp_rel_l1` | 1.77718 | Real-space amplitude L1 normalized by true amplitude sum. Lower is better. |
| `real_phase_l1_true_support` | 0.779567 | Wrapped phase L1 on the true support. Lower is better. |
| `real_phase_mse_true_support` | 1.14732 | Wrapped phase MSE on the true support. Lower is better. |
| `real_phase_rel_l1_true_support` | 1.23277 | Wrapped phase L1 normalized by target phase magnitude on the true support. Lower is better. |
| `real_support_l1` | 0.0340113 | Binary support mask L1. Lower is better. |
| `real_support_mse` | 0.0340113 | Binary support mask MSE. Lower is better. |
| `real_support_rel_l1` | 1.61675 | Binary support mask L1 normalized by true support volume. Lower is better. |
| `real_support_rmse` | 0.182419 | Binary support mask RMSE. Lower is better. |

## Files

- `evaluation_results.json`: full configuration, provenance, distributions, and per-sample values.
- `evaluation_samples.csv`: one row per evaluated sample.
- `evaluation_summary.md`: this readable summary.
- `evaluation.log`: execution log and resolved paths.
