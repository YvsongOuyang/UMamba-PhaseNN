# AutoPhaseNN Evaluation Summary

## Run

| Item | Value |
|---|---|
| Checkpoint | `/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/autophasenn_retrain_l1/checkpoint_best.pt` |
| Checkpoint epoch | 90 |
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
| Total evaluation wall time | 305.795 s |
| Mean model inference | 6.36247 ms/sample |
| Model throughput | 157.172 samples/s |

## Paper Metric Coverage

| Metric | Mean | Std | P50 | P95 | Better | Paper usage |
|---|---:|---:|---:|---:|---|---|
| `paper_modulus_mae` | 3.69721 | 0.79052 | 3.51088 | 5.19699 | lower | Eq. (1): MAE between measured and estimated diffraction modulus. |
| `chi2_modulus` | 0.0165266 | 0.0153732 | 0.0119957 | 0.0421238 | lower | Eq. (2): reciprocal-space chi-square on diffraction modulus. |
| `real_amp_ssim` | 0.957022 | 0.031314 | 0.9612 | 0.996891 | higher | Fig. 2: local-window 3D SSIM of real-space amplitude. |
| `r_factor_free` | 0.324797 | 0.059368 | 0.314378 | 0.438939 | lower | Free R-factor family evaluated on held-out reciprocal voxels. |
| `llk_free` | 275.794 | 344.597 | 171.833 | 809.043 | lower | Supplementary Note 3: free Poisson log-likelihood diagnostic. |
| `chi2_free` | 0.023327 | 0.0206861 | 0.0170128 | 0.0624763 | lower | Supplementary Note 3: free chi-square diagnostic. |

## Free-Metric Provenance

- Source: `generated_diagnostic_mask`
- Selected voxels: 12906 (4.92325%)
- Interpretation: The generated mask is reproducible but was not automatically excluded during model training. Treat Rfree, LLKfree, and chi2free as diagnostics, not as a numerically comparable reproduction of the paper.

## Mean Metrics by Group

### Reciprocal Primary

| Metric | Mean | Meaning |
|---|---:|---|
| `paper_modulus_mae` | 3.69721 | Primary paper-style far-field modulus L1. Lower is better. |
| `relative_l1_modulus` | 0.314188 | Scale-normalized far-field L1. Lower is better. |
| `chi2_modulus` | 0.0165266 | Paper chi-square style far-field error. Lower is better. |
| `pearson_corr` | 0.991564 | Far-field Pearson correlation. Higher is better. |

### Realspace Primary

| Metric | Mean | Meaning |
|---|---:|---|
| `real_amp_l1` | 0.00578829 | Real-space full-volume amplitude L1. Lower is better, but can be small for sparse objects. |
| `real_amp_ssim` | 0.957022 | Local-window 3D amplitude SSIM reported by the paper. Higher is better. |
| `real_amp_global_ssim` | 0.875431 | Global 3D amplitude SSIM-like score. Higher is better. |
| `real_support_iou` | 0.746818 | Intersection-over-union between predicted and true support. Higher is better. |
| `real_support_dice` | 0.845206 | Dice score between predicted and true support. Higher is better. |
| `real_support_pred_fraction` | 0.0236651 | Predicted support fraction in the 64^3 volume; should be close to true fraction. |
| `real_support_volume_ratio` | 1.01312 | pred_support_fraction / true_support_fraction; ideal is near 1. |
| `real_phase_mae_true_support` | 0.239835 | Wrapped phase MAE on the true support. Lower is better. |

### Reciprocal Diagnostic

| Metric | Mean | Meaning |
|---|---:|---|
| `relative_log_mse` | 0.10869 | Log-domain far-field diagnostic. Lower is better. |
| `pearson_loss` | 0.00843609 | 1 - pearson_corr. Lower is better. |
| `voxel_mse` | 62.1575 | Raw far-field MSE on the current data scale. Lower is better. |
| `voxel_rmse` | 7.27723 | Raw far-field RMSE on the current data scale. Lower is better. |

### Realspace Diagnostic

| Metric | Mean | Meaning |
|---|---:|---|
| `real_amp_mse` | 0.00362251 | Real-space full-volume amplitude MSE. Lower is better. |
| `real_amp_rmse` | 0.052491 | Real-space full-volume amplitude RMSE. Lower is better. |
| `real_support_true_fraction` | 0.0234351 | True support fraction in the 64^3 volume. |
| `real_phase_mae_intersection` | 0.126754 | Wrapped phase MAE on support intersection. Lower is better. |
| `real_phase_rmse_true_support` | 0.419603 | Wrapped phase RMSE on the true support. Lower is better. |

### Paper Free

| Metric | Mean | Meaning |
|---|---:|---|
| `r_factor_free` | 0.324797 | Amplitude R-factor on held-out reciprocal voxels. Lower is better. |
| `llk_free` | 275.794 | Mean Poisson deviance on held-out reciprocal voxels. Lower is better. |
| `chi2_free` | 0.023327 | Paper Eq. (2) chi-square restricted to held-out reciprocal voxels. Lower is better. |

### Other

| Metric | Mean | Meaning |
|---|---:|---|
| `inference_ms` | 6.36247 | Additional evaluation diagnostic. |
| `real_amp_rel_l1` | 0.32807 | Real-space amplitude L1 normalized by true amplitude sum. Lower is better. |
| `real_phase_l1_true_support` | 0.239835 | Wrapped phase L1 on the true support. Lower is better. |
| `real_phase_mse_true_support` | 0.240881 | Wrapped phase MSE on the true support. Lower is better. |
| `real_phase_rel_l1_true_support` | 0.347821 | Wrapped phase L1 normalized by target phase magnitude on the true support. Lower is better. |
| `real_support_l1` | 0.00697791 | Binary support mask L1. Lower is better. |
| `real_support_mse` | 0.00697791 | Binary support mask MSE. Lower is better. |
| `real_support_rel_l1` | 0.311616 | Binary support mask L1 normalized by true support volume. Lower is better. |
| `real_support_rmse` | 0.0761934 | Binary support mask RMSE. Lower is better. |

## Files

- `evaluation_results.json`: full configuration, provenance, distributions, and per-sample values.
- `evaluation_samples.csv`: one row per evaluated sample.
- `evaluation_summary.md`: this readable summary.
- `evaluation.log`: execution log and resolved paths.
