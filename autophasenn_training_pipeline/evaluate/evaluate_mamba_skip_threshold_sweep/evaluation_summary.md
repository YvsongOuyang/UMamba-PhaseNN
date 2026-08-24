# AutoPhaseNN Evaluation Summary

## Run

| Item | Value |
|---|---|
| Checkpoint | `/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/mamba_skip_scratch_bs4_lr1e-3_20260823_155916/checkpoint_best.pt` |
| Checkpoint epoch | 70 |
| Model variant | `mamba_skip` |
| Device | `cuda` |
| PyTorch | `2.2.2` |
| CUDA runtime | 11.8 |
| GPU | NVIDIA GeForce RTX 4090 |
| Samples | 5000 |
| Batch size | 16 |
| Support threshold | 0.1 |
| Support threshold sweep | 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4 |
| SSIM window | 7 x 7 x 7 |
| Real-space ground truth | True |
| Real-space post-process | `official_skimage_unwrap_batched_torch` |
| Post-process tensor device | `cuda` |
| Phase unwrap workers | 8 |
| Total evaluation wall time | 382.883 s |
| Mean model inference | 14.1598 ms/sample |
| Model throughput | 70.6226 samples/s |

## Paper Metric Coverage

| Metric | Mean | Std | P50 | P95 | Better | Paper usage |
|---|---:|---:|---:|---:|---|---|
| `paper_modulus_mae` | 4.06393 | 1.09076 | 3.98808 | 4.76477 | lower | Eq. (1): MAE between measured and estimated diffraction modulus. |
| `chi2_modulus` | 0.0294265 | 0.086607 | 0.0214353 | 0.0368984 | lower | Eq. (2): reciprocal-space chi-square on diffraction modulus. |
| `real_amp_ssim` | 0.955624 | 0.0175329 | 0.958402 | 0.978249 | higher | Fig. 2: local-window 3D SSIM of real-space amplitude. |
| `r_factor_free` | 0.356305 | 0.072201 | 0.348915 | 0.396802 | lower | Free R-factor family evaluated on held-out reciprocal voxels. |
| `llk_free` | 520.264 | 2178.9 | 327.523 | 773.744 | lower | Supplementary Note 3: free Poisson log-likelihood diagnostic. |
| `chi2_free` | 0.038238 | 0.0629183 | 0.0309315 | 0.0615745 | lower | Supplementary Note 3: free chi-square diagnostic. |

## Support Threshold Sweep

The primary threshold remains the headline operating point. Sweep-selected thresholds are validation diagnostics and must not be treated as an independently tested hyperparameter choice.

| Threshold | Primary | Amp L1 | Amp SSIM | Support IoU | Support Dice | Volume ratio | Phase MAE |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0.05 |  | 0.00660937 | 0.945242 | 0.54389 | 0.699448 | 1.65223 | 0.703169 |
| 0.075 |  | 0.0061718 | 0.955452 | 0.725092 | 0.839381 | 1.28674 | 0.391737 |
| 0.1 | yes | 0.0061336 | 0.955624 | 0.735305 | 0.846198 | 1.24846 | 0.38548 |
| 0.125 |  | 0.00610121 | 0.955693 | 0.741643 | 0.850381 | 1.21908 | 0.382599 |
| 0.15 |  | 0.00606912 | 0.955753 | 0.7462 | 0.853368 | 1.19351 | 0.380675 |
| 0.2 |  | 0.00601256 | 0.955799 | 0.751553 | 0.856854 | 1.15596 | 0.374702 |
| 0.25 |  | 0.0059616 | 0.955743 | 0.752708 | 0.857556 | 1.1353 | 0.367882 |
| 0.3 |  | 0.00591512 | 0.955614 | 0.752979 | 0.857698 | 1.1086 | 0.364947 |
| 0.35 |  | 0.00587207 | 0.955414 | 0.751782 | 0.856835 | 1.07776 | 0.365595 |
| 0.4 |  | 0.00580057 | 0.955245 | 0.748026 | 0.854312 | 1.04459 | 0.368249 |

- Best mean-IoU threshold: `0.3`.
- Threshold with mean volume ratio closest to one: `0.4`.

## Free-Metric Provenance

- Source: `generated_diagnostic_mask`
- Selected voxels: 12906 (4.92325%)
- Interpretation: The generated mask is reproducible but was not automatically excluded during model training. Treat Rfree, LLKfree, and chi2free as diagnostics, not as a numerically comparable reproduction of the paper.

## Mean Metrics by Group

### Reciprocal Primary

| Metric | Mean | Meaning |
|---|---:|---|
| `paper_modulus_mae` | 4.06393 | Primary paper-style far-field modulus L1. Lower is better. |
| `relative_l1_modulus` | 0.344542 | Scale-normalized far-field L1. Lower is better. |
| `chi2_modulus` | 0.0294265 | Paper chi-square style far-field error. Lower is better. |
| `pearson_corr` | 0.985741 | Far-field Pearson correlation. Higher is better. |

### Realspace Primary

| Metric | Mean | Meaning |
|---|---:|---|
| `real_amp_l1` | 0.0061336 | Real-space full-volume amplitude L1. Lower is better, but can be small for sparse objects. |
| `real_amp_ssim` | 0.955624 | Local-window 3D amplitude SSIM reported by the paper. Higher is better. |
| `real_amp_global_ssim` | 0.910898 | Global 3D amplitude SSIM-like score. Higher is better. |
| `real_support_iou` | 0.735305 | Intersection-over-union between predicted and true support. Higher is better. |
| `real_support_dice` | 0.846198 | Dice score between predicted and true support. Higher is better. |
| `real_support_pred_fraction` | 0.0286429 | Predicted support fraction in the 64^3 volume; should be close to true fraction. |
| `real_support_volume_ratio` | 1.24846 | pred_support_fraction / true_support_fraction; ideal is near 1. |
| `real_phase_mae_true_support` | 0.38548 | Wrapped phase MAE on the true support. Lower is better. |

### Reciprocal Diagnostic

| Metric | Mean | Meaning |
|---|---:|---|
| `relative_log_mse` | 0.115464 | Log-domain far-field diagnostic. Lower is better. |
| `pearson_loss` | 0.0142589 | 1 - pearson_corr. Lower is better. |
| `voxel_mse` | 125.961 | Raw far-field MSE on the current data scale. Lower is better. |
| `voxel_rmse` | 9.56111 | Raw far-field RMSE on the current data scale. Lower is better. |

### Realspace Diagnostic

| Metric | Mean | Meaning |
|---|---:|---|
| `real_amp_mse` | 0.00252403 | Real-space full-volume amplitude MSE. Lower is better. |
| `real_amp_rmse` | 0.0484642 | Real-space full-volume amplitude RMSE. Lower is better. |
| `real_support_true_fraction` | 0.0234351 | True support fraction in the 64^3 volume. |
| `real_phase_mae_intersection` | 0.352795 | Wrapped phase MAE on support intersection. Lower is better. |
| `real_phase_rmse_true_support` | 0.498333 | Wrapped phase RMSE on the true support. Lower is better. |

### Paper Free

| Metric | Mean | Meaning |
|---|---:|---|
| `r_factor_free` | 0.356305 | Amplitude R-factor on held-out reciprocal voxels. Lower is better. |
| `llk_free` | 520.264 | Mean Poisson deviance on held-out reciprocal voxels. Lower is better. |
| `chi2_free` | 0.038238 | Paper Eq. (2) chi-square restricted to held-out reciprocal voxels. Lower is better. |

### Other

| Metric | Mean | Meaning |
|---|---:|---|
| `inference_ms` | 14.1598 | Additional evaluation diagnostic. |
| `real_amp_rel_l1` | 0.344781 | Real-space amplitude L1 normalized by true amplitude sum. Lower is better. |
| `real_phase_l1_true_support` | 0.38548 | Wrapped phase L1 on the true support. Lower is better. |
| `real_phase_mse_true_support` | 0.275007 | Wrapped phase MSE on the true support. Lower is better. |
| `real_phase_rel_l1_true_support` | 0.649646 | Wrapped phase L1 normalized by target phase magnitude on the true support. Lower is better. |
| `real_support_l1` | 0.00770905 | Binary support mask L1. Lower is better. |
| `real_support_mse` | 0.00770905 | Binary support mask MSE. Lower is better. |
| `real_support_rel_l1` | 0.346818 | Binary support mask L1 normalized by true support volume. Lower is better. |
| `real_support_rmse` | 0.0859689 | Binary support mask RMSE. Lower is better. |

## Files

- `evaluation_results.json`: full configuration, provenance, distributions, and per-sample values.
- `evaluation_samples.csv`: one row per evaluated sample.
- `evaluation_summary.md`: this readable summary.
- `evaluation.log`: execution log and resolved paths.

- `evaluation_threshold_sweep.csv`: one aggregate row per threshold.
- `evaluation_threshold_sweep_samples.csv`: per-sample sweep metrics.
