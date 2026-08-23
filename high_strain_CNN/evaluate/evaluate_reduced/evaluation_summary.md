# HighStrain AutoPhaseNN Evaluation Summary

## Run

| Item | Value |
|---|---|
| Checkpoint | `/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/high_strain_cnn/high_strain_reduced_centered_resume120_to240_20260822_172432/checkpoint_best.pt` |
| Checkpoint epoch | 158 |
| Model variant | `reduced` |
| Model parameters | 39,160,897 |
| Project version | `1.4.0` |
| Git commit | `5be3014ccf79a8d69423b477ae0829562e3bfa8c` |
| Samples | 5000 |
| Batch size | 16 |
| Device | `cuda` |
| GPU | NVIDIA GeForce RTX 4090 |
| Ambiguity mode | `twin_aligned` |
| Support threshold | 0.1 |
| Phase unwrap workers | 8 |
| Evaluation wall time | 375.506 s |
| Mean inference | 22.2712 ms/sample |
| Throughput | 44.901 samples/s |

## Metric Interpretation

The real-space metrics below use the same AutoPhaseNN post-processing and metric implementations and are directly comparable. Reciprocal modulus metrics are not model-quality measurements for this method: reconstruction explicitly reuses the measured modulus, so their near-zero errors only test FFT/reprojection consistency.

## Comparable Quality Metrics

| Metric | Mean | Std | P50 | P95 | Better | Meaning |
|---|---:|---:|---:|---:|---|---|
| `phase_wca` | 0.667847 | 0.139546 | 0.650184 | 0.912103 | lower | Reciprocal-phase WCA objective used for training. |
| `real_amp_l1` | 0.00722869 | 0.00407635 | 0.00623544 | 0.0150865 | lower | Full-volume post-processed amplitude L1. |
| `real_amp_ssim` | 0.905985 | 0.0582976 | 0.922112 | 0.972255 | higher | Local-window 3D amplitude SSIM. |
| `real_support_iou` | 0.513093 | 0.088853 | 0.514104 | 0.656717 | higher | Post-processed support intersection-over-union. |
| `real_support_dice` | 0.673584 | 0.0788988 | 0.679087 | 0.792793 | higher | Post-processed support Dice score. |
| `real_support_volume_ratio` | 1.95707 | 0.3582 | 1.89451 | 2.63569 | near 1 | Predicted/true support volume ratio. |
| `real_phase_mae_true_support` | 0.649965 | 0.469745 | 0.521392 | 1.53949 | lower | Wrapped phase MAE on the post-processed true support. |

## AutoPhaseNN-Compatible Fixed Metrics

The `FT` row is retained for file-format compatibility but is a reprojection identity, not an independently predicted quantity.

| Group | Metric | Mean |
|---|---|---:|
| FT (reprojection only) | L1 | 7.06722e-06 |
| FT (reprojection only) | MSE | 3.27633e-10 |
| FT (reprojection only) | RMSE | 0.000101622 |
| FT (reprojection only) | RelL1 | 5.95328e-07 |
| Amplitude | L1 | 0.00722869 |
| Amplitude | MSE | 0.00182963 |
| Amplitude | RMSE | 0.0405421 |
| Amplitude | RelL1 | 0.392725 |
| Phase | L1 | 0.649965 |
| Phase | MSE | 0.718471 |
| Phase | RMSE | 0.720712 |
| Phase | RelL1 | 1.21091 |
| Support | L1 | 0.0235346 |
| Support | MSE | 0.0235346 |
| Support | RMSE | 0.14677 |
| Support | RelL1 | 0.992923 |

## Mean Metrics by Group

### Phase Retrieval

| Metric | Mean | Meaning |
|---|---:|---|
| `phase_wca` | 0.667847 | Published symmetry-aware reciprocal-phase WCA loss. |
| `phase_wca_direct` | 0.777611 | WCA error against the direct reciprocal phase. |
| `phase_wca_inverted` | 0.861853 | WCA error against the conjugate/twin phase. |
| `twin_flip_selected` | 0.3654 | Fraction indicator for evaluation-time conjugate/twin sign selection. |

### Realspace Primary

| Metric | Mean | Meaning |
|---|---:|---|
| `real_amp_l1` | 0.00722869 | Real-space full-volume amplitude L1. Lower is better, but can be small for sparse objects. |
| `real_amp_ssim` | 0.905985 | Local-window 3D amplitude SSIM reported by the paper. Higher is better. |
| `real_amp_global_ssim` | 0.929631 | Global 3D amplitude SSIM-like score. Higher is better. |
| `real_support_iou` | 0.513093 | Intersection-over-union between predicted and true support. Higher is better. |
| `real_support_dice` | 0.673584 | Dice score between predicted and true support. Higher is better. |
| `real_support_pred_fraction` | 0.0461296 | Predicted support fraction in the 64^3 volume; should be close to true fraction. |
| `real_support_volume_ratio` | 1.95707 | pred_support_fraction / true_support_fraction; ideal is near 1. |
| `real_phase_mae_true_support` | 0.649965 | Wrapped phase MAE on the true support. Lower is better. |

### Realspace Diagnostic

| Metric | Mean | Meaning |
|---|---:|---|
| `real_amp_mse` | 0.00182963 | Real-space full-volume amplitude MSE. Lower is better. |
| `real_amp_rmse` | 0.0405421 | Real-space full-volume amplitude RMSE. Lower is better. |
| `real_amp_rel_l1` | 0.392725 | Real-space amplitude L1 normalized by true amplitude sum. Lower is better. |
| `real_support_l1` | 0.0235346 | Binary support mask L1. Lower is better. |
| `real_support_rmse` | 0.14677 | Binary support mask RMSE. Lower is better. |
| `real_support_true_fraction` | 0.0234351 | True support fraction in the 64^3 volume. |
| `real_phase_mae_intersection` | 0.645743 | Wrapped phase MAE on support intersection. Lower is better. |
| `real_phase_rmse_true_support` | 0.720712 | Wrapped phase RMSE on the true support. Lower is better. |

### Reprojection Identity

| Metric | Mean | Meaning |
|---|---:|---|
| `paper_modulus_mae` | 7.06722e-06 | Measured-modulus reprojection L1; near zero by construction and not comparable to an independently predicted modulus. |
| `relative_l1_modulus` | 5.95328e-07 | Scale-normalized reprojection consistency. |
| `chi2_modulus` | 8.21224e-14 | Reprojection chi-square consistency. |
| `pearson_corr` | 1 | Measured/reprojected modulus correlation. |
| `voxel_mse` | 3.27633e-10 | Measured/reprojected modulus voxel MSE. |
| `voxel_rmse` | 0.000101622 | Measured/reprojected modulus voxel RMSE. |

### Timing

| Metric | Mean | Meaning |
|---|---:|---|
| `inference_ms` | 22.2712 | Mean model forward latency assigned to each sample. |

## Interpretation Notes

- `twin_aligned` uses the target only to choose between the two reciprocal-phase signs treated as equivalent by the published WCA loss.
- `phase_wca`, amplitude SSIM, support IoU/Dice, and real-space phase error are the meaningful quality indicators for this architecture.
- A support volume ratio far from one indicates that reciprocal-phase errors spread reconstructed energy outside the object.

## Files

- `evaluation_results.json`: configuration, provenance, distributions, and per-sample metrics.
- `evaluation_samples.csv`: one row per sample.
- `evaluation_summary.md`: this readable summary.
- `evaluation.log`: execution log.
