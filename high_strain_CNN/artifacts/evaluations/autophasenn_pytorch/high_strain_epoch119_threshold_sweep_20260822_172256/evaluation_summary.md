# HighStrain AutoPhaseNN Evaluation Summary

## Run

| Item | Value |
|---|---|
| Checkpoint | `/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/high_strain_cnn/high_strain_reduced_centered_resume_old60_lr1e-4_plateau_20260820_232732/checkpoint_best.pt` |
| Checkpoint epoch | 119 |
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
| Evaluation wall time | 449.563 s |
| Mean inference | 27.9031 ms/sample |
| Throughput | 35.8383 samples/s |

## Metric Interpretation

The real-space metrics below use the same AutoPhaseNN post-processing and metric implementations and are directly comparable. Reciprocal modulus metrics are not model-quality measurements for this method: reconstruction explicitly reuses the measured modulus, so their near-zero errors only test FFT/reprojection consistency.

## Comparable Quality Metrics

| Metric | Mean | Std | P50 | P95 | Better | Meaning |
|---|---:|---:|---:|---:|---|---|
| `phase_wca` | 0.676606 | 0.13122 | 0.657969 | 0.910479 | lower | Reciprocal-phase WCA objective used for training. |
| `real_amp_l1` | 0.0076051 | 0.00430019 | 0.00652481 | 0.0158298 | lower | Full-volume post-processed amplitude L1. |
| `real_amp_ssim` | 0.897686 | 0.0630138 | 0.915949 | 0.969127 | higher | Local-window 3D amplitude SSIM. |
| `real_support_iou` | 0.489877 | 0.0875281 | 0.488444 | 0.63195 | higher | Post-processed support intersection-over-union. |
| `real_support_dice` | 0.652926 | 0.079913 | 0.656314 | 0.774472 | higher | Post-processed support Dice score. |
| `real_support_volume_ratio` | 2.0534 | 0.386331 | 1.99472 | 2.79516 | near 1 | Predicted/true support volume ratio. |
| `real_phase_mae_true_support` | 0.706692 | 0.510701 | 0.566999 | 1.77613 | lower | Wrapped phase MAE on the post-processed true support. |

## Support Threshold Sweep

The primary support threshold remains unchanged. Sweep-selected operating points are validation diagnostics and do not replace the headline metrics above.

| Threshold | Primary | Amp L1 | Amp SSIM | Support IoU | Support Dice | Volume ratio | Phase MAE |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0.05 |  | 0.0140509 | 0.702201 | 0.184762 | 0.308693 | 5.74227 | 1.37551 |
| 0.075 |  | 0.00967829 | 0.837884 | 0.347833 | 0.510981 | 2.96242 | 1.02911 |
| 0.1 | yes | 0.0076051 | 0.897686 | 0.489877 | 0.652926 | 2.0534 | 0.706692 |
| 0.125 |  | 0.00648198 | 0.927244 | 0.599391 | 0.745704 | 1.65112 | 0.499835 |
| 0.15 |  | 0.00577608 | 0.943713 | 0.680538 | 0.806784 | 1.43377 | 0.378071 |
| 0.2 |  | 0.00496603 | 0.959932 | 0.779714 | 0.874059 | 1.22104 | 0.265038 |
| 0.25 |  | 0.00454342 | 0.966759 | 0.827139 | 0.903714 | 1.13101 | 0.22488 |

- Best mean-IoU threshold: `0.25`.
- Threshold with mean volume ratio closest to one: `0.25`.

## AutoPhaseNN-Compatible Fixed Metrics

The `FT` row is retained for file-format compatibility but is a reprojection identity, not an independently predicted quantity.

| Group | Metric | Mean |
|---|---|---:|
| FT (reprojection only) | L1 | 7.06623e-06 |
| FT (reprojection only) | MSE | 3.27304e-10 |
| FT (reprojection only) | RMSE | 0.000101621 |
| FT (reprojection only) | RelL1 | 5.95252e-07 |
| Amplitude | L1 | 0.0076051 |
| Amplitude | MSE | 0.00189812 |
| Amplitude | RMSE | 0.0413962 |
| Amplitude | RelL1 | 0.411792 |
| Phase | L1 | 0.706692 |
| Phase | MSE | 0.841713 |
| Phase | RMSE | 0.778774 |
| Phase | RelL1 | 1.30505 |
| Support | L1 | 0.0259792 |
| Support | MSE | 0.0259792 |
| Support | RMSE | 0.154046 |
| Support | RelL1 | 1.09003 |

## Mean Metrics by Group

### Phase Retrieval

| Metric | Mean | Meaning |
|---|---:|---|
| `phase_wca` | 0.676606 | Published symmetry-aware reciprocal-phase WCA loss. |
| `phase_wca_direct` | 0.783117 | WCA error against the direct reciprocal phase. |
| `phase_wca_inverted` | 0.864377 | WCA error against the conjugate/twin phase. |
| `twin_flip_selected` | 0.3652 | Fraction indicator for evaluation-time conjugate/twin sign selection. |

### Realspace Primary

| Metric | Mean | Meaning |
|---|---:|---|
| `real_amp_l1` | 0.0076051 | Real-space full-volume amplitude L1. Lower is better, but can be small for sparse objects. |
| `real_amp_ssim` | 0.897686 | Local-window 3D amplitude SSIM reported by the paper. Higher is better. |
| `real_amp_global_ssim` | 0.925925 | Global 3D amplitude SSIM-like score. Higher is better. |
| `real_support_iou` | 0.489877 | Intersection-over-union between predicted and true support. Higher is better. |
| `real_support_dice` | 0.652926 | Dice score between predicted and true support. Higher is better. |
| `real_support_pred_fraction` | 0.0485605 | Predicted support fraction in the 64^3 volume; should be close to true fraction. |
| `real_support_volume_ratio` | 2.0534 | pred_support_fraction / true_support_fraction; ideal is near 1. |
| `real_phase_mae_true_support` | 0.706692 | Wrapped phase MAE on the true support. Lower is better. |

### Realspace Diagnostic

| Metric | Mean | Meaning |
|---|---:|---|
| `real_amp_mse` | 0.00189812 | Real-space full-volume amplitude MSE. Lower is better. |
| `real_amp_rmse` | 0.0413962 | Real-space full-volume amplitude RMSE. Lower is better. |
| `real_amp_rel_l1` | 0.411792 | Real-space amplitude L1 normalized by true amplitude sum. Lower is better. |
| `real_support_l1` | 0.0259792 | Binary support mask L1. Lower is better. |
| `real_support_rmse` | 0.154046 | Binary support mask RMSE. Lower is better. |
| `real_support_true_fraction` | 0.0234351 | True support fraction in the 64^3 volume. |
| `real_phase_mae_intersection` | 0.702774 | Wrapped phase MAE on support intersection. Lower is better. |
| `real_phase_rmse_true_support` | 0.778774 | Wrapped phase RMSE on the true support. Lower is better. |

### Reprojection Identity

| Metric | Mean | Meaning |
|---|---:|---|
| `paper_modulus_mae` | 7.06623e-06 | Measured-modulus reprojection L1; near zero by construction and not comparable to an independently predicted modulus. |
| `relative_l1_modulus` | 5.95252e-07 | Scale-normalized reprojection consistency. |
| `chi2_modulus` | 8.20553e-14 | Reprojection chi-square consistency. |
| `pearson_corr` | 1 | Measured/reprojected modulus correlation. |
| `voxel_mse` | 3.27304e-10 | Measured/reprojected modulus voxel MSE. |
| `voxel_rmse` | 0.000101621 | Measured/reprojected modulus voxel RMSE. |

### Timing

| Metric | Mean | Meaning |
|---|---:|---|
| `inference_ms` | 27.9031 | Mean model forward latency assigned to each sample. |

## Interpretation Notes

- `twin_aligned` uses the target only to choose between the two reciprocal-phase signs treated as equivalent by the published WCA loss.
- `phase_wca`, amplitude SSIM, support IoU/Dice, and real-space phase error are the meaningful quality indicators for this architecture.
- A support volume ratio far from one indicates that reciprocal-phase errors spread reconstructed energy outside the object.

## Files

- `evaluation_results.json`: configuration, provenance, distributions, and per-sample metrics.
- `evaluation_samples.csv`: one row per sample.
- `evaluation_summary.md`: this readable summary.
- `evaluation.log`: execution log.
- `evaluation_threshold_sweep.csv`: one aggregate row per threshold.
- `evaluation_threshold_sweep_samples.csv`: per-sample sweep metrics.
