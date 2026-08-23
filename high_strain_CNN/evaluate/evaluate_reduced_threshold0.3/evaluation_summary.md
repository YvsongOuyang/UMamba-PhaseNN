# HighStrain AutoPhaseNN Evaluation Summary

## Run

| Item | Value |
|---|---|
| Checkpoint | `/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/high_strain_cnn/high_strain_reduced_centered_resume120_to240_20260822_172432/checkpoint_best.pt` |
| Checkpoint epoch | 158 |
| Model variant | `reduced` |
| Model parameters | 39,160,897 |
| Project version | `1.4.0` |
| Git commit | `e285faab8e44003723bc1427e1ad1b639192ddcd` |
| Samples | 5000 |
| Batch size | 16 |
| Device | `cuda` |
| GPU | NVIDIA GeForce RTX 4090 |
| Ambiguity mode | `twin_aligned` |
| Support threshold | 0.3 |
| Phase unwrap workers | 8 |
| Evaluation wall time | 334.67 s |
| Mean inference | 14.2482 ms/sample |
| Throughput | 70.1845 samples/s |

## Metric Interpretation

The real-space metrics below use the same AutoPhaseNN post-processing and metric implementations and are directly comparable. Reciprocal modulus metrics are not model-quality measurements for this method: reconstruction explicitly reuses the measured modulus, so their near-zero errors only test FFT/reprojection consistency.

## Comparable Quality Metrics

| Metric | Mean | Std | P50 | P95 | Better | Meaning |
|---|---:|---:|---:|---:|---|---|
| `phase_wca` | 0.667847 | 0.139546 | 0.650185 | 0.912103 | lower | Reciprocal-phase WCA objective used for training. |
| `real_amp_l1` | 0.00426837 | 0.0023861 | 0.00372957 | 0.00893623 | lower | Full-volume post-processed amplitude L1. |
| `real_amp_ssim` | 0.9706 | 0.0177258 | 0.974845 | 0.99103 | higher | Local-window 3D amplitude SSIM. |
| `real_support_iou` | 0.854146 | 0.0669528 | 0.869552 | 0.93299 | higher | Post-processed support intersection-over-union. |
| `real_support_dice` | 0.91985 | 0.0412474 | 0.930225 | 0.965333 | higher | Post-processed support Dice score. |
| `real_support_volume_ratio` | 1.07604 | 0.067581 | 1.05771 | 1.20942 | near 1 | Predicted/true support volume ratio. |
| `real_phase_mae_true_support` | 0.204646 | 0.0802342 | 0.185405 | 0.361017 | lower | Wrapped phase MAE on the post-processed true support. |

## AutoPhaseNN-Compatible Fixed Metrics

The `FT` row is retained for file-format compatibility but is a reprojection identity, not an independently predicted quantity.

| Group | Metric | Mean |
|---|---|---:|
| FT (reprojection only) | L1 | 7.06712e-06 |
| FT (reprojection only) | MSE | 3.28066e-10 |
| FT (reprojection only) | RMSE | 0.000101625 |
| FT (reprojection only) | RelL1 | 5.9532e-07 |
| Amplitude | L1 | 0.00426837 |
| Amplitude | MSE | 0.00145998 |
| Amplitude | RMSE | 0.0360849 |
| Amplitude | RelL1 | 0.238378 |
| Phase | L1 | 0.204646 |
| Phase | MSE | 0.125065 |
| Phase | RMSE | 0.330222 |
| Phase | RelL1 | 0.335128 |
| Support | L1 | 0.00367349 |
| Support | MSE | 0.00367349 |
| Support | RMSE | 0.0577324 |
| Support | RelL1 | 0.168848 |

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
| `real_amp_l1` | 0.00426837 | Real-space full-volume amplitude L1. Lower is better, but can be small for sparse objects. |
| `real_amp_ssim` | 0.9706 | Local-window 3D amplitude SSIM reported by the paper. Higher is better. |
| `real_amp_global_ssim` | 0.947245 | Global 3D amplitude SSIM-like score. Higher is better. |
| `real_support_iou` | 0.854146 | Intersection-over-union between predicted and true support. Higher is better. |
| `real_support_dice` | 0.91985 | Dice score between predicted and true support. Higher is better. |
| `real_support_pred_fraction` | 0.0242284 | Predicted support fraction in the 64^3 volume; should be close to true fraction. |
| `real_support_volume_ratio` | 1.07604 | pred_support_fraction / true_support_fraction; ideal is near 1. |
| `real_phase_mae_true_support` | 0.204646 | Wrapped phase MAE on the true support. Lower is better. |

### Realspace Diagnostic

| Metric | Mean | Meaning |
|---|---:|---|
| `real_amp_mse` | 0.00145998 | Real-space full-volume amplitude MSE. Lower is better. |
| `real_amp_rmse` | 0.0360849 | Real-space full-volume amplitude RMSE. Lower is better. |
| `real_amp_rel_l1` | 0.238378 | Real-space amplitude L1 normalized by true amplitude sum. Lower is better. |
| `real_support_l1` | 0.00367349 | Binary support mask L1. Lower is better. |
| `real_support_rmse` | 0.0577324 | Binary support mask RMSE. Lower is better. |
| `real_support_true_fraction` | 0.0226217 | True support fraction in the 64^3 volume. |
| `real_phase_mae_intersection` | 0.16961 | Wrapped phase MAE on support intersection. Lower is better. |
| `real_phase_rmse_true_support` | 0.330222 | Wrapped phase RMSE on the true support. Lower is better. |

### Reprojection Identity

| Metric | Mean | Meaning |
|---|---:|---|
| `paper_modulus_mae` | 7.06712e-06 | Measured-modulus reprojection L1; near zero by construction and not comparable to an independently predicted modulus. |
| `relative_l1_modulus` | 5.9532e-07 | Scale-normalized reprojection consistency. |
| `chi2_modulus` | 8.21975e-14 | Reprojection chi-square consistency. |
| `pearson_corr` | 1 | Measured/reprojected modulus correlation. |
| `voxel_mse` | 3.28066e-10 | Measured/reprojected modulus voxel MSE. |
| `voxel_rmse` | 0.000101625 | Measured/reprojected modulus voxel RMSE. |

### Timing

| Metric | Mean | Meaning |
|---|---:|---|
| `inference_ms` | 14.2482 | Mean model forward latency assigned to each sample. |

## Interpretation Notes

- `twin_aligned` uses the target only to choose between the two reciprocal-phase signs treated as equivalent by the published WCA loss.
- `phase_wca`, amplitude SSIM, support IoU/Dice, and real-space phase error are the meaningful quality indicators for this architecture.
- A support volume ratio far from one indicates that reciprocal-phase errors spread reconstructed energy outside the object.

## Files

- `evaluation_results.json`: configuration, provenance, distributions, and per-sample metrics.
- `evaluation_samples.csv`: one row per sample.
- `evaluation_summary.md`: this readable summary.
- `evaluation.log`: execution log.
