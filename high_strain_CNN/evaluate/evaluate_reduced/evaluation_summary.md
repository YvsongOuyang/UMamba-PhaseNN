# HighStrain AutoPhaseNN Evaluation Summary

## Run

| Item | Value |
|---|---|
| Checkpoint | `/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/high_strain_cnn/high_strain_reduced_scratch_bs16_lr1e-4_20260819_171256/checkpoint_best.pt` |
| Checkpoint epoch | 60 |
| Model variant | `reduced` |
| Model parameters | 39,160,897 |
| Project version | `1.2.0` |
| Git commit | `313c23dec5072bbd84078c85ef0303e280ab9c80` |
| Samples | 5000 |
| Batch size | 16 |
| Device | `cuda` |
| GPU | NVIDIA GeForce RTX 4090 |
| Ambiguity mode | `twin_aligned` |
| Support threshold | 0.1 |
| Phase unwrap workers | 8 |
| Evaluation wall time | 350.185 s |
| Mean inference | 19.377 ms/sample |
| Throughput | 51.6075 samples/s |

## Metric Interpretation

The real-space metrics below use the same AutoPhaseNN post-processing and metric implementations and are directly comparable. Reciprocal modulus metrics are not model-quality measurements for this method: reconstruction explicitly reuses the measured modulus, so their near-zero errors only test FFT/reprojection consistency.

## Comparable Quality Metrics

| Metric | Mean | Std | P50 | P95 | Better | Meaning |
|---|---:|---:|---:|---:|---|---|
| `phase_wca` | 0.766761 | 0.124782 | 0.756932 | 0.972891 | lower | Reciprocal-phase WCA objective used for training. |
| `real_amp_l1` | 0.010796 | 0.00578825 | 0.00960705 | 0.0204777 | lower | Full-volume post-processed amplitude L1. |
| `real_amp_ssim` | 0.855792 | 0.0734722 | 0.872615 | 0.945952 | higher | Local-window 3D amplitude SSIM. |
| `real_support_iou` | 0.385794 | 0.0793022 | 0.386658 | 0.513821 | higher | Post-processed support intersection-over-union. |
| `real_support_dice` | 0.551927 | 0.0853135 | 0.557683 | 0.678839 | higher | Post-processed support Dice score. |
| `real_support_volume_ratio` | 2.67555 | 0.790353 | 2.5345 | 3.70083 | near 1 | Predicted/true support volume ratio. |
| `real_phase_mae_true_support` | 0.864478 | 0.53013 | 0.7549 | 1.91471 | lower | Wrapped phase MAE on the post-processed true support. |

## AutoPhaseNN-Compatible Fixed Metrics

The `FT` row is retained for file-format compatibility but is a reprojection identity, not an independently predicted quantity.

| Group | Metric | Mean |
|---|---|---:|
| FT (reprojection only) | L1 | 7.07249e-06 |
| FT (reprojection only) | MSE | 3.2635e-10 |
| FT (reprojection only) | RMSE | 0.000101616 |
| FT (reprojection only) | RelL1 | 5.9581e-07 |
| Amplitude | L1 | 0.010796 |
| Amplitude | MSE | 0.00286567 |
| Amplitude | RMSE | 0.0514443 |
| Amplitude | RelL1 | 0.606294 |
| Phase | L1 | 0.864478 |
| Phase | MSE | 1.1418 |
| Phase | RMSE | 0.942893 |
| Phase | RelL1 | 1.5801 |
| Support | L1 | 0.0389045 |
| Support | MSE | 0.0389045 |
| Support | RMSE | 0.190152 |
| Support | RelL1 | 1.70823 |

## Mean Metrics by Group

### Phase Retrieval

| Metric | Mean | Meaning |
|---|---:|---|
| `phase_wca` | 0.766761 | Published symmetry-aware reciprocal-phase WCA loss. |
| `phase_wca_direct` | 0.844577 | WCA error against the direct reciprocal phase. |
| `phase_wca_inverted` | 0.907416 | WCA error against the conjugate/twin phase. |
| `twin_flip_selected` | 0.3486 | Fraction indicator for evaluation-time conjugate/twin sign selection. |

### Realspace Primary

| Metric | Mean | Meaning |
|---|---:|---|
| `real_amp_l1` | 0.010796 | Real-space full-volume amplitude L1. Lower is better, but can be small for sparse objects. |
| `real_amp_ssim` | 0.855792 | Local-window 3D amplitude SSIM reported by the paper. Higher is better. |
| `real_amp_global_ssim` | 0.876321 | Global 3D amplitude SSIM-like score. Higher is better. |
| `real_support_iou` | 0.385794 | Intersection-over-union between predicted and true support. Higher is better. |
| `real_support_dice` | 0.551927 | Dice score between predicted and true support. Higher is better. |
| `real_support_pred_fraction` | 0.0616046 | Predicted support fraction in the 64^3 volume; should be close to true fraction. |
| `real_support_volume_ratio` | 2.67555 | pred_support_fraction / true_support_fraction; ideal is near 1. |
| `real_phase_mae_true_support` | 0.864478 | Wrapped phase MAE on the true support. Lower is better. |

### Realspace Diagnostic

| Metric | Mean | Meaning |
|---|---:|---|
| `real_amp_mse` | 0.00286567 | Real-space full-volume amplitude MSE. Lower is better. |
| `real_amp_rmse` | 0.0514443 | Real-space full-volume amplitude RMSE. Lower is better. |
| `real_amp_rel_l1` | 0.606294 | Real-space amplitude L1 normalized by true amplitude sum. Lower is better. |
| `real_support_l1` | 0.0389045 | Binary support mask L1. Lower is better. |
| `real_support_rmse` | 0.190152 | Binary support mask RMSE. Lower is better. |
| `real_support_true_fraction` | 0.0234351 | True support fraction in the 64^3 volume. |
| `real_phase_mae_intersection` | 0.861981 | Wrapped phase MAE on support intersection. Lower is better. |
| `real_phase_rmse_true_support` | 0.942893 | Wrapped phase RMSE on the true support. Lower is better. |

### Reprojection Identity

| Metric | Mean | Meaning |
|---|---:|---|
| `paper_modulus_mae` | 7.07249e-06 | Measured-modulus reprojection L1; near zero by construction and not comparable to an independently predicted modulus. |
| `relative_l1_modulus` | 5.9581e-07 | Scale-normalized reprojection consistency. |
| `chi2_modulus` | 8.19935e-14 | Reprojection chi-square consistency. |
| `pearson_corr` | 1 | Measured/reprojected modulus correlation. |
| `voxel_mse` | 3.2635e-10 | Measured/reprojected modulus voxel MSE. |
| `voxel_rmse` | 0.000101616 | Measured/reprojected modulus voxel RMSE. |

### Timing

| Metric | Mean | Meaning |
|---|---:|---|
| `inference_ms` | 19.377 | Mean model forward latency assigned to each sample. |

## Interpretation Notes

- `twin_aligned` uses the target only to choose between the two reciprocal-phase signs treated as equivalent by the published WCA loss.
- `phase_wca`, amplitude SSIM, support IoU/Dice, and real-space phase error are the meaningful quality indicators for this architecture.
- A support volume ratio far from one indicates that reciprocal-phase errors spread reconstructed energy outside the object.

## Files

- `evaluation_results.json`: configuration, provenance, distributions, and per-sample metrics.
- `evaluation_samples.csv`: one row per sample.
- `evaluation_summary.md`: this readable summary.
- `evaluation.log`: execution log.
