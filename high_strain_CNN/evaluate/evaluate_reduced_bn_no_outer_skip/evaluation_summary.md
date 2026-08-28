# HighStrain AutoPhaseNN Evaluation Summary

## Run

| Item | Value |
|---|---|
| Checkpoint | `/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/high_strain_cnn/high_strain_reduced_bn_no_outer_skip_scratch_bs16_lr1e-3_20260825_153347/checkpoint_best.pt` |
| Checkpoint epoch | 130 |
| Model variant | `reduced_bn_no_outer_skip` |
| Model parameters | 39,121,665 |
| Project version | `1.5.0` |
| Git commit | `dc891e8a40455144a64af34464fabdeb62af4c25` |
| Samples | 5000 |
| Batch size | 16 |
| Device | `cuda` |
| GPU | NVIDIA GeForce RTX 4090 |
| Ambiguity mode | `twin_aligned` |
| Support threshold | 0.3 |
| Phase unwrap workers | 8 |
| Evaluation wall time | 377.333 s |
| Mean inference | 22.7976 ms/sample |
| Throughput | 43.8643 samples/s |

## Metric Interpretation

The real-space metrics below use the same AutoPhaseNN post-processing and metric implementations and are directly comparable. Reciprocal modulus metrics are not model-quality measurements for this method: reconstruction explicitly reuses the measured modulus, so their near-zero errors only test FFT/reprojection consistency.

## Comparable Quality Metrics

| Metric | Mean | Std | P50 | P95 | Better | Meaning |
|---|---:|---:|---:|---:|---|---|
| `phase_wca` | 0.63433 | 0.104447 | 0.624683 | 0.83129 | lower | Reciprocal-phase WCA objective used for training. |
| `real_amp_l1` | 0.00357609 | 0.0018721 | 0.00318119 | 0.00711896 | lower | Full-volume post-processed amplitude L1. |
| `real_amp_ssim` | 0.975945 | 0.0136075 | 0.978959 | 0.991774 | higher | Local-window 3D amplitude SSIM. |
| `real_support_iou` | 0.880243 | 0.043969 | 0.887518 | 0.934565 | higher | Post-processed support intersection-over-union. |
| `real_support_dice` | 0.935702 | 0.0259041 | 0.940407 | 0.966176 | higher | Post-processed support Dice score. |
| `real_support_volume_ratio` | 1.04796 | 0.0391319 | 1.04038 | 1.12095 | near 1 | Predicted/true support volume ratio. |
| `real_phase_mae_true_support` | 0.185617 | 0.0649952 | 0.17008 | 0.305624 | lower | Wrapped phase MAE on the post-processed true support. |

## AutoPhaseNN-Compatible Fixed Metrics

The `FT` row is retained for file-format compatibility but is a reprojection identity, not an independently predicted quantity.

| Group | Metric | Mean |
|---|---|---:|
| FT (reprojection only) | L1 | 7.06215e-06 |
| FT (reprojection only) | MSE | 3.27204e-10 |
| FT (reprojection only) | RMSE | 0.00010162 |
| FT (reprojection only) | RelL1 | 5.94919e-07 |
| Amplitude | L1 | 0.00357609 |
| Amplitude | MSE | 0.00110851 |
| Amplitude | RMSE | 0.0317837 |
| Amplitude | RelL1 | 0.19826 |
| Phase | L1 | 0.185617 |
| Phase | MSE | 0.106772 |
| Phase | RMSE | 0.308922 |
| Phase | RelL1 | 0.309561 |
| Support | L1 | 0.00289407 |
| Support | MSE | 0.00289407 |
| Support | RMSE | 0.0518624 |
| Support | RelL1 | 0.132427 |

## Mean Metrics by Group

### Phase Retrieval

| Metric | Mean | Meaning |
|---|---:|---|
| `phase_wca` | 0.63433 | Published symmetry-aware reciprocal-phase WCA loss. |
| `phase_wca_direct` | 0.801798 | WCA error against the direct reciprocal phase. |
| `phase_wca_inverted` | 0.806474 | WCA error against the conjugate/twin phase. |
| `twin_flip_selected` | 0.4944 | Fraction indicator for evaluation-time conjugate/twin sign selection. |

### Realspace Primary

| Metric | Mean | Meaning |
|---|---:|---|
| `real_amp_l1` | 0.00357609 | Real-space full-volume amplitude L1. Lower is better, but can be small for sparse objects. |
| `real_amp_ssim` | 0.975945 | Local-window 3D amplitude SSIM reported by the paper. Higher is better. |
| `real_amp_global_ssim` | 0.959733 | Global 3D amplitude SSIM-like score. Higher is better. |
| `real_support_iou` | 0.880243 | Intersection-over-union between predicted and true support. Higher is better. |
| `real_support_dice` | 0.935702 | Dice score between predicted and true support. Higher is better. |
| `real_support_pred_fraction` | 0.0236296 | Predicted support fraction in the 64^3 volume; should be close to true fraction. |
| `real_support_volume_ratio` | 1.04796 | pred_support_fraction / true_support_fraction; ideal is near 1. |
| `real_phase_mae_true_support` | 0.185617 | Wrapped phase MAE on the true support. Lower is better. |

### Realspace Diagnostic

| Metric | Mean | Meaning |
|---|---:|---|
| `real_amp_mse` | 0.00110851 | Real-space full-volume amplitude MSE. Lower is better. |
| `real_amp_rmse` | 0.0317837 | Real-space full-volume amplitude RMSE. Lower is better. |
| `real_amp_rel_l1` | 0.19826 | Real-space amplitude L1 normalized by true amplitude sum. Lower is better. |
| `real_support_l1` | 0.00289407 | Binary support mask L1. Lower is better. |
| `real_support_rmse` | 0.0518624 | Binary support mask RMSE. Lower is better. |
| `real_support_true_fraction` | 0.0226217 | True support fraction in the 64^3 volume. |
| `real_phase_mae_intersection` | 0.152737 | Wrapped phase MAE on support intersection. Lower is better. |
| `real_phase_rmse_true_support` | 0.308922 | Wrapped phase RMSE on the true support. Lower is better. |

### Reprojection Identity

| Metric | Mean | Meaning |
|---|---:|---|
| `paper_modulus_mae` | 7.06215e-06 | Measured-modulus reprojection L1; near zero by construction and not comparable to an independently predicted modulus. |
| `relative_l1_modulus` | 5.94919e-07 | Scale-normalized reprojection consistency. |
| `chi2_modulus` | 8.20385e-14 | Reprojection chi-square consistency. |
| `pearson_corr` | 1 | Measured/reprojected modulus correlation. |
| `voxel_mse` | 3.27204e-10 | Measured/reprojected modulus voxel MSE. |
| `voxel_rmse` | 0.00010162 | Measured/reprojected modulus voxel RMSE. |

### Timing

| Metric | Mean | Meaning |
|---|---:|---|
| `inference_ms` | 22.7976 | Mean model forward latency assigned to each sample. |

## Interpretation Notes

- `twin_aligned` uses the target only to choose between the two reciprocal-phase signs treated as equivalent by the published WCA loss.
- `phase_wca`, amplitude SSIM, support IoU/Dice, and real-space phase error are the meaningful quality indicators for this architecture.
- A support volume ratio far from one indicates that reciprocal-phase errors spread reconstructed energy outside the object.

## Files

- `evaluation_results.json`: configuration, provenance, distributions, and per-sample metrics.
- `evaluation_samples.csv`: one row per sample.
- `evaluation_summary.md`: this readable summary.
- `evaluation.log`: execution log.
