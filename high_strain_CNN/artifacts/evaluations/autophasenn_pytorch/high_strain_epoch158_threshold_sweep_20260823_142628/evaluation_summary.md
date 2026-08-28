# HighStrain AutoPhaseNN Evaluation Summary

## Run

| Item | Value |
|---|---|
| Checkpoint | `/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/high_strain_cnn/high_strain_reduced_centered_resume120_to240_20260822_172432/checkpoint_best.pt` |
| Checkpoint epoch | 158 |
| Model variant | `reduced` |
| Model parameters | 39,160,897 |
| Project version | `1.4.0` |
| Git commit | `bc8672c4bbdeef1ac4289c987de33ea7ba3374d3` |
| Samples | 5000 |
| Batch size | 16 |
| Device | `cuda` |
| GPU | NVIDIA GeForce RTX 4090 |
| Ambiguity mode | `twin_aligned` |
| Support threshold | 0.1 |
| Phase unwrap workers | 8 |
| Evaluation wall time | 412.436 s |
| Mean inference | 23.0501 ms/sample |
| Throughput | 43.3838 samples/s |

## Metric Interpretation

The real-space metrics below use the same AutoPhaseNN post-processing and metric implementations and are directly comparable. Reciprocal modulus metrics are not model-quality measurements for this method: reconstruction explicitly reuses the measured modulus, so their near-zero errors only test FFT/reprojection consistency.

## Comparable Quality Metrics

| Metric | Mean | Std | P50 | P95 | Better | Meaning |
|---|---:|---:|---:|---:|---|---|
| `phase_wca` | 0.667847 | 0.139546 | 0.650185 | 0.912103 | lower | Reciprocal-phase WCA objective used for training. |
| `real_amp_l1` | 0.00722868 | 0.00407634 | 0.00623501 | 0.0150871 | lower | Full-volume post-processed amplitude L1. |
| `real_amp_ssim` | 0.905985 | 0.0582974 | 0.922102 | 0.972233 | higher | Local-window 3D amplitude SSIM. |
| `real_support_iou` | 0.513093 | 0.088852 | 0.514086 | 0.656474 | higher | Post-processed support intersection-over-union. |
| `real_support_dice` | 0.673584 | 0.0788979 | 0.679071 | 0.792616 | higher | Post-processed support Dice score. |
| `real_support_volume_ratio` | 1.95706 | 0.358196 | 1.8944 | 2.63577 | near 1 | Predicted/true support volume ratio. |
| `real_phase_mae_true_support` | 0.649567 | 0.468788 | 0.521433 | 1.53263 | lower | Wrapped phase MAE on the post-processed true support. |

## Support Threshold Sweep

The primary support threshold remains unchanged. Sweep-selected operating points are validation diagnostics and do not replace the headline metrics above.

| Threshold | Primary | Amp L1 | Amp SSIM | Support IoU | Support Dice | Volume ratio | Phase MAE |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0.05 |  | 0.0133462 | 0.713955 | 0.194895 | 0.32276 | 5.44104 | 1.38804 |
| 0.075 |  | 0.0091299 | 0.848905 | 0.367564 | 0.532392 | 2.79286 | 0.973053 |
| 0.1 | yes | 0.00722868 | 0.905985 | 0.513093 | 0.673584 | 1.95706 | 0.649567 |
| 0.125 |  | 0.00620972 | 0.933256 | 0.62124 | 0.762526 | 1.59315 | 0.455615 |
| 0.15 |  | 0.0055728 | 0.948035 | 0.699053 | 0.819686 | 1.39712 | 0.347198 |
| 0.175 |  | 0.00514276 | 0.956817 | 0.754513 | 0.857425 | 1.27692 | 0.286755 |
| 0.2 |  | 0.00485027 | 0.962205 | 0.791326 | 0.881232 | 1.20565 | 0.249533 |
| 0.225 |  | 0.00463762 | 0.965702 | 0.816686 | 0.897118 | 1.15766 | 0.228331 |
| 0.25 |  | 0.00448248 | 0.96797 | 0.833649 | 0.907504 | 1.1251 | 0.215478 |
| 0.275 |  | 0.00435959 | 0.969566 | 0.846006 | 0.914975 | 1.09766 | 0.208688 |
| 0.3 |  | 0.00426806 | 0.970603 | 0.854146 | 0.91985 | 1.07604 | 0.204658 |
| 0.35 |  | 0.0041529 | 0.971639 | 0.863152 | 0.925237 | 1.0438 | 0.202039 |
| 0.4 |  | 0.00410453 | 0.971665 | 0.863633 | 0.925596 | 1.03569 | 0.196656 |

- Best mean-IoU threshold: `0.4`.
- Threshold with mean volume ratio closest to one: `0.4`.

## AutoPhaseNN-Compatible Fixed Metrics

The `FT` row is retained for file-format compatibility but is a reprojection identity, not an independently predicted quantity.

| Group | Metric | Mean |
|---|---|---:|
| FT (reprojection only) | L1 | 7.0672e-06 |
| FT (reprojection only) | MSE | 3.2787e-10 |
| FT (reprojection only) | RMSE | 0.000101624 |
| FT (reprojection only) | RelL1 | 5.95323e-07 |
| Amplitude | L1 | 0.00722868 |
| Amplitude | MSE | 0.00182963 |
| Amplitude | RMSE | 0.040542 |
| Amplitude | RelL1 | 0.392725 |
| Phase | L1 | 0.649567 |
| Phase | MSE | 0.717026 |
| Phase | RMSE | 0.720279 |
| Phase | RelL1 | 1.20913 |
| Support | L1 | 0.0235345 |
| Support | MSE | 0.0235345 |
| Support | RMSE | 0.14677 |
| Support | RelL1 | 0.992921 |

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
| `real_amp_l1` | 0.00722868 | Real-space full-volume amplitude L1. Lower is better, but can be small for sparse objects. |
| `real_amp_ssim` | 0.905985 | Local-window 3D amplitude SSIM reported by the paper. Higher is better. |
| `real_amp_global_ssim` | 0.929631 | Global 3D amplitude SSIM-like score. Higher is better. |
| `real_support_iou` | 0.513093 | Intersection-over-union between predicted and true support. Higher is better. |
| `real_support_dice` | 0.673584 | Dice score between predicted and true support. Higher is better. |
| `real_support_pred_fraction` | 0.0461295 | Predicted support fraction in the 64^3 volume; should be close to true fraction. |
| `real_support_volume_ratio` | 1.95706 | pred_support_fraction / true_support_fraction; ideal is near 1. |
| `real_phase_mae_true_support` | 0.649567 | Wrapped phase MAE on the true support. Lower is better. |

### Realspace Diagnostic

| Metric | Mean | Meaning |
|---|---:|---|
| `real_amp_mse` | 0.00182963 | Real-space full-volume amplitude MSE. Lower is better. |
| `real_amp_rmse` | 0.040542 | Real-space full-volume amplitude RMSE. Lower is better. |
| `real_amp_rel_l1` | 0.392725 | Real-space amplitude L1 normalized by true amplitude sum. Lower is better. |
| `real_support_l1` | 0.0235345 | Binary support mask L1. Lower is better. |
| `real_support_rmse` | 0.14677 | Binary support mask RMSE. Lower is better. |
| `real_support_true_fraction` | 0.0234351 | True support fraction in the 64^3 volume. |
| `real_phase_mae_intersection` | 0.64534 | Wrapped phase MAE on support intersection. Lower is better. |
| `real_phase_rmse_true_support` | 0.720279 | Wrapped phase RMSE on the true support. Lower is better. |

### Reprojection Identity

| Metric | Mean | Meaning |
|---|---:|---|
| `paper_modulus_mae` | 7.0672e-06 | Measured-modulus reprojection L1; near zero by construction and not comparable to an independently predicted modulus. |
| `relative_l1_modulus` | 5.95323e-07 | Scale-normalized reprojection consistency. |
| `chi2_modulus` | 8.21624e-14 | Reprojection chi-square consistency. |
| `pearson_corr` | 1 | Measured/reprojected modulus correlation. |
| `voxel_mse` | 3.2787e-10 | Measured/reprojected modulus voxel MSE. |
| `voxel_rmse` | 0.000101624 | Measured/reprojected modulus voxel RMSE. |

### Timing

| Metric | Mean | Meaning |
|---|---:|---|
| `inference_ms` | 23.0501 | Mean model forward latency assigned to each sample. |

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
