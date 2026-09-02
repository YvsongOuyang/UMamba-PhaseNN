# Pytorch Model on Reproduced Simulations

## Protocol

- Samples: 5907 (3383 calibration, 2524 held-out evaluation)
- Model parameters: 39,121,665
- Selected support threshold: `0.275`
- Selection rule: calibration IoU within 0.001 of the maximum, then support-volume ratio closest to one.
- Target support: exact boolean support saved by simulation generator.

## Held-out Evaluation

| Metric | Mean | Std | Median | 5% | 95% |
|---|---:|---:|---:|---:|---:|
| amplitude_mae | 2.31602e+07 | 8.71907e+06 | 2.19525e+07 | 1.11907e+07 | 3.96045e+07 |
| amplitude_nrmse | 0.0375292 | 0.0155764 | 0.0348819 | 0.0168241 | 0.0670835 |
| amplitude_scale | 1.20583e+10 | 8.8602e+09 | 9.34234e+09 | 2.77264e+09 | 3.05822e+10 |
| phase_offset_rad | 0.0134162 | 1.86835 | 0.0271976 | -2.87359 | 2.83133 |
| phase_wca | 0.470102 | 0.197215 | 0.465463 | 0.155634 | 0.794288 |
| phase_wca_direct | 0.62116 | 0.267862 | 0.618865 | 0.187728 | 0.978797 |
| phase_wca_inverted | 0.706411 | 0.248653 | 0.777812 | 0.236709 | 0.982811 |
| support_dice | 0.883201 | 0.074342 | 0.896327 | 0.751565 | 0.967937 |
| support_iou | 0.797884 | 0.10708 | 0.812131 | 0.602006 | 0.937866 |
| support_precision | 0.899858 | 0.0519871 | 0.905872 | 0.806537 | 0.971139 |
| support_recall | 0.874944 | 0.10623 | 0.902708 | 0.668726 | 0.986257 |
| support_volume_ratio | 0.974301 | 0.118962 | 1.00699 | 0.736895 | 1.09626 |
| wrapped_phase_mae_rad | 0.316844 | 0.209398 | 0.263588 | 0.0850477 | 0.731283 |

## Files

- `evaluation_results.json`: full provenance, statistics, and per-sample rows.
- `evaluation_samples.csv`: held-out per-sample metrics at the selected threshold.
- `threshold_sweep.csv`: calibration and evaluation means for every threshold.
- `evaluation.log`: Pytorch inference and evaluation progress.
