# Official TensorFlow Model on Reproduced Simulations

## Protocol

- Samples: 72 (36 calibration, 36 held-out evaluation)
- Model parameters: 143,759,937
- Selected support threshold: `0.15`
- Selection rule: calibration IoU within 0.001 of the maximum, then support-volume ratio closest to one.
- Target support: exact support saved by the simulation generator.

## Held-out Evaluation

| Metric | Mean | Std | Median | 5% | 95% |
|---|---:|---:|---:|---:|---:|
| amplitude_mae | 0.0541082 | 0.0140291 | 0.0504914 | 0.0365753 | 0.0753659 |
| amplitude_nrmse | 0.113785 | 0.0199731 | 0.108437 | 0.087504 | 0.143194 |
| amplitude_scale | 31.8525 | 50.7536 | 14.0067 | 1.71272 | 140.055 |
| phase_offset_rad | -0.393183 | 1.66977 | -0.541442 | -2.55717 | 2.48823 |
| phase_wca | 0.930546 | 0.0832456 | 0.962443 | 0.703953 | 0.988215 |
| phase_wca_direct | 0.947923 | 0.0723604 | 0.974534 | 0.768242 | 0.990546 |
| phase_wca_inverted | 0.939743 | 0.0831534 | 0.970251 | 0.70556 | 0.99707 |
| support_dice | 0.71222 | 0.0876123 | 0.710518 | 0.558662 | 0.831764 |
| support_iou | 0.559955 | 0.101828 | 0.551011 | 0.388192 | 0.712022 |
| support_precision | 0.754623 | 0.102015 | 0.758069 | 0.610457 | 0.89373 |
| support_recall | 0.713464 | 0.165095 | 0.701789 | 0.434003 | 0.94499 |
| support_volume_ratio | 0.986021 | 0.345135 | 0.988501 | 0.529765 | 1.45097 |
| wrapped_phase_mae_rad | 1.00516 | 0.345286 | 1.15923 | 0.433461 | 1.46171 |

## Files

- `evaluation_results.json`: full provenance, statistics, and per-sample rows.
- `evaluation_samples.csv`: held-out per-sample metrics at the selected threshold.
- `threshold_sweep.csv`: calibration and evaluation means for every threshold.
- `evaluation.log`: TensorFlow inference and evaluation progress.
