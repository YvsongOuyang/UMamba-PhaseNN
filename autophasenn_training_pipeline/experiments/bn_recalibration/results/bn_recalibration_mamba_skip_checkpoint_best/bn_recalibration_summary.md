# BatchNorm Recalibration Experiment

## Setup

| Item | Value |
|---|---|
| Checkpoint | `/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/mamba_skip_scratch_bs4_lr1e-3_20260823_155916/checkpoint_best.pt` |
| Checkpoint epoch | 70 |
| Model | `mamba_skip` |
| Threshold | 0.1 (checkpoint_metadata) |
| Validation samples | 5000 |
| Calibration samples | 25000 |
| Calibration passes | 1 |
| Calibration batches | 1563 |
| BatchNorm layers | 30 |

Only BatchNorm running mean, running variance, and batch counters are changed. All learned parameters and BatchNorm affine parameters remain untouched.

## Paired Validation Results

| Metric | Normal eval | BN recalibrated | Raw delta | Relative improvement | Samples improved |
|---|---:|---:|---:|---:|---:|
| `paper_modulus_mae` | 4.06393 | 4.19917 | +0.135246 | -3.328% | 12.90% |
| `chi2_modulus` | 0.0294265 | 0.0409883 | +0.0115618 | -39.290% | 12.38% |
| `pearson_corr` | 0.985741 | 0.983411 | -0.0023305 | -0.236% | 23.94% |
| `real_amp_l1` | 0.0061336 | 0.00630832 | +0.000174722 | -2.849% | 75.00% |
| `real_amp_ssim` | 0.955624 | 0.954997 | -0.000626166 | -0.066% | 64.56% |
| `real_support_iou` | 0.735305 | 0.73645 | +0.00114461 | +0.156% | 85.36% |
| `real_support_dice` | 0.846198 | 0.846591 | +0.000393381 | +0.046% | 85.36% |
| `real_support_volume_ratio` | 1.24846 | 1.23521 | -0.0132537 | +4.391% | 97.10% |
| `real_phase_mae_true_support` | 0.38548 | 0.394093 | +0.00861269 | -2.234% | 15.90% |

Positive relative improvement always means better according to the metric's direction. Raw delta is recalibrated minus normal.

## Largest BN Buffer Changes

| Layer | Mean relative L2 change | Variance relative L2 change |
|---|---:|---:|
| `layers.batch_normalization_6` | 0.00844229 | 0.0247511 |
| `layers.batch_normalization_11` | 0.00471538 | 0.0231434 |
| `layers.batch_normalization_7` | 0.0077125 | 0.0200593 |
| `layers.batch_normalization` | 0.0142725 | 0.0141592 |
| `layers.batch_normalization_19` | 0.00485852 | 0.0130982 |
| `phase_fuse8.bn` | 0.00753867 | 0.0103036 |
| `layers.batch_normalization_16` | 0.00575833 | 0.00858659 |
| `phase_fuse16.bn` | 0.00615892 | 0.00819487 |
| `layers.batch_normalization_5` | 0.00813847 | 0.00427933 |
| `layers.batch_normalization_20` | 0.00307645 | 0.00807333 |

## Interpretation

A material and sample-consistent improvement after recalibration supports a checkpoint BN-statistics mismatch. Little change, mixed change, or degradation means stale running statistics are unlikely to be the primary cause of the validation behavior.
