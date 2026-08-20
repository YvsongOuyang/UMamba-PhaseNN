# high_strain_CNN AutoPhaseNN evaluation

## Run

| Item | Value |
|---|---|
| Checkpoint | `/data_ssd/oyys/autophasenn/autophasenn_pipeline_output/high_strain_cnn/high_strain_reduced_scratch_bs16_lr1e-4_20260819_171256/checkpoint_best.pt` |
| Model variant | `reduced` |
| Model parameters | 39,160,897 |
| Project version | `1.2.0` |
| Git commit | `e92d862adb5682c516e216c6b924d0b92afb2fa2` |
| Samples | 5000 |
| Device | `cuda` |
| Ambiguity mode | `twin_aligned` |
| Support threshold | 0.1 |

## AutoPhaseNN-scale metrics

| Group | Metric | Mean |
|---|---|---:|
| FT | L1 | 7.07273e-06 |
| FT | MSE | 3.26623e-10 |
| FT | RMSE | 0.000101618 |
| FT | RelL1 | 5.95828e-07 |
| Amplitude | L1 | 0.0107959 |
| Amplitude | MSE | 0.00286559 |
| Amplitude | RMSE | 0.0514436 |
| Amplitude | RelL1 | 0.606284 |
| Phase | L1 | 0.864446 |
| Phase | MSE | 1.14231 |
| Phase | RMSE | 0.942882 |
| Phase | RelL1 | 1.58014 |
| Support | L1 | 0.0389045 |
| Support | MSE | 0.0389045 |
| Support | RMSE | 0.190152 |
| Support | RelL1 | 1.70823 |

## Phase retrieval diagnostics

- WCA loss: `0.766761`
- Twin/conjugate selection fraction: `0.3486`
- Mean model inference: `19.8631 ms/sample`

## Interpretation

The real-space amplitude, phase, and support metrics use the same official AutoPhaseNN post-processing and metric functions. Reciprocal-space modulus metrics are expected to be nearly zero because reconstruction explicitly reuses the measured modulus; WCA is the meaningful reciprocal-phase metric.

`twin_aligned` uses the real-space target only during evaluation to select between the two signs explicitly treated as equivalent by the published WCA loss. Use `raw` to measure the uncorrected model output.
