# Reduced HighStrain result analysis

## Scope

This report compares the 5000-sample `reduced` HighStrain evaluation with the
current AutoPhaseNN baseline. Both evaluations use support threshold `0.1` and
the same real-space phase unwrapping, phase-offset removal, center-of-mass
alignment, and metric implementations. HighStrain uses `twin_aligned` sign
selection, which is an evaluation-only oracle allowed by its WCA formulation.

## Comparable results

| Metric | HighStrain reduced | AutoPhaseNN baseline | Observation |
|---|---:|---:|---|
| Amplitude L1 | 0.010796 | 0.005788 | 1.865x higher error |
| Amplitude SSIM | 0.855792 | 0.957022 | 0.101229 lower |
| Support IoU | 0.385794 | 0.746818 | 0.361024 lower |
| Support Dice | 0.551927 | 0.845206 | 0.293279 lower |
| Support volume ratio | 2.675543 | 1.013121 | predicted support is much too large |
| Phase MAE on true support | 0.864446 rad | 0.239835 rad | 3.604x higher error |
| Phase MAE on support intersection | 0.861945 rad | 0.126754 rad | 6.800x higher error |
| Inference time | 19.8631 ms/sample | 6.3625 ms/sample | 3.122x slower |

The global amplitude SSIM values are similar (`0.876324` versus `0.875431`),
but this aggregate diagnostic is dominated by the large zero-valued background.
The local-window 3D amplitude SSIM above is the meaningful primary SSIM metric.

## Reciprocal-space interpretation

The mean phase WCA is `0.766761`; for this loss, zero is best. This confirms
that the learned reciprocal phase is still inaccurate. The reported reciprocal
modulus MAE (`7.07e-06`), relative L1 (`5.96e-07`), and correlation (approximately
`1.0`) are not model-quality results: reconstruction explicitly combines the
predicted phase with the measured modulus. These values only verify numerical
FFT consistency and must not be compared with AutoPhaseNN's independently
predicted diffraction modulus.

## Verification and conclusion

An oracle round trip using the target reciprocal phase with the measured
modulus produced amplitude L1 about `1.75e-10`, phase MAE about `3e-8`, support
IoU `1.0`, and support volume ratio `1.0` after the official post-processing.
The FFT convention, shifts, support construction, and real-space metric path are
therefore consistent.

The current discrepancy is a model result rather than an evaluation arithmetic
error. In particular, reciprocal-phase errors spread reconstructed energy beyond
the true object, which agrees with the support volume ratio of `2.68` and low
support overlap. This single run cannot distinguish insufficient optimization
from a capacity or data-distribution mismatch. The next useful diagnostic is to
compare train and validation WCA histories; high values on both indicate
underfitting, while a large gap indicates generalization failure.
