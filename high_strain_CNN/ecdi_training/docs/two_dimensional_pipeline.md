# 512 x 512 Two-Stage Pipeline

The native 2D path is `ecdi_training`. The existing 3D training, author simulation,
TensorFlow reference and checkpoints remain available through their existing
entrypoints. Shared backbones select 2D operators with `spatial_dims=2`;
default construction still uses 3D and preserves the old state-dictionary keys.

For the e-CDI generator's `observed_counts` + complex exit-wave HDF5 dataset,
use `ecdi_training/configs/ecdi_2d.json` and [the dedicated two-stage workflow](../README.md).
The headered NPY route below remains available.

## Architecture

The default phase U-Net is `reduced_bn_no_outer_skip`: BatchNorm2d,
LeakyReLU(0.2), the same dilated convolution branches, SAME padding and inner
skip connections, with no full-resolution skip. Its resolution progression is
512 -> 256 -> 128 -> 64 -> 32 -> 16, with 1024 bottleneck channels.
Its output is reciprocal phase, not intensity or amplitude.
The default 2D phase backbone has 13,109,217 parameters, compared with
39,121,665 for the corresponding 3D BN/no-outer-skip backbone.

The measured modulus and predicted phase form a complex spectrum. Centered
inverse FFT2 yields a complex `[B,1,512,512]` initial object. Complex MoMamba uses
2D complex convolutions and transposed convolutions, separate real/imaginary
BatchNorm2d, shared complex dropout masks, and the existing positional top-k
expert router. Sparse Mamba sequences have 65536, 16384, 4096 and 1024 tokens
at the 256, 128, 64 and 32 resolutions. The output remains an identity-initialized
complex residual. Production uses official `mamba-ssm`; no substitute mixer is
used for training.

512^2 and 64^3 contain the same number of pixels/voxels. This does not guarantee
the same training memory or throughput: 2D retains more tokens at deep scales
and distributes convolution activations differently. Start with batch size 1
and measure CUDA memory before increasing it. No GPU memory benchmark has been
performed by the local CPU contract tests.

## Paired Data

Edit `ecdi_training/configs/diffraction_2d.json` or pass `--data-dir`. Each split contains two
standard **headered NPY** files:

| File | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `<split>_diff.npy` | `[N,512,512]` | float32 | Centered measured intensity or modulus |
| `<split>_real.npy` | `[N,512,512]` | complex64 | Corresponding centered-grid complex object |

Set `diffraction_kind` explicitly to `intensity` (default) or `modulus`.
For intensity, the code takes its square root for reconstruction. For modulus,
it squares it before intensity preprocessing. Phase U-Net input uses per-image
min-max normalized `log1p(intensity)`, and inverse FFT uses unit-maximum modulus.
Complex targets are amplitude-COM centered by a Fourier phase ramp before phase
supervision and complex-object supervision. Preprocessing runs on the training
device after loading, not in the Dataset workers. Workers read only paired arrays.

Existing raw AutoPhaseNN 64^3 memmaps are not 2D data, even though their per-sample
element counts match 512^2. Neither reshaping them nor pairing independently
selected real-space and diffraction slices gives a physical 2D training pair.
A 2D target must have corresponding diffraction under FFT2 (plus explicitly
modelled measurement effects). No 2D dataset is synthesized by these entrypoints.

## Training

Run from `high_strain_CNN`. Phase training requires only the existing PyTorch
dependencies. Refiner training additionally requires the official dependency in
`ecdi_training/requirements/mamba.txt` on a compatible Linux CUDA environment.

```bash
python -u -m ecdi_training.train --stage phase --run-name phase_2d512_scratch

python -u -m ecdi_training.train --stage refiner \
  --phase-checkpoint /data_ssd/oyys/high_strain_cnn/pipeline_output_2d/phase_2d512_scratch/checkpoint_best.pt \
  --run-name momamba_2d512_stage2
```

Default phase training: Adam, LR 1e-3, plateau scheduler, 100 epochs, WCA loss.
Default second stage: frozen phase U-Net (including its BN), Adam, LR 2e-4,
cosine scheduler, 50 epochs, ambiguity-aware real MAE + imaginary MAE.
Optional physics-loss weight defaults to zero. These are starting configurations,
not an experimentally established optimal budget for a new 2D dataset. Both
stages use float32/complex64. `--batch-size`, `--epochs`, `--learning-rate`,
`--num-workers`, `--max-batches-per-epoch` and `--resume` are available.
`--epochs` is the final epoch number when resuming.

Logs/config/history go to `artifacts/training/ecdi_training/<run-name>`;
large checkpoints go to `pipeline_output_2d/<run-name>` under the SSD root above.
3D checkpoints are rejected, and the 2D phase network must be trained first.

## Evaluation

```bash
python -m ecdi_training.evaluate \
  --checkpoint /data_ssd/oyys/high_strain_cnn/pipeline_output_2d/momamba_2d512_stage2/checkpoint_best.pt \
  --split test --support-threshold 0.3
```

Outputs: `metrics.json`, `per_sample.csv` and up to four sample NPZ files in
`artifacts/evaluations/ecdi_training/<run-name>_<split>`. Reported metrics include
complex NRMSE, component MAE, reciprocal WCA, Fourier modulus L1/RelL1/MSE/RMSE,
amplitude L1/RelL1/MSE/RMSE/PSNR/local 2D SSIM, wrapped support-phase MAE/RMSE,
and support IoU/Dice/volume ratio. Both initial and refined outputs are evaluated
for the refiner stage. Real-space comparisons align translation, scale, global
phase and twin ambiguity. Fourier metrics use the full object without true-support
masking. The initial object's Fourier modulus matches the measurement by
construction, so its FT metrics are a consistency reference.

The 2D training and evaluation conventions are saved in native checkpoints;
comparisons with earlier 3D experiments must account for dimensionality, data
semantics, alignment and threshold differences.
