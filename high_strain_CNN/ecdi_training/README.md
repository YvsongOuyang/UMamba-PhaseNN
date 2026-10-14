# e-CDI Exit-Wave Two-Stage Training

This workflow trains the existing native 512 x 512 U-Net + complex MoMamba model
on `deliverables/ecdi_10k_dataset_generator` output. It reads HDF5 directly;
no NPY export, regeneration or new random data partition is required.

## Folder Layout

```text
high_strain_CNN/ecdi_training/
  train.py, run_pipeline.py       staged training and controller
  model.py                       2D adapters for the shared backbones
  data.py, ecdi_data.py           paired NPY and e-CDI HDF5 reading
  validate_data.py                split and Fourier-pair checks
  evaluate.py, visualize.py       metrics and comparison figures
  configs/                       dataset/training configurations
  requirements/                  2D and optional Mamba dependencies
  scripts/train_ecdi_2d.sh        background launcher
  tests/                         2D and HDF5 contract tests
  docs/                          generic NPY pipeline documentation
```

Run module commands from the parent `high_strain_CNN` directory as
`python -m ecdi_training.<entrypoint>`. Model and physics helpers are reused
from `high_strain_CNN/pytorch_autophasenn`; keep the parent project checkout
alongside this folder. This is an organized subproject, not a separate copy
of the shared model implementation.

## Dataset Contract

The default root is `/data_ssd/oyys/eCDI`:

```text
data/ecdi_10k.h5
manifests/splits_training_with_val.json
quality/validation_report.json
```

The manifest selects 8500 train, 500 validation and 1000 test indices for a
completed default 10k generation. Small generator smoke datasets can use the
same format with their own counts. The reader checks the full disjoint partition,
HDF5/manifest configuration hashes, image shapes, dtypes, valid flags and the
matching successful generator quality report. It does not silently filter samples.
Paths in Windows-generated relative manifests are portable to Linux.

| Role | HDF5 field | Use |
| --- | --- | --- |
| Measured input | `observed_counts` | Raw Poisson counts, processed on the training device |
| Complex target | `exit_wave_real` + i `exit_wave_imag` | Clean probe-sample exit wave |
| Clean Fourier reference | `ideal_intensity` | Sampled preflight and evaluation only |
| Stored support | `support` | Ignored: the extended-crystal field is a full-grid placeholder |
| Display image | `observed_log` | Ignored: recompute the model's own input from counts |
| Material/index | `metadata_json`, manifest indices | Per-material evaluation and exact sample provenance |

The generator uses `fftshift(fft2(ifftshift(exit_wave)))`, matching the model's
centered FFT2. The target is an exit wave including Gaussian illumination and
multislice scattering, not a material potential or standalone particle density.
The generator's physical settings need not be changed for this training route.

Each DataLoader worker independently opens the HDF5 read-only on first use.
Training reads only counts and exit-wave components, not ideal diffraction or
support. Validation/test evaluation can additionally read ideal diffraction.
No full-dataset RAM/GPU cache is used; the default is four worker processes.

## Training Definitions

1. Train `HighStrainPhaseUNet2D` from scratch. Input is min-max normalized
   `log1p(observed_counts)`. The phase label is the FFT2 phase of the
   amplitude-COM-centered clean exit wave, with its central phase subtracted.
   Loss is the existing symmetry-aware weighted circular-average loss (WCA),
   weighted by the processed measured input.
2. Load the best validation U-Net checkpoint and freeze its weights and BN
   statistics. Combine its phase with unit-maximum `sqrt(observed_counts)` and
   inverse FFT2 to create the real-space initialization. Train only the complex
   MoMamba refiner against the centered clean exit wave using the existing
   ambiguity-aware real MAE + imaginary MAE, with per-image RMS normalization,
   global-phase and direct/twin alignment.

| Stage | Optimizer | Starting LR | Scheduler | Default epochs |
| --- | --- | --- | --- | --- |
| Phase U-Net | Adam | 1e-3 | ReduceLROnPlateau, factor .5, patience 5 | 100 |
| Frozen-U-Net + MoMamba | Adam | 2e-4 | Cosine | 50 |

These are configurable starting budgets. Both stages use float32/complex64.
Batch size defaults to 1 for each stage until CUDA memory is measured. Gradient
clipping is 1.0. The refiner output is initially the identity reconstruction.
No new architecture, attention, loss or hard support projection is introduced.
Physics-loss weight remains zero and no hard measured-modulus projection is used:
the measured modulus is noisy and the clean target need not match it exactly.
The optional physics diagnostic now uses the full object, not a true-support mask.

## Environment

Use the existing Linux CUDA PyTorch training environment, then install the
HDF5/plotting helpers as needed:

```bash
cd /home/oyys/code/UMamba-AutoPhaseNN/high_strain_CNN
python -m pip install -r ecdi_training/requirements/pytorch-2d.txt
python -m pip install --no-build-isolation -r ecdi_training/requirements/mamba.txt
```

The generator's abTEM/CuPy environment is not required to train. Stage 2 uses
official Mamba, not a CPU test substitute. The automatic controller checks CUDA
availability and imports Mamba before committing to the phase training run.

## Run Both Stages in the Background

```bash
cd /home/oyys/code/UMamba-AutoPhaseNN/high_strain_CNN
bash ecdi_training/scripts/train_ecdi_2d.sh
```

The launcher assigns a dated run name, writes a PID and runs:

```text
validate -> phase U-Net -> frozen-U-Net MoMamba -> final test evaluation -> figures
```

Any failed stage stops the sequence. Test indices are not used for training,
scheduler decisions or checkpoint selection. Final test evaluation compares
the U-Net initialization and refined object on the same samples. A dry run only
writes/displays planned commands and does not start model training:

```bash
python -m ecdi_training.run_pipeline --dry-run --run-name ecdi_plan
```

Optional environment overrides for the background launcher:

```bash
DATA_DIR=/data_ssd/oyys/eCDI \
GPU_ID=0 PHASE_BATCH_SIZE=2 REFINER_BATCH_SIZE=1 NUM_WORKERS=4 \
bash ecdi_training/scripts/train_ecdi_2d.sh
```

For an initial short server correctness run, use small prefixes and few epochs,
with a new run name. This checks execution, not reconstruction performance:

```bash
bash ecdi_training/scripts/train_ecdi_2d.sh \
  --phase-epochs 1 --refiner-epochs 1 \
  --num-samples-train 16 --num-samples-val 4 --num-samples-test 4
```

Do not interpret these tiny prefixes as material-balanced experiments. Formal
training uses every indexed training example by default.

## Logs, Checkpoints and Resume

```text
high_strain_CNN/artifacts/training/ecdi_2d/<run>/
  launcher.log                  aggregate stdout/stderr when launched by Bash
  pipeline.log                  controller commands and status
  pipeline.pid                  controller process
  pipeline_status.json          active stage and child PID, or final status
  pipeline_config.json          resolved child commands
  data_validation.json          reader checks plus sampled clean FFT checks
  stages/<run>_phase/            phase console.log, config.json, history.json
  stages/<run>_refiner/          refiner console.log, config.json, history.json
  evaluation/                   metrics, per-sample CSV, sample NPZs and figures

/data_ssd/oyys/high_strain_cnn/pipeline_output_2d/
  <run>_phase/checkpoint_{best,last}.pt
  <run>_refiner/checkpoint_{best,last}.pt
```

Follow the printed `launcher.log` with `tail -f`. Checkpoints contain model,
optimizer, scheduler, configuration, epoch, history and dataset fingerprints.
Existing runs with checkpoints require explicit resume; scratch training cannot
silently overwrite them. Three-dimensional checkpoints are rejected.

Resume only stage 2, keeping the same run name and data:

```bash
python -u -m ecdi_training.run_pipeline \
  --run-name <same-run-name> --start-stage refiner \
  --refiner-resume /data_ssd/oyys/high_strain_cnn/pipeline_output_2d/<same-run-name>_refiner/checkpoint_last.pt
```

Resume stage 1 with `--phase-resume .../checkpoint_last.pt`. `--phase-epochs`
and `--refiner-epochs` mean the final epoch numbers, not extra epochs.
When resuming to a new run name, explicitly manage the source/best-checkpoint
paths; the standard controller assumes the original run directories.
To repeat final evaluation and plots after training, use
`--run-name <same-run-name> --start-stage evaluate`.

## Interpret Evaluation

Evaluation reports complex NRMSE, component MAE, amplitude L1/RelL1/MSE/RMSE,
PSNR, 2D SSIM, wrapped real-space phase errors, reciprocal WCA and Fourier
modulus errors. Phase is also evaluated with amplitude weighting over the full
field. `per_sample.csv` keeps original HDF5 indices and material names;
`metrics_by_material` and `paired_comparison` expose material-specific results
and the fraction of samples improved by refinement.

For the extended Gaussian exit wave, support IoU/Dice at threshold 0.3 measure
the **effective illuminated region**, not recovery of a true compact crystal
boundary. They are secondary diagnostics. Complex NRMSE, real-space phase,
amplitude SSIM and clean Fourier error are more directly relevant.

Distinguish measured consistency (`paper_modulus_mae`, `relative_l1_modulus`)
from clean-reference recovery (`clean_modulus_rel_l1`, `clean_intensity_nrmse`).
The U-Net initialization reuses noisy measured modulus, so it has near-zero
measured Fourier error by construction. A useful denoising refiner can increase
measured Fourier error while decreasing clean-reference and real-space errors.
That change alone does not show diffraction distortion.

Figures show target/initial/refined amplitude, phase, measured diffraction,
object-derived diffraction and aligned complex error. Phase outside the
effective illuminated region is masked for display; it is not a support
constraint applied to the model. Each reconstruction's conjugate-twin ambiguity
is aligned to the same target orientation for readable visual comparisons.

The generator's test is within the same six source structures. Good test scores
here establish same-distribution simulated exit-wave reconstruction, not unseen
material or real-instrument generalization.
