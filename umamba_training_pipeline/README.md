# UMamba Training Pipeline

This folder is a hybrid training entrypoint for the current debugging work:

- Model construction follows `oyys_lcrc_train_singleGPU.py`.
- Data loading follows `autophasenn_training_pipeline`, with memmap loading, optional RAM caching, optional `scale_i`, and overfit mode.
- Checkpoint loading follows `oyys_lcrc_train_singleGPU.py`: it accepts partial model matches, skips incompatible keys, and resumes optimizer/scheduler state only when the checkpoint is fully compatible.
- Scheduler options follow `oyys_lcrc_train_singleGPU.py`, but the default is `plateau`, matching the AutoPhaseNN overfit probe.
- Loss functions follow `autophasenn_training_pipeline/losses.py`.
- The local `UMambaEnc_3d.py` uses the same postprocessing contract as the AutoPhaseNN pipeline: hard support threshold, masked object as the second output, and `torch.abs(FFT)` far-field modulus.
- The default UMamba overfit probe enables `--center-pad-last-upsample true`: the last decoder stage center-pads the 32^3 feature to 64^3 and skips the outermost 64^3 skip connection, to test the AutoPhaseNN-like center prior.

## Quick Overfit Probe

```bash
bash umamba_training_pipeline/test_overfit_small.sh
```

The script mirrors `autophasenn_training_pipeline/test_overfit_small.sh`: it trains
from scratch on the first `SAMPLES` training samples, validates on the same sample
pool, evaluates `best_model.pt` on that train subset, and visualizes only samples
from that same overfit pool.

Current defaults are chosen for the center-pad ablation:

```bash
SAMPLES=100
EPOCHS=500
BATCH_SIZE=16
LOSS_TYPE=l1
LOSS_SCOPE=diff
SUPPORT_WEIGHT=0.0
LR_SCHEDULER=plateau
GRAD_CLIP=0.0
CENTER_PAD_LAST_UPSAMPLE=true
```

Switch models with:

```bash
--model-name autophasenn
--model-name autophasenn_relu
```

Resume or load pretrained weights with:

```bash
--checkpoint /path/to/checkpoint.pt
```

Use `--reset-optimizer` if you want checkpoint weights but fresh optimizer and scheduler state.

## Evaluation And Visualization

```bash
python umamba_training_pipeline/evaluate.py \
  --model-name umamba \
  --checkpoint /path/to/checkpoint.pt \
  --batch-size 4
```

```bash
python umamba_training_pipeline/visualize_postprocessed.py \
  --model-name umamba \
  --checkpoint /path/to/checkpoint.pt \
  --num-samples 5 \
  --seed 42
```

## Loss Difference Summary

`oyys_lcrc_train_singleGPU.py` defines many metric-like losses, but training currently uses only:

```python
criterion = nn.L1Loss()
loss = criterion(y, ft_images)
```

Validation in `oyys` reports raw L1, normalized chi2 (`loss_sq`), normalized L1 (`loss_mae`), SmoothL1, PCC loss, and combined chi2/PCC losses.

`autophasenn_training_pipeline/losses.py` defines the same family more explicitly and with batch-mean reduction:

- `l1` / `paper_mae`: raw diffraction-modulus L1.
- `chi2` / `sq`: per-sample normalized squared error.
- `relative_l1` / `mae`: per-sample normalized L1.
- `log`: normalized MSE in `log10(x + 1)`.
- `pcc`: `1 - PearsonCorr`.
- `comb`: average of chi2 and PCC loss.
- `comb2`: average of sqrt(chi2) and PCC loss.
- `comb_log`: weighted chi2/PCC/log combination.

This new trainer defaults to the pipeline's actual default behavior: `--loss-type l1 --loss-scope diff`.
