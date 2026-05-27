# UMamba Training Pipeline

This folder is a hybrid training entrypoint for the current debugging work:

- Model construction follows `oyys_lcrc_train_singleGPU.py`.
- Data loading follows `autophasenn_training_pipeline`, with memmap loading, optional RAM caching, optional `scale_i`, and overfit mode.
- Checkpoint loading follows `oyys_lcrc_train_singleGPU.py`: it accepts partial model matches, skips incompatible keys, and resumes optimizer/scheduler state only when the checkpoint is fully compatible.
- Scheduler options follow `oyys_lcrc_train_singleGPU.py`, but the default is `none`, matching the pipeline default.
- Loss functions follow `autophasenn_training_pipeline/losses.py`.
- The local `UMambaEnc_3d.py` uses the same postprocessing contract as the AutoPhaseNN pipeline: hard support threshold, masked object as the second output, and `torch.abs(FFT)` far-field modulus.

## Quick 100-Sample Overfit Probe

```bash
python umamba_training_pipeline/train.py \
  --model-name umamba \
  --from-scratch \
  --loss-type l1 \
  --loss-scope diff \
  --overfit-samples 100 \
  --cache-data \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-3 \
  --lr-type none \
  --debug-output-delta \
  --debug-grad-norm
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
