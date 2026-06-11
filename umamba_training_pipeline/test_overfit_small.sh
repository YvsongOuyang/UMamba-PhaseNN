#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_DIR="${DATA_DIR:-/data_ssd/oyys/autophasenn}"
SAMPLES="${SAMPLES:-100}"
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
LR="${LR:-1e-3}"
LOSS_TYPE="${LOSS_TYPE:-l1}"
LOSS_SCOPE="${LOSS_SCOPE:-diff}"
SUPPORT_WEIGHT="${SUPPORT_WEIGHT:-0.0}"
LR_SCHEDULER="${LR_SCHEDULER:-plateau}"
MIN_LR="${MIN_LR:-1e-6}"
GAMMA="${GAMMA:-0.5}"
PATIENCE="${PATIENCE:-5}"
GRAD_CLIP="${GRAD_CLIP:-0.0}"
CENTER_PAD_LAST_UPSAMPLE="${CENTER_PAD_LAST_UPSAMPLE:-false}"
DROP_LAST_SKIP="${DROP_LAST_SKIP:-false}"
CENTER_MASK_OUTPUT="${CENTER_MASK_OUTPUT:-true}"
CENTER_MASK_SIZE="${CENTER_MASK_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"
MODEL_NAME="${MODEL_NAME:-umamba}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}_centermask${CENTER_MASK_SIZE}_overfit${SAMPLES}_${LOSS_TYPE}_fromscratch_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-./umamba_training_pipeline/output/${RUN_NAME}}"

echo "UMamba pipeline overfit test"
echo "  model_name:      ${MODEL_NAME}"
echo "  data_dir:        ${DATA_DIR}"
echo "  output_dir:      ${OUT_DIR}"
echo "  samples:         ${SAMPLES}"
echo "  epochs:          ${EPOCHS}"
echo "  loss_type:       ${LOSS_TYPE}"
echo "  loss_scope:      ${LOSS_SCOPE}"
echo "  support_weight:  ${SUPPORT_WEIGHT}"
echo "  lr_scheduler:    ${LR_SCHEDULER}"
echo "  min_lr:          ${MIN_LR}"
echo "  gamma:           ${GAMMA}"
echo "  patience:        ${PATIENCE}"
echo "  grad_clip:       ${GRAD_CLIP}"
echo "  center_pad:      ${CENTER_PAD_LAST_UPSAMPLE}"
echo "  drop_last_skip:  ${DROP_LAST_SKIP}"
echo "  center_mask:     ${CENTER_MASK_OUTPUT}"
echo "  center_mask_size:${CENTER_MASK_SIZE}"
echo

python umamba_training_pipeline/train.py \
  --model-name "${MODEL_NAME}" \
  --center-pad-last-upsample "${CENTER_PAD_LAST_UPSAMPLE}" \
  --drop-last-skip "${DROP_LAST_SKIP}" \
  --center-mask-output "${CENTER_MASK_OUTPUT}" \
  --center-mask-size "${CENTER_MASK_SIZE}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUT_DIR}" \
  --from-scratch \
  --checkpoint "" \
  --overfit-samples "${SAMPLES}" \
  --cache-data \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --lr "${LR}" \
  --loss-type "${LOSS_TYPE}" \
  --loss-scope "${LOSS_SCOPE}" \
  --support-weight "${SUPPORT_WEIGHT}" \
  --lr-scheduler "${LR_SCHEDULER}" \
  --min-lr "${MIN_LR}" \
  --gamma "${GAMMA}" \
  --patience "${PATIENCE}" \
  --grad-clip "${GRAD_CLIP}" \
  --save-every 50 \
  --save-model 1

python umamba_training_pipeline/evaluate.py \
  --model-name "${MODEL_NAME}" \
  --center-pad-last-upsample "${CENTER_PAD_LAST_UPSAMPLE}" \
  --drop-last-skip "${DROP_LAST_SKIP}" \
  --center-mask-output "${CENTER_MASK_OUTPUT}" \
  --center-mask-size "${CENTER_MASK_SIZE}" \
  --checkpoint "${OUT_DIR}/best_model.pt" \
  --data-dir "${DATA_DIR}" \
  --data-diff train_diff.npy \
  --data-real train_real.npy \
  --num-samples "${SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers 0 \
  --device "${DEVICE}" \
  --output-json "${OUT_DIR}/evaluation_train_subset.json"

python umamba_training_pipeline/visualize_postprocessed.py \
  --model-name "${MODEL_NAME}" \
  --center-pad-last-upsample "${CENTER_PAD_LAST_UPSAMPLE}" \
  --drop-last-skip "${DROP_LAST_SKIP}" \
  --center-mask-output "${CENTER_MASK_OUTPUT}" \
  --center-mask-size "${CENTER_MASK_SIZE}" \
  --checkpoint "${OUT_DIR}/best_model.pt" \
  --data-dir "${DATA_DIR}" \
  --data-diff train_diff.npy \
  --data-real train_real.npy \
  --dataset-size "${SAMPLES}" \
  --overfit-samples "${SAMPLES}" \
  --num-samples 5 \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --output-png "${OUT_DIR}/visualization_train_subset.png"

python - "${OUT_DIR}/history.json" "${OUT_DIR}/evaluation_train_subset.json" <<'PY'
import json
import sys
from pathlib import Path

history_path = Path(sys.argv[1])
eval_path = Path(sys.argv[2])

print()
print("UMamba pipeline overfit summary")
if history_path.exists():
    history = json.loads(history_path.read_text(encoding="utf-8"))
    train = history.get("train", [])
    val = history.get("val", [])
    if train:
        print("  first train:", train[0])
        print("  last train: ", train[-1])
    if val:
        print("  first val:  ", val[0])
        print("  last val:   ", val[-1])

if eval_path.exists():
    report = json.loads(eval_path.read_text(encoding="utf-8"))
    mean = report.get("mean", {})
    keys = [
        "paper_modulus_mae",
        "relative_l1_modulus",
        "chi2_modulus",
        "pearson_corr",
        "real_amp_l1",
        "real_amp_global_ssim",
        "real_support_iou",
        "real_support_dice",
        "real_support_pred_fraction",
        "real_support_volume_ratio",
        "real_phase_mae_true_support",
    ]
    for key in keys:
        if key in mean:
            print(f"  {key}: {mean[key]:.6g}")
PY

echo
echo "Done. Outputs:"
echo "  ${OUT_DIR}/history.json"
echo "  ${OUT_DIR}/evaluation_train_subset.json"
echo "  ${OUT_DIR}/visualization_train_subset.png"
