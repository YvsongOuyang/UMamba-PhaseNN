#!/usr/bin/env bash
set -euo pipefail

# Launch from this directory so relative imports and plans_diffraction_3d.json work.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${SCRIPT:-oyys_lcrc_train_multiGPU_DDP_fp16.py}"

echo "Python interpreter:"
"$PYTHON_BIN" -c 'import sys; print(sys.executable)'
echo "Current directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Dataset files. These can be overridden at launch time, for example:
# DATA_PATH=/path/to/data BATCH_SIZE=4 EPOCHS=20 bash lcrc_run_interactive.sh
data_path="${DATA_PATH:-/data_hdd/oyys/autophaseNN/}"
data_train_diff="${DATA_TRAIN_DIFF:-train_diff.npy}"
data_train_real="${DATA_TRAIN_REAL:-train_real.npy}"
data_val_diff="${DATA_VAL_DIFF:-val_diff.npy}"
data_val_real="${DATA_VAL_REAL:-val_real.npy}"

note="${NOTE:-supervised on defect free data}"
note="${note// /_}"

fp16="${FP16:-false}"
unsupervise="${UNSUPERVISE:-false}"
use_down_stride="${USE_DOWN_STRIDE:-false}"
use_up_stride="${USE_UP_STRIDE:-false}"

train_size="${TRAIN_SIZE:-25000}"
num_samples_train="${NUM_SAMPLES_TRAIN:-25000}"
num_samples_val="${NUM_SAMPLES_VAL:-5000}"
batch_size="${BATCH_SIZE:-8}"
n_epoch="${EPOCHS:-100}"
init_lr="${INIT_LR:-2e-5}"
temperature="${T:-0.1}"
scale_I="${SCALE_I:-1}"
loss_type="${LOSS_TYPE:-comb2}"
lr_type="${LR_TYPE:-clr}"
optim_type="${OPTIM_TYPE:-adam}"
save_model="${SAVE_MODEL:-10}"
n_workers="${NUM_WORKERS:-12}"
checkpoint="${CHECKPOINT:-}"
model_name="${MODEL_NAME:-umamba}" # umamba or autophasenn

run_name="Unsup${unsupervise}_D${use_down_stride}_U${use_up_stride}_T${temperature}_${loss_type}_batch${batch_size}_${lr_type}_Init${init_lr}_${optim_type}_scale${scale_I}"
result_path="${RESULT_PATH:-${data_path%/}/runs/${run_name}/}"
log_file="${LOG_FILE:-logging.txt}"

mkdir -p "$result_path"

extra_args=()
if [[ "$fp16" == "true" ]]; then
    extra_args+=(--fp16)
fi
if [[ "$unsupervise" == "true" ]]; then
    extra_args+=(--unsupervise)
fi
if [[ "$use_down_stride" == "true" ]]; then
    extra_args+=(--use_down_stride)
fi
if [[ "$use_up_stride" == "true" ]]; then
    extra_args+=(--use_up_stride)
fi

echo "Dataset path: $data_path"
echo "Saving path: $result_path"
echo "Run note: $note"

"$PYTHON_BIN" "$SCRIPT" \
    --notes "$note" \
    --save_model "$save_model" \
    --checkpoint "$checkpoint" \
    --model_name "$model_name" \
    --plans_file plans_diffraction_3d.json \
    --device cuda \
    --loss_type "$loss_type" \
    --OutputFolder "$result_path" \
    --DataFolder "$data_path" \
    --num_workers "$n_workers" \
    --batch_size "$batch_size" \
    --epoch "$n_epoch" \
    --optim_type "$optim_type" \
    --Initlr "$init_lr" \
    --train_size "$train_size" \
    --lr_type "$lr_type" \
    --T "$temperature" \
    --data_train_diff "$data_train_diff" \
    --data_train_real "$data_train_real" \
    --data_val_diff "$data_val_diff" \
    --data_val_real "$data_val_real" \
    --num_samples_train "$num_samples_train" \
    --num_samples_val "$num_samples_val" \
    --scale_I "$scale_I" \
    "${extra_args[@]}" 2>&1 | tee -a "$result_path/$log_file"
