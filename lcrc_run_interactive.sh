#!/bin/bash
set -euo pipefail

# 指定只使用第一块显卡（如果机器上有多张卡但你只用一张4090）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Which python?"
command -v python

echo "Current Directory: "
pwd

SCRIPT="${SCRIPT_DIR}/oyys_lcrc_train_singleGPU.py"

data_path=/data_ssd/oyys/autophasenn
data_train_diff='train_diff.npy'
data_train_real='train_real.npy'
data_val_diff='val_diff.npy'
data_val_real='val_real.npy'


output='logging.txt'

echo "Dataset path $data_path"

note="supervised on defect free data"
note="${note// /_}"
echo "$note"

fp16=false  
unsupervise=false
use_down_stride=false
use_up_stride=false
reset_optimizer=true

train_size='25000'
num_samples_train=25000
num_samples_val=5000
train_perc=0.9
batch_size=8
n_epoch=100
Initlr=4e-4
min_lr=1e-5
T=0.1
scale_I=1  # NOTE: current memmap DataLoader keeps this for compatibility but does not normalize by scale_I.
lr_type='cosine' #'cosine', 'clr', 'step', 'plateau'
optim_type='adam' #'adam', 'adamw'
loss_type='l1'  # Training intentionally uses fixed L1Loss in oyys_lcrc_train_singleGPU.py.

save_model=10
n_workers=8  # <--- 修改了这里，单机推荐 8-32 之间
checkpoint='/data_ssd/oyys/autophasenn/Unsupfalse_Dfalse_Ufalse_T0.1_comb2_batch8_plateau_Init1e-3_adam_scale1/best_model.pt'  # 断点续训模型路径，留空表示不使用断点续训、
# checkpoint='/data_ssd/oyys/autophasenn/Unsupfalse_Dfalse_Ufalse_T0.1_comb2_batch8_plateau_Init1e-3_adam_scale1/best_model.pt'
#checkpoint='/home/oyys/code/AutoPhaseNN/PyTorch/AutoPhase/best_model.pth'     
#checkpoint='/' #断点续训模型路径，留空表示不使用断点续训
result_path="/data_ssd/oyys/autophasenn/Unsup${unsupervise}_D${use_down_stride}_U${use_up_stride}_T${T}_${loss_type}_batch${batch_size}_${lr_type}_Init${Initlr}_${optim_type}_scale${scale_I}"
log_file="${result_path}/${output}"

model_name='umamba' # autophasenn or umamba

echo "Saving path $result_path"
mkdir -p "$result_path"

extra_args=()
if [[ "$fp16" == true ]]; then
    extra_args+=(--fp16)
fi
if [[ "$reset_optimizer" == true ]]; then
    extra_args+=(--reset_optimizer)
fi

# 直接使用 python 运行即可
python "$SCRIPT" \
    --notes "$note" \
    --save_model "$save_model" \
    --checkpoint "$checkpoint" \
    --model_name "$model_name" \
    --device cuda \
    --loss_type "$loss_type" \
    --OutputFolder "$result_path" \
    --DataFolder "$data_path" \
    --num_workers "$n_workers" \
    --batch_size "$batch_size" \
    --epoch "$n_epoch" \
    --optim_type "$optim_type" \
    --Initlr "$Initlr" \
    --min_lr "$min_lr" \
    "${extra_args[@]}" \
    --unsupervise "$unsupervise" \
    --use_down_stride "$use_down_stride" \
    --use_up_stride "$use_up_stride" \
    --train_size "$train_size" \
    --train_perc "$train_perc" \
    --lr_type "$lr_type" \
    --T "$T" \
    --data_train_diff "$data_train_diff" \
    --data_train_real "$data_train_real" \
    --data_val_diff "$data_val_diff" \
    --data_val_real "$data_val_real" \
    --num_samples_train "$num_samples_train" \
    --num_samples_val "$num_samples_val" \
    --scale_I "$scale_I" 2>&1 | tee -a "$log_file"
