#!/usr/bin/env bash
set -euo pipefail

# 预训练权重文件路径；默认假设 best_model.pt 在当前 AutoPhaseNN 文件夹下。
CHECKPOINT="/data_ssd/oyys/autophasenn/Unsupfalse_Dfalse_Ufalse_T0.1_comb2_batch8_plateau_Init1e-3_adam_scale1/best_model.pt"

# 验证数据所在文件夹；里面应包含 val_diff.npy 和 val_real.npy。
DATA_FOLDER="/data_ssd/oyys/autophasenn"

# 验证衍射数据文件名，和 DATA_FOLDER 拼成完整路径。
DATA_VAL_DIFF="val_diff.npy"

# 验证真实空间复数数据文件名，和 DATA_FOLDER 拼成完整路径。
DATA_VAL_REAL="val_real.npy"

# 验证集样本数；需要和 val_diff.npy / val_real.npy memmap 的真实样本数一致。
NUM_SAMPLES_VAL=5000

# Batch size；显存不够就调小，比如 1 或 2。
BATCH_SIZE=4

# DataLoader 进程数；Windows/Git Bash 下建议先用 0，Linux 服务器可适当调大。
NUM_WORKERS=0

# 输出目录；会保存 validation_summary.json、validation_batches.csv、每层热力图 PNG 和聚合 npy。
OUTPUT_DIR="./autophasenn_heatmap_results"

# 运行设备；有 CUDA 时用 cuda，没有 GPU 可改成 cpu。
DEVICE="cuda"

# 是否启用 fp16；0 表示关闭，1 表示加上 --fp16。
USE_FP16=0

# 最大验证 batch 数；0 表示跑完整个验证集，调成 1/2 可快速试跑。
MAX_BATCHES=100

# 保存前多少个 batch 的单次热力图；完整验证集的 aggregate 综合热力图总会保存。
SAVE_BATCH_HEATMAPS=3

# 按 notebook 末尾 plot6 风格保存多少个样本的 FT/Amp/Phase 对比切片图。
# 会输出 FT input/Pred FT/Diff FT，Support/True Amp/Pred Amp，True ph/Pred Ph/Diff Ph 三张图。
SAVE_RECON_PLOTS=2

# 输入空间里的 z 切片位置；例如 64x64x64 时填 32 表示中心附近。
# 如果设为空字符串，则使用下面的 SLICE_FRACTION。
SLICE_INDEX=32

# 相对 z 切片位置；仅当 SLICE_INDEX 为空时生效。0.5 表示中心切片。
SLICE_FRACTION=0.5

# 3D 数据边长；当前模型和数据默认是 64x64x64。
SHAPE=64

# SupportLayer 阈值相关参数，和训练脚本保持一致。
T=0.1

# 初始卷积通道数，需和训练模型结构一致。
NCONV=32

# Encoder block 数，需和训练模型结构一致。
N_BLOCKS=4

# 衍射强度缩放参数，需和训练/验证时一致。
SCALE_I=1

FP16_ARG=()
if [[ "${USE_FP16}" == "1" ]]; then
  FP16_ARG=(--fp16)
fi

SLICE_ARG=(--slice_fraction "${SLICE_FRACTION}")
if [[ -n "${SLICE_INDEX}" ]]; then
  SLICE_ARG=(--slice_index "${SLICE_INDEX}")
fi

python test_autophasenn_heatmaps.py \
  --checkpoint "${CHECKPOINT}" \
  --data_folder "${DATA_FOLDER}" \
  --data_val_diff "${DATA_VAL_DIFF}" \
  --data_val_real "${DATA_VAL_REAL}" \
  --num_samples_val "${NUM_SAMPLES_VAL}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --output_dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --max_batches "${MAX_BATCHES}" \
  --save_batch_heatmaps "${SAVE_BATCH_HEATMAPS}" \
  --save_recon_plots "${SAVE_RECON_PLOTS}" \
  "${SLICE_ARG[@]}" \
  --shape "${SHAPE}" \
  --T "${T}" \
  --nconv "${NCONV}" \
  --n_blocks "${N_BLOCKS}" \
  --scale_I "${SCALE_I}" \
  "${FP16_ARG[@]}"
