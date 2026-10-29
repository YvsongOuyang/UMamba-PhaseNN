#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-/data_ssd/oyys/bcdi_2d/dataset}"
RUN_NAME="${RUN_NAME:-bcdi_2d512_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${PROJECT_DIR}/artifacts/training/bcdi_2d/${RUN_NAME}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data_ssd/oyys/bcdi_2d/checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"

if [[ ! -f "${DATA_DIR}/training_config.json" ]]; then
  printf 'Prepare the BCDI data first. Missing: %s/training_config.json\n' "${DATA_DIR}" >&2
  exit 1
fi
mkdir -p "${RUN_DIR}"
cd -- "${PROJECT_DIR}"

nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u \
  -m bcdi_training.run_pipeline \
  --data-config "${DATA_DIR}/training_config.json" \
  --run-name "${RUN_NAME}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --phase-batch-size "${PHASE_BATCH_SIZE:-16}" \
  --refiner-batch-size "${REFINER_BATCH_SIZE:-16}" \
  --num-workers "${NUM_WORKERS:-4}" \
  "$@" > "${RUN_DIR}/launcher.log" 2>&1 < /dev/null &

PIPELINE_PID=$!
printf '%s\n' "${PIPELINE_PID}" > "${RUN_DIR}/pipeline.pid"
printf 'BCDI pipeline submitted\nPID: %s\nLog: %s/launcher.log\nCheckpoints: %s/%s_{phase,refiner}\n' \
  "${PIPELINE_PID}" "${RUN_DIR}" "${CHECKPOINT_ROOT}" "${RUN_NAME}"
