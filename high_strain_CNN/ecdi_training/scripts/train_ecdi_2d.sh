#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-/data_ssd/oyys/eCDI}"
RUN_NAME="${RUN_NAME:-ecdi_2d512_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${PROJECT_DIR}/artifacts/training/ecdi_2d/${RUN_NAME}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data_ssd/oyys/high_strain_cnn/pipeline_output_2d}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"

if [[ ! -f "${DATA_DIR}/manifests/splits_training_with_val.json" ]]; then
  printf 'Missing split manifest: %s\n' "${DATA_DIR}/manifests/splits_training_with_val.json" >&2
  exit 1
fi
mkdir -p "${RUN_DIR}"
cd -- "${PROJECT_DIR}"

nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u \
  -m ecdi_training.run_pipeline \
  --data-dir "${DATA_DIR}" \
  --run-name "${RUN_NAME}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --phase-batch-size "${PHASE_BATCH_SIZE:-12}" \
  --refiner-batch-size "${REFINER_BATCH_SIZE:-12}" \
  --num-workers "${NUM_WORKERS:-4}" \
  "$@" > "${RUN_DIR}/launcher.log" 2>&1 < /dev/null &

PIPELINE_PID=$!
printf '%s\n' "${PIPELINE_PID}" > "${RUN_DIR}/pipeline.pid"
printf 'Pipeline submitted\nPID: %s\nRun: %s\nLog: %s/launcher.log\nCheckpoints: %s/%s_{phase,refiner}\n' \
  "${PIPELINE_PID}" "${RUN_DIR}" "${RUN_DIR}" "${CHECKPOINT_ROOT}" "${RUN_NAME}"
