#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-/data/wyz}"
DATASET_NAME="$(basename -- "${DATA_DIR}")"
LOG_DIR="${PROJECT_DIR}/artifacts/generation/bcdi_2d/${DATASET_NAME}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WORKERS="${WORKERS:-4}"
GENERATION_CONFIG="${GENERATION_CONFIG:-${PROJECT_DIR}/bcdi_training/configs/generation_512.json}"
mkdir -p "${LOG_DIR}"
cd -- "${PROJECT_DIR}"

if [[ -f "${LOG_DIR}/generate.pid" ]]; then
  previous_pid="$(<"${LOG_DIR}/generate.pid")"
  if [[ "${previous_pid}" =~ ^[0-9]+$ ]] && kill -0 "${previous_pid}" 2>/dev/null; then
    printf 'Generation may still be running with PID %s; inspect it before launching another writer.\n' "${previous_pid}" >&2
    exit 1
  fi
fi

nohup "${PYTHON_BIN}" -u -m bcdi_training.generate_dataset \
  --config "${GENERATION_CONFIG}" --output-dir "${DATA_DIR}" \
  --log-dir "${LOG_DIR}" --workers "${WORKERS}" \
  "$@" >> "${LOG_DIR}/launcher.log" 2>&1 < /dev/null &

GENERATION_PID=$!
printf '%s\n' "${GENERATION_PID}" > "${LOG_DIR}/generate.pid"
printf 'BCDI generation submitted\nPID: %s\nData: %s\nLog: %s/launcher.log\nTraining config after completion: %s/training_config.json\n' \
  "${GENERATION_PID}" "${DATA_DIR}" "${LOG_DIR}" "${DATA_DIR}"
