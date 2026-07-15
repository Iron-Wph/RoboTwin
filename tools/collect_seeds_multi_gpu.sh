#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash tools/collect_seeds_multi_gpu.sh <task_name> <task_config> <total_seeds> <gpu_ids> [seed_start] [output_root]

Arguments:
  task_name    RoboTwin task, for example: place_empty_cup
  task_config  task_config/<task_config>.yml
  total_seeds  Total number of successful seeds across all GPUs
  gpu_ids      Comma-separated physical GPU ids, for example: 0,1,2,3
  seed_start   First candidate seed (default: auto; after the current output seed.txt)
  output_root  Dataset root from the task config (default: ./data)

Environment:
  MAX_ATTEMPTS  Maximum candidates per worker; 0 means unlimited (default: 0)

Each worker writes only seed_workers/gpu_<n>.txt. After all workers finish,
the files are merged into <output_root>/<task_name>/<task_config>/seed.txt;
an existing canonical seed file is preserved and extended.
No trajectory, video, cache, or HDF5 file is intentionally written.
EOF
}

if [[ $# -lt 4 || $# -gt 6 ]]; then
  usage
  exit 2
fi

TASK_NAME="$1"
TASK_CONFIG="$2"
TOTAL_SEEDS="$3"
GPU_IDS="$4"
SEED_START_ARG="${5:-auto}"
OUTPUT_ROOT="${6:-./data}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"

if ! [[ "${TOTAL_SEEDS}" =~ ^[0-9]+$ ]] || [[ "${TOTAL_SEEDS}" -lt 1 ]]; then
  echo "total_seeds must be a positive integer: ${TOTAL_SEEDS}" >&2
  exit 2
fi
if ! [[ "${MAX_ATTEMPTS}" =~ ^[0-9]+$ ]]; then
  echo "MAX_ATTEMPTS must be a non-negative integer: ${MAX_ATTEMPTS}" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
GPU_COUNT="${#GPU_ARRAY[@]}"
if [[ "${GPU_COUNT}" -lt 1 ]]; then
  echo "At least one GPU id is required" >&2
  exit 2
fi

for gpu in "${GPU_ARRAY[@]}"; do
  if ! [[ "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU id: ${gpu}" >&2
    exit 2
  fi
done

if [[ "${SEED_START_ARG}" == "auto" ]]; then
  CANONICAL_SEED_FILE="${OUTPUT_ROOT%/}/${TASK_NAME}/${TASK_CONFIG}/seed.txt"
  if [[ -f "${CANONICAL_SEED_FILE}" ]]; then
    LAST_SEED="$(awk '{for (i=1; i<=NF; i++) if ($i ~ /^-?[0-9]+$/ && ($i+0) > max) max=$i+0} END {print max+0}' "${CANONICAL_SEED_FILE}")"
    SEED_START=$((LAST_SEED + 1))
  else
    SEED_START=0
  fi
else
  if ! [[ "${SEED_START_ARG}" =~ ^[0-9]+$ ]]; then
    echo "seed_start must be a non-negative integer or auto: ${SEED_START_ARG}" >&2
    exit 2
  fi
  SEED_START="${SEED_START_ARG}"
fi

SEED_DIR="${OUTPUT_ROOT%/}/${TASK_NAME}/${TASK_CONFIG}/seed_workers"
mkdir -p "${SEED_DIR}"

declare -a PIDS=()
declare -a WORKER_FILES=()

merge_seed_files() {
  local canonical_seed_file="${OUTPUT_ROOT%/}/${TASK_NAME}/${TASK_CONFIG}/seed.txt"
  local merge_tmp="${canonical_seed_file}.tmp"

  mkdir -p "$(dirname "${canonical_seed_file}")"

  # A worker can fail before creating its file. Read only files that exist so
  # that already accepted seeds can still be merged and resumed later.
  {
    if [[ -f "${canonical_seed_file}" ]]; then
      cat "${canonical_seed_file}"
    fi
    for worker_file in "${WORKER_FILES[@]}"; do
      if [[ -f "${worker_file}" ]]; then
        cat "${worker_file}"
      fi
    done
  } | awk '{for (i=1; i<=NF; i++) if ($i ~ /^-?[0-9]+$/) print $i}' \
    | sort -n -u >"${merge_tmp}"
  mv "${merge_tmp}" "${canonical_seed_file}"

  MERGED_COUNT="$(wc -l <"${canonical_seed_file}" | tr -d '[:space:]')"
  echo "Merged ${MERGED_COUNT} seeds into ${canonical_seed_file}"
}

for ((worker=0; worker<GPU_COUNT; worker++)); do
  # Divide the requested total as evenly as possible. Worker 0 gets the
  # remainder, so the sum of all worker quotas is exactly TOTAL_SEEDS.
  quota=$((TOTAL_SEEDS / GPU_COUNT))
  if [[ "${worker}" -lt $((TOTAL_SEEDS % GPU_COUNT)) ]]; then
    quota=$((quota + 1))
  fi
  if [[ "${quota}" -eq 0 ]]; then
    continue
  fi

  worker_file="${SEED_DIR}/gpu_${worker}.txt"
  worker_start=$((SEED_START + worker))
  WORKER_FILES+=("${worker_file}")

  echo "Launching worker ${worker}: GPU=${GPU_ARRAY[$worker]}, quota=${quota}, seed_start=${worker_start}"
  (
    export CUDA_VISIBLE_DEVICES="${GPU_ARRAY[$worker]}"
    PYTHONWARNINGS=ignore::UserWarning PYTHONUNBUFFERED=1 \
      python -u script/collect_seeds.py "${TASK_NAME}" "${TASK_CONFIG}" \
        --num-seeds "${quota}" \
        --seed-file "${worker_file}" \
        --seed-start "${worker_start}" \
        --seed-step "${GPU_COUNT}" \
        --max-attempts "${MAX_ATTEMPTS}"
  ) >"${worker_file}.log" 2>&1 &
  PIDS+=("$!")
done

failed_workers=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    failed_workers=$((failed_workers + 1))
  fi
done

# Merge partial results even when one worker failed. Every accepted seed is
# already durable in its worker file, so a rerun can continue from there.
merge_seed_files

if [[ "${failed_workers}" -ne 0 ]]; then
  echo "${failed_workers} worker(s) did not reach their quota. See *.log under ${SEED_DIR}." >&2
  exit 1
fi

if [[ "${MERGED_COUNT}" -lt "${TOTAL_SEEDS}" ]]; then
  echo "Only ${MERGED_COUNT}/${TOTAL_SEEDS} seeds are available; rerun to resume." >&2
  exit 1
fi

echo "Canonical seed file now contains ${MERGED_COUNT} unique seeds."
echo "Worker logs: ${SEED_DIR}/gpu_*.log"
