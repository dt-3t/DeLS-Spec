#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

DELS_OPTION_REQUESTED=0
[[ -n "${DELS_ALPHA+x}" ]] && DELS_OPTION_REQUESTED=1
[[ -n "${DELS_BETA+x}" ]] && DELS_OPTION_REQUESTED=1
[[ -n "${DELS_BASELINE_PATH+x}" ]] && DELS_OPTION_REQUESTED=1
[[ -n "${DELS_BASELINE_SMOOTHING+x}" ]] && DELS_OPTION_REQUESTED=1
BENCHMARK_CODE_ROOT="${BENCHMARK_CODE_ROOT:-${SCRIPT_DIR}/code}"
PYTHON="${PYTHON:-python}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-8B-DFlash-b16}"
TASKS="${TASKS:-gsm8k:128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
BLOCK_SIZE="${BLOCK_SIZE:-}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
DOMINO_GUIDANCE_ALPHA="${DOMINO_GUIDANCE_ALPHA:-1.0}"
DELS_MERGED_LM_HEAD="${DELS_MERGED_LM_HEAD:-}"
DELS_LOCAL_HEAD="${DELS_LOCAL_HEAD:-}"
DELS_ALPHA="${DELS_ALPHA:-1.0}"
DELS_BETA="${DELS_BETA:-1.0}"
DELS_BASELINE_PATH="${DELS_BASELINE_PATH:-checkpoints/dels-spec/qwen3-8b/loss_mask_unigram.pt}"
DELS_BASELINE_SMOOTHING="${DELS_BASELINE_SMOOTHING:-1.0}"
POSITION_MIX_JOINT="${POSITION_MIX_JOINT:-}"
MASTER_PORT="${MASTER_PORT:-29601}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/outputs/hf_$(date +%Y%m%d_%H%M%S)}"

if (( DELS_OPTION_REQUESTED )) && [[ -z "${DELS_LOCAL_HEAD}" ]]; then
  echo "DeLS options were set, but DELS_LOCAL_HEAD is empty." >&2
  echo "Set DELS_LOCAL_HEAD=/path/to/local_head.pt or markov_local_head.pt, or unset DELS_ALPHA/DELS_BETA." >&2
  exit 2
fi

DEFAULT_NUM_GPUS=1
VISIBLE_GPU_COUNT=0
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "all" ]]; then
  IFS=',' read -r -a VISIBLE_GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
  for visible_gpu in "${VISIBLE_GPU_ARRAY[@]}"; do
    visible_gpu="${visible_gpu//[[:space:]]/}"
    if [[ -n "${visible_gpu}" && "${visible_gpu}" != "-1" ]]; then
      VISIBLE_GPU_COUNT=$((VISIBLE_GPU_COUNT + 1))
    fi
  done
fi
NUM_GPUS="${NUM_GPUS:-${DEFAULT_NUM_GPUS}}"
if (( VISIBLE_GPU_COUNT > 0 && NUM_GPUS > VISIBLE_GPU_COUNT )); then
  echo "NUM_GPUS=${NUM_GPUS} exceeds visible CUDA devices (${CUDA_VISIBLE_DEVICES}; count=${VISIBLE_GPU_COUNT})." >&2
  echo "Unset NUM_GPUS or set NUM_GPUS=${VISIBLE_GPU_COUNT}." >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

safe_task_name() {
  printf "%s" "$1" | tr '/: ,' '____'
}

IFS=',' read -r -a TASK_ARRAY <<< "${TASKS}"
for task in "${TASK_ARRAY[@]}"; do
  task="${task#"${task%%[![:space:]]*}"}"
  task="${task%"${task##*[![:space:]]}"}"
  if [[ "${task}" != *:* ]]; then
    echo "Invalid task spec ${task}; expected dataset:max_samples." >&2
    exit 2
  fi
  DATASET="${task%:*}"
  MAX_SAMPLES="${task##*:}"
  if [[ -z "${DATASET}" || -z "${MAX_SAMPLES}" ]]; then
    echo "Invalid task spec ${task}; expected dataset:max_samples." >&2
    exit 2
  fi
  DATASET_TAG="$(safe_task_name "${DATASET}")"
  LOG_FILE="${OUT_DIR}/${DATASET_TAG}_t${TEMPERATURE}.log"
  ANSWER_FILE="${OUT_DIR}/${DATASET_TAG}_t${TEMPERATURE}_answers.jsonl"
  EXTRA_ARGS=()
  if [[ -n "${BLOCK_SIZE}" ]]; then
    EXTRA_ARGS+=(--block-size "${BLOCK_SIZE}")
  fi
  case "${SKIP_BASELINE,,}" in
    true|1|yes|y|on)
      EXTRA_ARGS+=(--skip-baseline)
      ;;
    false|0|no|n|off|"")
      ;;
    *)
      echo "Invalid SKIP_BASELINE=${SKIP_BASELINE}; use true or false." >&2
      exit 2
      ;;
  esac
  JOINT_MODE_COUNT=0
  [[ -n "${POSITION_MIX_JOINT}" ]] && JOINT_MODE_COUNT=$((JOINT_MODE_COUNT + 1))
  [[ -n "${DELS_LOCAL_HEAD}" ]] && JOINT_MODE_COUNT=$((JOINT_MODE_COUNT + 1))
  if (( JOINT_MODE_COUNT > 1 )); then
    echo "Set only one of POSITION_MIX_JOINT and DELS_LOCAL_HEAD." >&2
    exit 2
  fi
  if [[ -n "${POSITION_MIX_JOINT}" ]]; then
    EXTRA_ARGS+=(--position-mix-joint-path "${POSITION_MIX_JOINT}")
  fi
  if [[ -n "${DELS_LOCAL_HEAD}" ]]; then
    EXTRA_ARGS+=(--dels-local-head-path "${DELS_LOCAL_HEAD}")
    EXTRA_ARGS+=(--dels-alpha "${DELS_ALPHA}")
    EXTRA_ARGS+=(--dels-beta "${DELS_BETA}")
    EXTRA_ARGS+=(--dels-baseline-path "${DELS_BASELINE_PATH}")
    EXTRA_ARGS+=(--dels-baseline-smoothing "${DELS_BASELINE_SMOOTHING}")
    if [[ -n "${DELS_MERGED_LM_HEAD}" ]]; then
      EXTRA_ARGS+=(--dels-merged-lm-head-path "${DELS_MERGED_LM_HEAD}")
    fi
  fi

  echo "dataset=${DATASET} max_samples=${MAX_SAMPLES} temperature=${TEMPERATURE} block_size=${BLOCK_SIZE:-auto} skip_baseline=${SKIP_BASELINE} domino_guidance_alpha=${DOMINO_GUIDANCE_ALPHA} dels_local_head=${DELS_LOCAL_HEAD} dels_alpha=${DELS_ALPHA} dels_beta=${DELS_BETA} dels_rnn_input=block_prefix dels_baseline_path=${DELS_BASELINE_PATH} dels_baseline_smoothing=${DELS_BASELINE_SMOOTHING} position_mix_joint=${POSITION_MIX_JOINT}" | tee "${LOG_FILE}"

  "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${BENCHMARK_CODE_ROOT}/benchmark.py" \
    --dataset "${DATASET}" \
    --max-samples "${MAX_SAMPLES}" \
    --model-name-or-path "${TARGET_MODEL}" \
    --draft-name-or-path "${DRAFT_MODEL}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --domino-guidance-alpha "${DOMINO_GUIDANCE_ALPHA}" \
    --use-bias \
    --use-graph \
    "${EXTRA_ARGS[@]}" \
    --answer-file "${ANSWER_FILE}" 2>&1 | tee -a "${LOG_FILE}"
done

echo "Wrote outputs to ${OUT_DIR}"
