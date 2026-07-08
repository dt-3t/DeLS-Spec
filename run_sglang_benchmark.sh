#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DELS_OPTION_REQUESTED=0
[[ -n "${DELS_ALPHA+x}" ]] && DELS_OPTION_REQUESTED=1
[[ -n "${DELS_BETA+x}" ]] && DELS_OPTION_REQUESTED=1
[[ -n "${DELS_BASELINE_PATH+x}" ]] && DELS_OPTION_REQUESTED=1
[[ -n "${DELS_BASELINE_SMOOTHING+x}" ]] && DELS_OPTION_REQUESTED=1
[[ -n "${DELS_MERGED_LM_HEAD+x}" ]] && DELS_OPTION_REQUESTED=1
[[ -n "${DELS_CANDIDATE_POOL_SIZE+x}" ]] && DELS_OPTION_REQUESTED=1

BENCHMARK_CODE_ROOT="${BENCHMARK_CODE_ROOT:-${SCRIPT_DIR}/code}"
PYTHON="${PYTHON:-python}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-8B-DFlash-b16}"
DELS_LOCAL_HEAD="${DELS_LOCAL_HEAD:-}"
DELS_ALPHA="${DELS_ALPHA:-1.0}"
DELS_BETA="${DELS_BETA:-1.0}"
DELS_BASELINE_PATH="${DELS_BASELINE_PATH:-checkpoints/dels-spec/qwen3-8b/loss_mask_unigram.pt}"
DELS_BASELINE_SMOOTHING="${DELS_BASELINE_SMOOTHING:-1.0}"
DELS_MERGED_LM_HEAD="${DELS_MERGED_LM_HEAD:-}"
DELS_CANDIDATE_POOL_SIZE="${DELS_CANDIDATE_POOL_SIZE:-2048}"
TASKS="${TASKS:-gsm8k:128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
CONCURRENCIES="${CONCURRENCIES:-1,2,4,8,16,32}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/outputs/sglang_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUT_DIR}"

if (( DELS_OPTION_REQUESTED )) && [[ -z "${DELS_LOCAL_HEAD}" ]]; then
  echo "DeLS options were set, but DELS_LOCAL_HEAD is empty." >&2
  echo "Set DELS_LOCAL_HEAD=/path/to/local_head.pt, or unset the DELS_* options." >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${DELS_LOCAL_HEAD}" ]]; then
  EXTRA_ARGS+=(--dels-local-head-path "${DELS_LOCAL_HEAD}")
  EXTRA_ARGS+=(--dels-alpha "${DELS_ALPHA}")
  EXTRA_ARGS+=(--dels-beta "${DELS_BETA}")
  EXTRA_ARGS+=(--dels-baseline-path "${DELS_BASELINE_PATH}")
  EXTRA_ARGS+=(--dels-baseline-smoothing "${DELS_BASELINE_SMOOTHING}")
  EXTRA_ARGS+=(--dels-candidate-pool-size "${DELS_CANDIDATE_POOL_SIZE}")
  if [[ -n "${DELS_MERGED_LM_HEAD}" ]]; then
    EXTRA_ARGS+=(--dels-merged-lm-head-path "${DELS_MERGED_LM_HEAD}")
  fi
fi

"${PYTHON}" "${BENCHMARK_CODE_ROOT}/benchmark_sglang_tasks.py" \
  --mode dflash \
  --target-model "${TARGET_MODEL}" \
  --draft-model "${DRAFT_MODEL}" \
  --tasks "${TASKS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --concurrencies "${CONCURRENCIES}" \
  --timeout-s 3600 \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --output-md "${OUT_DIR}/sglang_domino_tasks.md" \
  --output-jsonl "${OUT_DIR}/sglang_domino_tasks.jsonl" \
  "${EXTRA_ARGS[@]}"

echo "Wrote outputs to ${OUT_DIR}"
