#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON="${PYTHON:-python}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
DELS_LOCAL_HEAD="${DELS_LOCAL_HEAD:-checkpoints/dels-spec/qwen3-4b/local_head.pt}"
TASKS="${TASKS:-gsm8k:128,math500:128,humaneval:164,mbpp:128,livecodebench:128,mt-bench:80}"
SKIP_BASELINE="${SKIP_BASELINE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 20000))}"

DELS_ALPHA_VALUES="${DELS_ALPHA_VALUES:-0.0,0.02,0.05,0.1,0.2}"
DELS_BETA_VALUES="${DELS_BETA_VALUES:-0.0,0.05,0.1}"
DELS_BETA_BIAS_VALUES="${DELS_BETA_BIAS_VALUES:-}"
DELS_BASELINE_PATH="${DELS_BASELINE_PATH:-checkpoints/dels-spec/qwen3-4b/loss_mask_unigram.pt}"
DELS_BASELINE_SMOOTHING="${DELS_BASELINE_SMOOTHING:-1.0}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/outputs/dels_alpha_beta_scan_$(date +%Y%m%d_%H%M%S)}"
SKIP_DONE="${SKIP_DONE:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
DRY_RUN="${DRY_RUN:-0}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"

# Set SCAN_GPUS explicitly, or use CUDA_VISIBLE_DEVICES. When non-empty, the
# scan runs JOBS_PER_GPU alpha/beta combinations per listed GPU with NUM_GPUS=1 per job.
SCAN_GPUS="${SCAN_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"

if ! [[ "${JOBS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "JOBS_PER_GPU must be a positive integer, got ${JOBS_PER_GPU}." >&2
  exit 2
fi

bool_is_on() {
  case "${1,,}" in
    true|1|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

safe_value() {
  printf "%s" "$1" | tr ',' '_' | tr '-' 'm' | tr '.' 'p'
}

add_float() {
  "${PYTHON}" - "$1" "$2" <<'PY'
import sys

value = float(sys.argv[1]) + float(sys.argv[2])
print(f"{value:.12g}")
PY
}

collect_metrics() {
  local out_dir="$1"
  local metrics=""
  local log_file dataset per_step per_sample speedup
  shopt -s nullglob
  for log_file in "${out_dir}"/*_t"${TEMPERATURE}".log; do
    dataset="$(basename "${log_file}" "_t${TEMPERATURE}.log")"
    per_step="$(grep -F "Average Acceptance length (per step):" "${log_file}" | tail -n 1 | awk '{print $NF}' || true)"
    per_sample="$(grep -F "Average Acceptance length (per sample):" "${log_file}" | tail -n 1 | awk '{print $NF}' || true)"
    speedup="$(grep -F "Decoding speedup:" "${log_file}" | tail -n 1 | sed 's/^Decoding speedup: //' || true)"
    metrics+="${dataset}:step=${per_step:-NA},sample=${per_sample:-NA},speedup=${speedup:-NA};"
  done
  shopt -u nullglob
  printf "%s" "${metrics}"
}

write_combo_summary() {
  local alpha="$1"
  local beta="$2"
  local status="$3"
  local out_dir="$4"
  local metrics="$5"
  mkdir -p "${out_dir}"
  printf "%s\t%s\t%s\t%s\t%s\n" \
    "${alpha}" "${beta}" "${status}" "${out_dir}" "${metrics}" \
    > "${out_dir}/.summary.tsv"
}

rebuild_summary() {
  local summary_file="${OUT_ROOT}/summary.tsv"
  local tmp_file="${summary_file}.tmp"
  local combo_summary
  printf "alpha\tbeta\tstatus\tout_dir\tmetrics\n" > "${tmp_file}"
  shopt -s nullglob
  for combo_summary in "${OUT_ROOT}"/a*_b*/.summary.tsv; do
    cat "${combo_summary}" >> "${tmp_file}"
  done
  shopt -u nullglob
  mv "${tmp_file}" "${summary_file}"
}

run_combo() {
  local alpha="$1"
  local beta="$2"
  local out_dir="$3"
  local combo_port="$4"
  local gpu="${5:-}"
  local done_marker="${out_dir}/.done"
  local metrics status scan_log

  mkdir -p "${out_dir}"
  scan_log="${out_dir}/scan.log"

  if bool_is_on "${SKIP_DONE}" && [[ -f "${done_marker}" ]]; then
    echo "[skip] alpha=${alpha} beta=${beta} already marked done: ${out_dir}"
    metrics="$(collect_metrics "${out_dir}")"
    write_combo_summary "${alpha}" "${beta}" "skipped" "${out_dir}" "${metrics}"
    return 0
  fi

  if [[ -n "${gpu}" ]]; then
    echo "[run] gpu=${gpu} alpha=${alpha} beta=${beta} out=${out_dir} master_port=${combo_port}"
  else
    echo "[run] alpha=${alpha} beta=${beta} out=${out_dir} master_port=${combo_port}"
  fi

  local cmd=(
    env
    PYTHON="${PYTHON}"
    TARGET_MODEL="${TARGET_MODEL}"
    DRAFT_MODEL="${DRAFT_MODEL}"
    DELS_LOCAL_HEAD="${DELS_LOCAL_HEAD}"
    TASKS="${TASKS}"
    SKIP_BASELINE="${SKIP_BASELINE}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}"
    BLOCK_SIZE="${BLOCK_SIZE}"
    TEMPERATURE="${TEMPERATURE}"
    MASTER_PORT="${combo_port}"
    OUT_DIR="${out_dir}"
    DELS_ALPHA="${alpha}"
    DELS_BETA="${beta}"
    DELS_BASELINE_PATH="${DELS_BASELINE_PATH}"
    DELS_BASELINE_SMOOTHING="${DELS_BASELINE_SMOOTHING}"
  )
  if [[ -n "${gpu}" ]]; then
    cmd+=(CUDA_VISIBLE_DEVICES="${gpu}" NUM_GPUS=1)
  else
    cmd+=(NUM_GPUS="${NUM_GPUS}")
  fi
  cmd+=("${SCRIPT_DIR}/run_hf_benchmark.sh")

  printf "%q " "${cmd[@]}" > "${out_dir}/scan_command.sh"
  printf "\n" >> "${out_dir}/scan_command.sh"

  if bool_is_on "${DRY_RUN}"; then
    cat "${out_dir}/scan_command.sh"
    write_combo_summary "${alpha}" "${beta}" "dry_run" "${out_dir}" ""
    return 0
  fi

  status="ok"
  if ! "${cmd[@]}" > "${scan_log}" 2>&1; then
    status="failed"
  fi

  metrics="$(collect_metrics "${out_dir}")"
  write_combo_summary "${alpha}" "${beta}" "${status}" "${out_dir}" "${metrics}"

  if [[ "${status}" == "ok" ]]; then
    touch "${done_marker}"
    return 0
  fi
  return 1
}

declare -a WORKER_GPUS=()
declare -a WORKER_SLOTS=()
if [[ -n "${SCAN_GPUS}" && "${SCAN_GPUS}" != "all" ]]; then
  IFS=',' read -r -a RAW_GPU_ARRAY <<< "${SCAN_GPUS}"
  for gpu in "${RAW_GPU_ARRAY[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    if [[ -n "${gpu}" && "${gpu}" != "-1" ]]; then
      WORKER_GPUS+=("${gpu}")
      for ((slot = 0; slot < JOBS_PER_GPU; slot++)); do
        WORKER_SLOTS+=("${gpu}")
      done
    fi
  done
fi

if (( ${#WORKER_GPUS[@]} == 0 )); then
  NUM_GPUS="${NUM_GPUS:-8}"
fi

mkdir -p "${OUT_ROOT}"

echo "Writing scan outputs to ${OUT_ROOT}"
echo "alpha values: ${DELS_ALPHA_VALUES}"
if [[ -n "${DELS_BETA_BIAS_VALUES}" ]]; then
  echo "scan mode:    beta = alpha + beta_bias"
  echo "beta biases:  ${DELS_BETA_BIAS_VALUES}"
else
  echo "scan mode:    alpha x beta"
  echo "beta values:  ${DELS_BETA_VALUES}"
fi
echo "rnn input:    block_prefix"
echo "baseline:     ${DELS_BASELINE_PATH} (smoothing=${DELS_BASELINE_SMOOTHING})"
echo "master port:  ${MASTER_PORT} + combo_index"
if (( ${#WORKER_GPUS[@]} > 0 )); then
  echo "worker GPUs: ${WORKER_GPUS[*]} (JOBS_PER_GPU=${JOBS_PER_GPU}, NUM_GPUS=1 per job)"
  echo "worker slots: ${WORKER_SLOTS[*]}"
else
  echo "worker GPUs: none (serial scan, NUM_GPUS=${NUM_GPUS} per combination)"
fi

declare -a COMBO_ALPHAS=()
declare -a COMBO_BETAS=()
declare -a COMBO_OUT_DIRS=()
declare -a COMBO_PORTS=()

add_combo() {
  local alpha="$1"
  local beta="$2"
  local alpha_tag beta_tag
  alpha_tag="$(safe_value "${alpha}")"
  beta_tag="$(safe_value "${beta}")"
  COMBO_ALPHAS+=("${alpha}")
  COMBO_BETAS+=("${beta}")
  COMBO_OUT_DIRS+=("${OUT_ROOT}/a${alpha_tag}_b${beta_tag}")
  COMBO_PORTS+=("$((MASTER_PORT + combo_index))")
  combo_index=$((combo_index + 1))
}

combo_index=0
if [[ -n "${DELS_BETA_BIAS_VALUES}" ]]; then
  for alpha in ${DELS_ALPHA_VALUES//,/ }; do
    for beta_bias in ${DELS_BETA_BIAS_VALUES//,/ }; do
      beta="$(add_float "${alpha}" "${beta_bias}")"
      add_combo "${alpha}" "${beta}"
    done
  done
else
  for alpha in ${DELS_ALPHA_VALUES//,/ }; do
    for beta in ${DELS_BETA_VALUES//,/ }; do
      add_combo "${alpha}" "${beta}"
    done
  done
fi

failure_count=0
total_combos="${#COMBO_ALPHAS[@]}"
worker_count="${#WORKER_SLOTS[@]}"

if (( worker_count == 0 )); then
  for ((i = 0; i < total_combos; i++)); do
    if ! run_combo \
      "${COMBO_ALPHAS[$i]}" \
      "${COMBO_BETAS[$i]}" \
      "${COMBO_OUT_DIRS[$i]}" \
      "${COMBO_PORTS[$i]}"; then
      failure_count=$((failure_count + 1))
      if ! bool_is_on "${CONTINUE_ON_ERROR}"; then
        rebuild_summary
        exit 1
      fi
    fi
  done
else
  declare -A PID_TO_GPU=()
  active_count=0
  next_combo=0
  stop_scheduling=0

  launch_next_on_gpu() {
    local gpu="$1"
    local combo_i="${next_combo}"
    local pid

    if (( combo_i >= total_combos )); then
      return 1
    fi

    next_combo=$((next_combo + 1))
    echo "[dispatch] gpu=${gpu} combo=$((combo_i + 1))/${total_combos}"
      run_combo \
        "${COMBO_ALPHAS[$combo_i]}" \
        "${COMBO_BETAS[$combo_i]}" \
        "${COMBO_OUT_DIRS[$combo_i]}" \
        "${COMBO_PORTS[$combo_i]}" \
        "${gpu}" &
    pid="$!"
    PID_TO_GPU["${pid}"]="${gpu}"
    active_count=$((active_count + 1))
    return 0
  }

  for gpu in "${WORKER_SLOTS[@]}"; do
    launch_next_on_gpu "${gpu}" || true
  done

  while (( active_count > 0 )); do
    finished_pid=""
    if wait -n -p finished_pid; then
      finished_status=0
    else
      finished_status=$?
    fi
    finished_gpu="${PID_TO_GPU[${finished_pid}]:-}"
    unset "PID_TO_GPU[${finished_pid}]"
    active_count=$((active_count - 1))

    if (( finished_status != 0 )); then
      failure_count=$((failure_count + 1))
      if ! bool_is_on "${CONTINUE_ON_ERROR}"; then
        stop_scheduling=1
      fi
    fi

    if (( stop_scheduling == 0 && next_combo < total_combos )) \
      && [[ -n "${finished_gpu}" ]]; then
      launch_next_on_gpu "${finished_gpu}" || true
    fi
  done

  if (( failure_count > 0 )) && ! bool_is_on "${CONTINUE_ON_ERROR}"; then
    rebuild_summary
    exit 1
  fi
fi

rebuild_summary
echo "Wrote summary to ${OUT_ROOT}/summary.tsv"

if (( failure_count > 0 )); then
  echo "Completed with ${failure_count} failed combination(s)." >&2
  exit 1
fi
