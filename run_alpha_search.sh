#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_HF_BENCHMARK="${RUN_HF_BENCHMARK:-${SCRIPT_DIR}/run_hf_benchmark.sh}"
ALPHAS="${ALPHAS:-0,0.25,0.5,0.75,1.0,1.25,1.5,2.0}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/outputs/alpha_search_$(date +%Y%m%d_%H%M%S)}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29601}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

mkdir -p "${OUT_ROOT}"

SUMMARY_FILE="${OUT_ROOT}/summary.tsv"
printf "alpha\tstatus\tdataset\tspeedup\tacceptance_per_step\tacceptance_per_sample\tlog_file\n" > "${SUMMARY_FILE}"

IFS=',' read -r -a ALPHA_ARRAY <<< "${ALPHAS}"

alpha_index=0
for alpha in "${ALPHA_ARRAY[@]}"; do
  alpha="$(printf '%s' "${alpha}" | xargs)"
  if [[ -z "${alpha}" ]]; then
    continue
  fi

  alpha_tag="${alpha//./p}"
  alpha_tag="${alpha_tag//-/m}"
  alpha_out_dir="${OUT_ROOT}/alpha_${alpha_tag}"
  master_port=$((MASTER_PORT_BASE + alpha_index))
  alpha_index=$((alpha_index + 1))

  echo "=== alpha=${alpha} out=${alpha_out_dir} master_port=${master_port} ==="

  status="ok"
  if ! DOMINO_GUIDANCE_ALPHA="${alpha}" \
       OUT_DIR="${alpha_out_dir}" \
       MASTER_PORT="${master_port}" \
       "${RUN_HF_BENCHMARK}"; then
    status="failed"
    echo "alpha=${alpha} failed"
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit 1
    fi
  fi

  shopt -s nullglob
  log_files=("${alpha_out_dir}"/*.log)
  shopt -u nullglob

  if [[ "${#log_files[@]}" -eq 0 ]]; then
    printf "%s\t%s\t\t\t\t\t\n" "${alpha}" "${status}" >> "${SUMMARY_FILE}"
    continue
  fi

  for log_file in "${log_files[@]}"; do
    dataset="$(basename "${log_file}")"
    dataset="${dataset%%_t*}"
    speedup="$(awk -F': ' '/Decoding speedup:/ {v=$2} END {print v}' "${log_file}")"
    acc_step="$(awk -F': *' '/Average Acceptance length \(per step\):/ {v=$2} END {print v}' "${log_file}")"
    acc_sample="$(awk -F': *' '/Average Acceptance length \(per sample\):/ {v=$2} END {print v}' "${log_file}")"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${alpha}" "${status}" "${dataset}" "${speedup}" "${acc_step}" "${acc_sample}" "${log_file}" \
      >> "${SUMMARY_FILE}"
  done
done

echo "Wrote alpha-search outputs to ${OUT_ROOT}"
echo "Wrote summary to ${SUMMARY_FILE}"
