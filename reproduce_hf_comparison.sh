#!/usr/bin/env bash
# Compare a fixed DFlash draft model with and without a trained RNN local head.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHON="${PYTHON:-python}"
export DELS_LOCAL_HEAD="${DELS_LOCAL_HEAD:?Set DELS_LOCAL_HEAD to the trained local_head.pt}"
export DELS_BASELINE_PATH="${DELS_BASELINE_PATH:?Set DELS_BASELINE_PATH to its unigram prior}"
[[ -f "$DELS_LOCAL_HEAD" ]] || { echo "Missing local head: $DELS_LOCAL_HEAD" >&2; exit 2; }
[[ -e "$DELS_BASELINE_PATH" ]] || { echo "Missing unigram prior: $DELS_BASELINE_PATH" >&2; exit 2; }
export TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-8B}"
export DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-8B-DFlash-b16}"
export TASKS="${TASKS:-gsm8k:128}"
export NUM_GPUS="${NUM_GPUS:-1}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export BLOCK_SIZE="${BLOCK_SIZE:-16}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export SKIP_BASELINE=0
export DELS_ALPHA="${DELS_ALPHA:-0.3}"
export DELS_BETA="${DELS_BETA:-0.3}"
export DELS_BASELINE_SMOOTHING="${DELS_BASELINE_SMOOTHING:-1.0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/outputs/comparison_$(date +%Y%m%d_%H%M%S)}"
# Keep checkpoint provenance unambiguous for this basic comparison.
unset POSITION_MIX_JOINT DELS_MERGED_LM_HEAD BENCHMARK_CODE_ROOT
if [[ -e "$OUT_ROOT" ]]; then
  echo "OUT_ROOT already exists; choose a fresh directory: $OUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUT_ROOT"
"$PYTHON" - <<'PY'
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
import torch

names = ('TARGET_MODEL', 'DRAFT_MODEL', 'TASKS', 'NUM_GPUS', 'CUDA_VISIBLE_DEVICES',
         'TEMPERATURE', 'BLOCK_SIZE', 'MAX_NEW_TOKENS', 'SKIP_BASELINE',
         'DELS_LOCAL_HEAD', 'DELS_BASELINE_PATH', 'DELS_ALPHA', 'DELS_BETA',
         'DELS_BASELINE_SMOOTHING', 'HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE',
         'TRANSFORMERS_OFFLINE')
metadata = {'settings': {key: os.environ.get(key) for key in names},
            'python': platform.python_version(), 'pytorch_cuda_build': torch.version.cuda,
            'evaluation_seed': 0, 'packages': {}, 'artifacts': {}, 'gpus': []}
for name in ('torch', 'transformers', 'datasets', 'triton', 'accelerate', 'flash-attn'):
    try:
        metadata['packages'][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        metadata['packages'][name] = None
for key in ('DELS_LOCAL_HEAD', 'DELS_BASELINE_PATH'):
    path = Path(os.environ[key])
    if path.is_dir():
        path = next((path / name for name in ('loss_mask_unigram.pt', 'loss_mask_unigram.npz')
                     if (path / name).is_file()), None)
        if path is None:
            raise ValueError('Prior directory has no loss_mask_unigram.pt or .npz')
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    metadata['artifacts'][key] = {'path': str(path.resolve()), 'sha256': digest.hexdigest()}
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    metadata['gpus'].append({'name': props.name, 'total_memory_bytes': props.total_memory,
                             'compute_capability': [props.major, props.minor]})
result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
metadata['runtime_commit'] = result.stdout.strip() if result.returncode == 0 else None
result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
metadata['runtime_has_local_changes'] = bool(result.stdout.strip()) if result.returncode == 0 else None
with (Path(os.environ['OUT_ROOT']) / 'run_metadata.json').open('w') as target:
    json.dump(metadata, target, indent=2)
PY
(
  unset DELS_LOCAL_HEAD DELS_ALPHA DELS_BETA DELS_BASELINE_PATH DELS_BASELINE_SMOOTHING
  OUT_DIR="$OUT_ROOT/dflash" bash "$SCRIPT_DIR/run_hf_benchmark.sh"
)
OUT_DIR="$OUT_ROOT/dels" bash "$SCRIPT_DIR/run_hf_benchmark.sh"
echo "Comparison logs and metadata: $OUT_ROOT"
