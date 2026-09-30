# Basic DeLS-Spec reproduction

This guide trains a short-context RNN head independently and evaluates it with
a fixed DFlash draft model. The prior and head use the same target tokenizer,
data, template, and length limit. It is a baseline recipe; exact paper-table
reproduction additionally requires the paper's final data/model revisions,
checkpoint selection, coefficient schedules, and benchmark environment.

## Matching sources and installation

Training repository: [dt-3t/SpecForge, add-dels-spec](https://github.com/dt-3t/SpecForge/tree/add-dels-spec).
Runtime repository: [dt-3t/DeLS-Spec](https://github.com/dt-3t/DeLS-Spec).

Use the training `add-dels-spec` branch and runtime `main` containing this guide.
The initial public runtime release `ab9be1b4` does not support fused RNN exports;
use the updated runtime. This recipe supersedes the older `5503b2a` training
snapshot and its full-target-embedding RNN architecture.

Clone the repositories alongside each other:

```bash
git clone --branch add-dels-spec https://github.com/dt-3t/SpecForge.git DeLS-Spec-training
git clone https://github.com/dt-3t/DeLS-Spec.git
```

Record the actual commits or release tags for both repositories. Run in a fresh
Python 3.11 CUDA environment; from the training checkout:

```bash
cd DeLS-Spec-training
python -m pip install -e .
python -m pip install -r ../DeLS-Spec/requirements-hf.txt
# Optional, only for unigram plots:
python -m pip install -e '.[dels-plots]'
```

This checkout pins PyTorch 2.11.0, Transformers 5.8.1, and SGLang 0.5.14 in
`pyproject.toml`. Install a CUDA PyTorch build compatible with the machine's
driver. The CPU compatibility tests were run in an existing environment with
Python 3.11.15, PyTorch 2.9.1+cu128, Transformers 4.57.1, SGLang 0.5.9,
datasets 5.0.0, Triton 3.5.1, and Accelerate 1.14.0. These tests do not validate
a clean installation of the declared dependencies or full GPU training. No
minimum VRAM, GPU type, or benchmark performance has been verified for this
snapshot. The runtime uses FlashAttention2 when installed, otherwise SDPA;
record the actual backend printed in the benchmark log.

## Data and training

Run from the training checkout:

```bash
mkdir -p cache/dataset
hf download yaro1214/open_data \
  sharegpt_train_regen_temperature0_no_think.jsonl \
  --repo-type dataset --local-dir cache/dataset

# Use a fresh output directory for each run.
export OUT_ROOT="$PWD/outputs/qwen3-8b-dels-reproduction"
export TARGET_MODEL_PATH=Qwen/Qwen3-8B
export TRAIN_DATA_PATH="$PWD/cache/dataset/sharegpt_train_regen_temperature0_no_think.jsonl"
export CUDA_VISIBLE_DEVICES=0
export NUM_EPOCHS=1
export BATCH_SIZE=4
export SEED=42
export LOCAL_HEAD_PARAMETERIZATION=target_factorized
bash examples/run_qwen3_8b_dels_local_head.sh 1
```

This uses conversation data, the `qwen` template, and assistant-token loss masks.
The trainer filters samples with fewer than 32 supervised tokens; the prior
counts assistant tokens before that filter. After training, rank 0 generates
`$OUT_ROOT/loss_mask_unigram/loss_mask_unigram.pt` from the same processed data.
Plotting is optional. Pass `8` instead of `1` for eight visible GPUs; that changes
the effective batch size. Lower `BATCH_SIZE` if needed, recording the change.
These explicit settings describe the provided launcher, not a verified optimal
paper configuration:

| Setting | Value |
| --- | --- |
| Head / parameterization | RNN / target_factorized |
| Input rank / GRU hidden / output rank | 256 / 1024 / 256 |
| Block size / anchors | 16 / 512 |
| Rank activation / output projection init | SiLU / zero |
| Loss decay gamma | 7.0 |
| Batch per GPU / accumulation | 4 / 1 |
| Learning rate / warmup ratio | 0.0006 / 0.04 |
| Max gradient norm / sequence length | 1.0 / 3072 |
| Epochs / seed | 1 / 42 |

This version always trains next-token targets at offsets 1 through block_size-1
with zero initial GRU state. It has no `--shift-label`, `--pure-draft-prefix-len`,
or `--local-lm-head-mode` flags. Do not copy those flags from the older trainer.
The documented evaluation uses the public DFlash model's normal unshifted
alignment; a draft checkpoint with `dflash_config.shift_label=true` requires a
matching shifted training implementation and is outside this recipe.

Select a checkpoint explicitly from
`$OUT_ROOT/local_head/epoch_<epoch>_step_<step>/`. The final save uses
`epoch_<num_epochs>`, whereas intermediate saves use the training epoch counter.
Use `merged_rnn_local_head.pt` for inference. `local_head.pt` keeps training
state and may include target-dependent input/output projections. Export is
on by default. An existing checkpoint can also be converted:

```bash
python scripts/convert_local_head.py /absolute/path/to/local_head.pt \
  --output-path /absolute/path/to/merged_rnn_local_head.pt
```

For `target_factorized`, conversion needs the original target weights; use the
same model revision as training. For `direct_vocab`, the learned embedding and
output head are already vocab-space weights. Both parameterizations produce a
self-contained merged head supported by the updated runtime. This is distinct
from the older released head architecture, which fed full target embeddings
straight into the GRU; do not assume they reproduce the same trained weights.

For Qwen3-4B, change `TARGET_MODEL_PATH`, use
`TY233/qwen3-4b-instruct-100k-backup/qwen3-4b-100k.jsonl`, select a fresh output
directory, and evaluate with `z-lab/Qwen3-4B-DFlash-b16`. Keep head, prior, and
target consistent. Record dataset and model revisions and dataset hashes.

## Standalone unigram statistics and data formats

To generate the same prior independently, run:

```bash
python scripts/count_unigram.py \
  --target-model-path "$TARGET_MODEL_PATH" \
  --train-data-path "$TRAIN_DATA_PATH" \
  --data-format conversation --chat-template qwen --max-length 3072 \
  --output-dir "$OUT_ROOT/loss_mask_unigram"
```

The trainer also supports `--data-format raw_text` and `preformatted`. Raw text
uses the `text` column and next-token loss; preformatted input uses the tokenizer
without a conversation template. Streaming raw-text training requires
`--max-train-steps` and skips the integrated prior export. For that path, use
`scripts/count_unigram.py --data-format raw_text --streaming` separately, with
the same data and tokenization settings. The provided basic launcher intentionally
uses non-streaming conversation data. See `tests/test_data/` for schemas.
Dataset and model licenses are separate from this repository's code license.

## Paired evaluation

Run from the updated runtime checkout:

```bash
cd ../DeLS-Spec
export CUDA_VISIBLE_DEVICES=0
export TARGET_MODEL=Qwen/Qwen3-8B
export DRAFT_MODEL=z-lab/Qwen3-8B-DFlash-b16
export DELS_LOCAL_HEAD=/absolute/path/to/merged_rnn_local_head.pt
export DELS_BASELINE_PATH=/absolute/path/to/loss_mask_unigram/loss_mask_unigram.pt
export TASKS=gsm8k:128
export OUT_ROOT="$PWD/outputs/qwen3-8b-comparison"
bash reproduce_hf_comparison.sh
```

Replace the artifact paths with the actual training outputs. Both runs use one
GPU, temperature 0, block size 16, and maximum new tokens 2048 by default.
Alpha=beta=0.3 are example starting values; select and report coefficients using
held-out data, not the final test set. A quick execution check can use
`TASKS=gsm8k:8 MAX_NEW_TOKENS=128`; it is not a throughput result. The comparison
script enables downloads by default. Use offline settings only with populated
caches.

`dflash/` and `dels/` contain logs and answer JSONL files, including a target-only
baseline for each run. Compare `Decoding speedup` (against each run's target-only
decoding) and `Average Acceptance length (per sample)` (accepted tokens per
step, averaged over samples). These are not task accuracy metrics. The runtime
seeds sample selection and decoding with 0. Keep GPU load, backend, tasks, and
token limits identical. `run_metadata.json` records runtime commit, dirty state,
software/GPU information, settings, and artifact hashes.

For Markov, set `LOCAL_HEAD_TYPE=markov` in the trainer launcher and evaluate
`merged_markov_local_head.pt`; both parameterizations support this export.
Unmerged direct-vocab Markov checkpoints are not supported by this runtime.
Optional joint-finetuning variants are outside this guide.

## Validation and reference results

Existing training/data tests and CPU integration tests verify checkpoint type
detection, RNN export/loading, step-by-step logits, preservation of target
embeddings, and old checkpoint compatibility. Shell and mocked launcher checks
verify argument handling without GPU training. Full training, GPU decoding,
clean installation with the declared dependency versions, and paper metrics
remain unverified for this snapshot. Fill this table only after measured runs:

| Run | GPU / backend | Task / samples | Acceptance length per sample | Speedup vs target |
| --- | --- | --- | --- | --- |
| DFlash | To be measured | gsm8k / 128 | To be measured | To be measured |
| DeLS-Spec | Same setup | gsm8k / 128 | To be measured | To be measured |

Save training commands, data hashes/revisions, target/draft revisions, selected
checkpoint, `run_metadata.json`, and both logs with the results. Repeat timings
and measure task accuracy separately when reporting paper results.
