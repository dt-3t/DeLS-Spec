import argparse
import json
import time
import random
from itertools import chain
from loguru import logger
import numpy as np
import torch
from rich import print
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available
from model import load_and_process_dataset
from dflash import DFlashDraftModel, is_domino_projector
import distributed as dist
from kernel.domino import DraftCorrectionGraphRunner
from kernel.dels import DeLSGraphRunner
from kernel.markov_dels import MarkovDeLSGraphRunner
from dels import (
    DeLSLocalHead,
    MarkovDeLSLocalHead,
    PositionMixJointHead,
    detect_local_head_checkpoint_kind,
    load_unigram_log_prior,
    parse_scalar_schedule,
)
import os


def normalize_draft_config_for_benchmark(config):
    dflash_config = dict(getattr(config, "dflash_config", {}) or {})

    for key in (
        "mask_token_id",
        "target_layer_ids",
        "projector_type",
        "pure_draft_prefix_len",
        "shift_label",
    ):
        if key not in dflash_config and hasattr(config, key):
            value = getattr(config, key)
            if value is not None:
                dflash_config[key] = value

    if dflash_config.get("projector_type") == "causal_v5":
        dflash_config["projector_type"] = "domino"

    if "emb_dim" not in dflash_config:
        emb_dim = getattr(config, "emb_dim", None)
        if emb_dim is not None:
            dflash_config["emb_dim"] = emb_dim

    if "gru_hidden_dim" not in dflash_config:
        gru_hidden_dim = getattr(config, "gru_hidden_dim", None)
        if gru_hidden_dim is not None:
            dflash_config["gru_hidden_dim"] = gru_hidden_dim
        elif "emb_dim" in dflash_config:
            dflash_config["gru_hidden_dim"] = dflash_config["emb_dim"]

    if "shift_label" not in dflash_config:
        logger.warning(
            "!!! WARNING: DFlash checkpoint config has no shift_label; "
            "defaulting DFlash shift_label to False. !!!"
        )
        dflash_config["shift_label"] = False

    if "pure_draft_prefix_len" not in dflash_config:
        dflash_config["pure_draft_prefix_len"] = 1

    config.dflash_config = dflash_config
    return config


def load_draft_model_for_benchmark(model_name_or_path: str, attn_impl: str):
    draft_config = AutoConfig.from_pretrained(model_name_or_path)
    draft_config = normalize_draft_config_for_benchmark(draft_config)
    return DFlashDraftModel.from_pretrained(
        model_name_or_path,
        config=draft_config,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--draft-name-or-path", type=str, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--use-graph", action="store_true")
    parser.add_argument("--use-bias", action="store_true")
    parser.add_argument(
        "--domino-guidance-alpha",
        type=float,
        default=1.0,
        help="Scale for Domino correction logits: base_logits + alpha * correction_bias.",
    )
    parser.add_argument(
        "--dels-local-head-path",
        type=str,
        default=None,
        help=(
            "Path to a SpecForge local_head.pt, markov_local_head.pt, or "
            "merged_markov_local_head.pt. The checkpoint type is detected "
            "automatically."
        ),
    )
    parser.add_argument(
        "--dels-merged-lm-head-path",
        type=str,
        default=None,
        help="Optional path to merged_lm_head.pt for DeLS low-rank logits.",
    )
    parser.add_argument(
        "--dels-alpha",
        type=str,
        default="1.0",
        help=(
            "DeLS alpha schedule. Use a scalar or comma-separated per-position "
            "values for final_logits=long+alpha*short-beta*baseline."
        ),
    )
    parser.add_argument(
        "--dels-beta",
        type=str,
        default="1.0",
        help=(
            "DeLS beta schedule. Use a scalar or comma-separated per-position "
            "values for final_logits=long+alpha*short-beta*baseline."
        ),
    )
    parser.add_argument(
        "--dels-baseline-path",
        type=str,
        default="checkpoints/dels-spec/qwen3-8b/loss_mask_unigram.pt",
        help=(
            "Path to loss-mask unigram stats used as b_0(v) for DeLS "
            "block_prefix mode. Can be a directory, .pt, or .npz."
        ),
    )
    parser.add_argument(
        "--dels-baseline-smoothing",
        type=float,
        default=1.0,
        help="Additive smoothing for unigram counts before taking log probabilities.",
    )
    parser.add_argument(
        "--position-mix-joint-path",
        type=str,
        default=None,
        help=(
            "Path to SpecForge block_prefix_position_mix_joint.pt. Enables "
            "block-prefix per-position alpha/beta joint finetune evaluation."
        ),
    )
    parser.add_argument("--dump-benchmark-manifest", type=str, default=None)
    parser.add_argument("--dump-only", action="store_true")
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help=(
            "Skip the block_size=1 target-only baseline. This disables speedup "
            "reporting but still reports acceptance length for block_size=k."
        ),
    )
    parser.add_argument("--answer-file", type=str, default=None, help="Output answer file (jsonl) to store generation results for both b=1 and b=k.")
    parser.add_argument("--attn-implementation", type=str, default=None, choices=["eager", "sdpa", "flash_attention_2"], help="Attention implementation for target and draft models. Default: auto-detect flash_attn.")

    args = parser.parse_args()

    if not args.dump_only and (args.model_name_or_path is None or args.draft_name_or_path is None):
        parser.error("--model-name-or-path and --draft-name-or-path are required unless --dump-only is set")

    # Fast path: dump-only mode skips model loading / CUDA init entirely
    if args.dump_only:
        dataset = load_and_process_dataset(args.dataset)
        if args.max_samples is not None and len(dataset) > args.max_samples:
            dataset = dataset.shuffle(seed=0).select(range(args.max_samples))
        if args.dump_benchmark_manifest:
            with open(args.dump_benchmark_manifest, "w", encoding="utf-8") as f:
                for selected_sample_idx, instance in enumerate(dataset):
                    record = {
                        "selected_sample_idx": selected_sample_idx,
                        "question_id": selected_sample_idx,
                        "turns": instance["turns"],
                        "num_turns": len(instance["turns"]),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Saved benchmark manifest with {len(dataset)} samples to {args.dump_benchmark_manifest}")
        return

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    if args.attn_implementation is not None:
        attn_impl = args.attn_implementation
        logger.info(f"Using specified attention implementation: {attn_impl}")
    else:
        def has_flash_attn():
            if is_flash_attn_2_available():
                return True
            logger.warning("FlashAttention2 is not available. Falling back to torch.sdpa. The speedup will be lower.")
            return False

        installed_flash_attn = has_flash_attn()
        attn_impl = "flash_attention_2" if installed_flash_attn else "sdpa"
        logger.info(f"Auto-detected attention implementation: {attn_impl}")

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = load_draft_model_for_benchmark(
        args.draft_name_or_path,
        attn_impl,
    ).to(device).eval()
    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    enabled_joint_modes = [
        args.dels_local_head_path is not None,
        args.position_mix_joint_path is not None,
    ]
    if sum(enabled_joint_modes) > 1:
        parser.error(
            "Use only one of --dels-local-head-path and "
            "--position-mix-joint-path"
        )

    dels_local_head = None
    markov_dels_local_head = None
    dels_alpha = args.dels_alpha
    dels_beta = args.dels_beta
    dels_baseline_logits = None
    markov_dels_baseline_logits = None
    if args.dels_local_head_path is not None:
        config_dels_prefix_len = int(getattr(draft_model, "pure_draft_prefix_len", 0))
        config_dels_shift_label = bool(
            getattr(draft_model.config, "dflash_config", {}).get("shift_label", False)
        )
        local_head_kind = detect_local_head_checkpoint_kind(args.dels_local_head_path)
        if local_head_kind == "markov":
            if args.dels_merged_lm_head_path is not None:
                parser.error(
                    "--dels-merged-lm-head-path is only valid for RNN DeLS "
                    "local-head checkpoints."
                )
            markov_dels_local_head = MarkovDeLSLocalHead.from_checkpoint(
                checkpoint_path=args.dels_local_head_path,
                target_model=target,
                dtype=torch.bfloat16,
                device=device,
            ).eval()
            k_draft = block_size if config_dels_shift_label else block_size - 1
            markov_steps = max(k_draft - 1, 0)
            markov_beta_requires_baseline = False
            if markov_steps > 0:
                try:
                    markov_beta_values = parse_scalar_schedule(
                        args.dels_beta,
                        markov_steps,
                        name="Markov DeLS beta",
                    )
                except ValueError as exc:
                    parser.error(str(exc))
                markov_beta_requires_baseline = any(
                    beta != 0.0 for beta in markov_beta_values
                )
            if markov_beta_requires_baseline:
                markov_dels_baseline_logits = load_unigram_log_prior(
                    args.dels_baseline_path,
                    vocab_size=int(markov_dels_local_head.vocab_size),
                    dtype=torch.bfloat16,
                    device=device,
                    smoothing=float(args.dels_baseline_smoothing),
                )
            logger.info(
                "Enabled Markov DeLS via --dels-local-head-path: "
                f"local_head={args.dels_local_head_path}, "
                f"alpha={args.dels_alpha}, "
                f"beta={args.dels_beta}, "
                f"rank={markov_dels_local_head.rank}, "
                f"dflash_pure_draft_prefix_len={config_dels_prefix_len}, "
                f"dflash_shift_label={config_dels_shift_label}, "
                f"baseline_path={args.dels_baseline_path}, "
                f"baseline_smoothing={args.dels_baseline_smoothing}, "
                f"baseline_required={markov_beta_requires_baseline}, "
                "inference_pure_draft_prefix_len=1"
            )
        else:
            dels_local_head = DeLSLocalHead.from_checkpoint(
                checkpoint_path=args.dels_local_head_path,
                target_model=target,
                dtype=torch.bfloat16,
                device=device,
                merged_lm_head_path=args.dels_merged_lm_head_path,
            ).eval()
            k_draft = block_size if config_dels_shift_label else block_size - 1
            dels_steps = max(k_draft - 1, 0)
            dels_beta_requires_baseline = False
            if dels_steps > 0:
                try:
                    dels_beta_values = parse_scalar_schedule(
                        args.dels_beta, dels_steps, name="DeLS beta"
                    )
                except ValueError as exc:
                    parser.error(str(exc))
                dels_beta_requires_baseline = any(
                    beta != 0.0 for beta in dels_beta_values
                )
            if dels_beta_requires_baseline:
                dels_baseline_logits = load_unigram_log_prior(
                    args.dels_baseline_path,
                    vocab_size=int(dels_local_head.vocab_size),
                    dtype=torch.bfloat16,
                    device=device,
                    smoothing=float(args.dels_baseline_smoothing),
                )
            if config_dels_shift_label and not bool(dels_local_head.shift_label):
                parser.error(
                    "Invalid DeLS shift_label combination: "
                    f"dflash={config_dels_shift_label}, "
                    f"local_head={bool(dels_local_head.shift_label)}. "
                    "A shifted DFlash checkpoint requires a shifted local head; "
                    "the reverse mismatch is allowed."
                )
            logger.info(
                "Enabled block-prefix DeLS: "
                f"local_head={args.dels_local_head_path}, "
                f"merged_lm_head={args.dels_merged_lm_head_path}, "
                f"alpha={args.dels_alpha}, "
                f"beta={args.dels_beta}, "
                "rnn_input=block_prefix, "
                f"dflash_pure_draft_prefix_len={config_dels_prefix_len}, "
                f"dflash_shift_label={config_dels_shift_label}, "
                f"local_head_pure_draft_prefix_len={dels_local_head.pure_draft_prefix_len}, "
                f"local_head_shift_label={dels_local_head.shift_label}, "
                f"local_head_rank_activation={dels_local_head.rank_activation}, "
                f"baseline_path={args.dels_baseline_path}, "
                f"baseline_smoothing={args.dels_baseline_smoothing}, "
                f"baseline_required={dels_beta_requires_baseline}, "
                "inference_pure_draft_prefix_len=1"
            )
    elif args.position_mix_joint_path is not None:
        dflash_shift_label = bool(
            getattr(draft_model.config, "dflash_config", {}).get("shift_label", False)
        )
        dels_local_head = PositionMixJointHead.from_checkpoint(
            checkpoint_path=args.position_mix_joint_path,
            target_model=target,
            block_size=block_size,
            dtype=torch.bfloat16,
            device=device,
        ).eval()
        position_rnn_input = getattr(dels_local_head, "rnn_input", "block_prefix")
        if position_rnn_input != "block_prefix":
            parser.error(
                "Only block-prefix position joint checkpoints are supported. "
                f"Got rnn_input={position_rnn_input!r}."
            )
        allow_shifted_position_with_unshifted_dflash = (
            not dflash_shift_label and bool(dels_local_head.shift_label)
        )
        if (
            bool(dels_local_head.shift_label) != dflash_shift_label
            and not allow_shifted_position_with_unshifted_dflash
        ):
            parser.error(
                "shift_label conflict between DFlash checkpoint and joint head: "
                f"dflash={dflash_shift_label}, "
                f"joint_head={bool(dels_local_head.shift_label)}. "
                "DFlash checkpoint is the only source of truth."
            )
        dels_alpha, dels_beta = dels_local_head.alpha_beta_schedules()
        dels_baseline_logits = dels_local_head.baseline_logits
        logger.info(
            "Enabled block-prefix position-mixing joint checkpoint as DeLS "
            "schedules: "
            f"checkpoint={args.position_mix_joint_path}, "
            f"pure_draft_prefix_len={dels_local_head.pure_draft_prefix_len}, "
            f"shift_label={dels_local_head.shift_label}, "
            f"rnn_input={position_rnn_input}, "
            f"mix_form={dels_local_head.mix_form}, "
            f"suffix_positions={int(dels_local_head.position_gates.shape[0])}"
        )
    logger.info(f"[VERIFY] Target attn_implementation: {target.config._attn_implementation}")
    logger.info(f"[VERIFY] Draft attn_implementation: {draft_model.config._attn_implementation}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    if args.dump_benchmark_manifest and dist.is_main():
        with open(args.dump_benchmark_manifest, "w", encoding="utf-8") as f:
            for selected_sample_idx, instance in enumerate(dataset):
                record = {
                    "selected_sample_idx": selected_sample_idx,
                    "question_id": selected_sample_idx,
                    "turns": instance["turns"],
                    "num_turns": len(instance["turns"]),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved benchmark manifest with {len(dataset)} samples to {args.dump_benchmark_manifest}")

    hidden_size = int(target.lm_head.weight.shape[1])
    vocab_size = int(target.lm_head.weight.shape[0])
    prefix_len = int(getattr(draft_model, "pure_draft_prefix_len", 0))
    projector_type = getattr(draft_model, "projector_type", None)
    is_domino = is_domino_projector(projector_type)
    graph_runner = None
    if dels_local_head is not None:
        logger.info(
            "DeLS block_prefix mode is enabled; the first draft token "
            "uses DFlash long logits, then the RNN consumes only generated "
            "block-prefix tokens and applies the unigram prior baseline."
        )
    elif markov_dels_local_head is not None:
        logger.info(
            "Markov DeLS mode is enabled; the first draft token uses "
            "DFlash long logits, then the Markov local head consumes the "
            "previous generated token and applies the unigram prior baseline."
        )
    elif is_domino:
        logger.info("Detected Domino draft checkpoint; Domino correction is enabled.")
    else:
        logger.info(
            "Detected non-Domino DFlash draft checkpoint "
            f"(projector_type={projector_type!r}); running plain DFlash decoding."
        )

    if args.use_graph and dels_local_head is not None:
        shift_label = bool(
            getattr(draft_model.config, "dflash_config", {}).get("shift_label", False)
        )
        K = block_size if shift_label else block_size - 1
        steps = K - 1
        if steps <= 0:
            logger.info(
                "--use-graph is ignored for block-prefix DeLS because "
                f"there are no suffix correction steps (block_size={block_size})."
            )
        else:
            try:
                alpha_values = parse_scalar_schedule(
                    dels_alpha, steps, name="DeLS alpha"
                )
                beta_values = parse_scalar_schedule(
                    dels_beta, steps, name="DeLS beta"
                )
            except ValueError as exc:
                parser.error(str(exc))
            if int(dels_local_head.vocab_size) != vocab_size:
                parser.error(
                    "DeLS local-head vocab size does not match target lm_head: "
                    f"local_head={dels_local_head.vocab_size}, target={vocab_size}."
                )
            graph_runner = DeLSGraphRunner(
                local_head=dels_local_head,
                batch_size=1,
                steps=steps,
                vocab_size=vocab_size,
                alpha_values=alpha_values,
                beta_values=beta_values,
                baseline_logits=dels_baseline_logits,
                device=device,
            )
            logger.info(
                "Enabled block-prefix DeLS CUDA graph runner: "
                f"steps={steps}, triton_fused={graph_runner._can_triton_fuse}."
            )
    elif args.use_graph and markov_dels_local_head is not None:
        shift_label = bool(
            getattr(draft_model.config, "dflash_config", {}).get("shift_label", False)
        )
        K = block_size if shift_label else block_size - 1
        steps = K - 1
        if steps <= 0:
            logger.info(
                "--use-graph is ignored for Markov DeLS because there are "
                f"no Markov rollout steps (block_size={block_size})."
            )
        else:
            try:
                alpha_values = parse_scalar_schedule(
                    dels_alpha, steps, name="Markov DeLS alpha"
                )
                beta_values = parse_scalar_schedule(
                    dels_beta, steps, name="Markov DeLS beta"
                )
            except ValueError as exc:
                parser.error(str(exc))
            if int(markov_dels_local_head.vocab_size) != vocab_size:
                parser.error(
                    "Markov DeLS local-head vocab size mismatch: "
                    f"local_head={markov_dels_local_head.vocab_size}, "
                    f"target={vocab_size}."
                )
            graph_runner = MarkovDeLSGraphRunner(
                local_head=markov_dels_local_head,
                batch_size=1,
                steps=steps,
                vocab_size=vocab_size,
                alpha_values=alpha_values,
                beta_values=beta_values,
                baseline_logits=markov_dels_baseline_logits,
                device=device,
            )
            logger.info(
                "Enabled Markov DeLS CUDA graph runner: "
                f"steps={steps}, rank={markov_dels_local_head.rank}, "
                f"triton_fused={graph_runner._can_triton_fuse}."
            )
    elif args.use_graph and is_domino:
        shift_label = bool(
            getattr(draft_model.config, "dflash_config", {}).get("shift_label", False)
        )
        K = block_size if shift_label else block_size - 1
        steps = K - prefix_len

        graph_runner = DraftCorrectionGraphRunner(
            draft_model=draft_model,
            target_model=target,
            batch_size=1,
            steps=steps,
            hidden_dim=hidden_size,
            gru_hidden_dim=draft_model.prefix_gru.hidden_size,
            vocab_size=vocab_size,
            prefix_token_count=1 + prefix_len,
            device=device,
            guidance_alpha=args.domino_guidance_alpha,
        )
    elif args.use_graph:
        logger.info("--use-graph is ignored for non-Domino DFlash checkpoints.")

    use_bias = (
        args.use_bias
        and is_domino
        and dels_local_head is None
        and markov_dels_local_head is None
    )
    if args.use_bias and dels_local_head is not None:
        logger.info("--use-bias is ignored for block-prefix DeLS.")
    elif args.use_bias and markov_dels_local_head is not None:
        logger.info("--use-bias is ignored for Markov DeLS.")
    elif args.use_bias and not is_domino:
        logger.info("--use-bias is ignored for non-Domino DFlash checkpoints.")

    answers = []
    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        choice_b1 = {"index": 0, "block_size": 1, "turns": [], "new_tokens": [], "wall_time": [], "prefill_times": [], "decode_times": [], "acceptance_lengths": []}
        choice_bk = {"index": 1, "block_size": block_size, "turns": [], "new_tokens": [], "wall_time": [], "prefill_times": [], "decode_times": [], "acceptance_lengths": []}
        for turn_index, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            run_block_sizes = [block_size] if args.skip_baseline else [1, block_size]
            for bs in run_block_sizes:
                response[bs] = draft_model.spec_generate(
                    target=target,
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    block_size=bs,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    graph_runner=graph_runner,
                    use_bias=use_bias,
                    domino_guidance_alpha=args.domino_guidance_alpha,
                    dels_local_head=dels_local_head,
                    dels_alpha=dels_alpha,
                    dels_beta=dels_beta,
                    dels_baseline_logits=dels_baseline_logits,
                    markov_dels_local_head=markov_dels_local_head,
                    markov_dels_baseline_logits=markov_dels_baseline_logits,
                    return_dict=True,
                )

            # Record results for b=k, and for b=1 unless baseline was skipped.
            record_choices = [(choice_bk, block_size)]
            if not args.skip_baseline:
                record_choices.insert(0, (choice_b1, 1))
            for choice, bs in record_choices:
                r = response[bs]
                generated_ids = r.output_ids[0, r.num_input_tokens:]
                output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                choice["turns"].append(output_text)
                choice["new_tokens"].append(int(r.num_output_tokens))
                prefill_t = float(r.time_to_first_token)
                decode_t = float(r.time_per_output_token) * int(r.num_output_tokens)
                choice["prefill_times"].append(prefill_t)
                choice["decode_times"].append(decode_t)
                choice["wall_time"].append(prefill_t + decode_t)
                choice["acceptance_lengths"].append([int(x) for x in r.acceptance_lengths])

            # Use b=k result as conversation history (same as original logic)
            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})

        answers.append({
            "question_id": idx,
            "choices": [choice_b1, choice_bk],
            "tstamp": time.time(),
        })

    if dist.size() > 1:
        answers = dist.gather(answers, dst=0)
        if not dist.is_main():
            return
        answers = list(chain(*answers))

    answers.sort(key=lambda x: x["question_id"])

    # Write answer file
    if args.answer_file and dist.is_main():
        os.makedirs(os.path.dirname(args.answer_file) or ".", exist_ok=True)
        with open(args.answer_file, "w", encoding="utf-8") as f:
            for ans in answers:
                f.write(json.dumps(ans, ensure_ascii=False) + "\n")
        print(f"Saved answer file with {len(answers)} samples to {args.answer_file}")

    # Compute and print stats
    tb = np.mean([ans["choices"][1]["decode_times"][0] / max(1, ans["choices"][1]["new_tokens"][0]) for ans in answers if ans["choices"][1]["new_tokens"]])
    if args.skip_baseline:
        print("Decoding speedup: N/A (baseline skipped)")
    else:
        t1 = np.mean([ans["choices"][0]["decode_times"][0] / max(1, ans["choices"][0]["new_tokens"][0]) for ans in answers if ans["choices"][0]["new_tokens"]])
        print(f"Decoding speedup: {t1 / tb:.2f}")

    acceptance_lengths = list(chain(*[ans["choices"][1]["acceptance_lengths"][0] for ans in answers if ans["choices"][1]["acceptance_lengths"]]))
    tau_per_step = np.mean(acceptance_lengths) if acceptance_lengths else 0
    tau_per_sample = np.mean([np.mean(ans["choices"][1]["acceptance_lengths"][0]) for ans in answers if ans["choices"][1]["acceptance_lengths"]])
    print(f"Average Acceptance length (per step):  {tau_per_step:.2f}")
    print(f"Average Acceptance length (per sample): {tau_per_sample:.2f}")

    shift_label = bool(
        getattr(draft_model.config, "dflash_config", {}).get("shift_label", False)
    )
    if shift_label:
        histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 2)]
    else:
        histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")

if __name__ == "__main__":
    main()
