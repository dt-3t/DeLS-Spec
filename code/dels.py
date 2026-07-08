# coding=utf-8
"""Runtime helpers for DeLS block-prefix speculative decoding."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_VALID_RANK_ACTIVATIONS = {"identity", "silu"}


def load_unigram_log_prior(
    path: str,
    *,
    vocab_size: int,
    dtype: torch.dtype,
    device: torch.device,
    smoothing: float = 1.0,
) -> torch.Tensor:
    """Load a finite full-vocab log unigram prior as [1, 1, vocab_size]."""
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative.")

    stat_path = Path(path)
    if stat_path.is_dir():
        pt_path = stat_path / "loss_mask_unigram.pt"
        npz_path = stat_path / "loss_mask_unigram.npz"
        if pt_path.exists():
            stat_path = pt_path
        elif npz_path.exists():
            stat_path = npz_path
        else:
            raise ValueError(
                f"Unigram baseline directory {path!r} must contain "
                "loss_mask_unigram.pt or loss_mask_unigram.npz."
            )

    if stat_path.suffix == ".pt":
        stats = torch.load(stat_path, map_location="cpu", weights_only=False)
        if isinstance(stats, dict) and "counts" in stats:
            counts = stats["counts"].to(torch.float64)
        elif torch.is_tensor(stats):
            counts = stats.to(torch.float64)
        else:
            raise ValueError(
                f"Unsupported unigram baseline .pt payload in {stat_path}: "
                "expected a tensor or a dict with key 'counts'."
            )
    elif stat_path.suffix == ".npz":
        import numpy as np

        stats = np.load(stat_path)
        if "counts" not in stats.files:
            raise ValueError(f"Unigram baseline npz {stat_path} has no 'counts'.")
        counts = torch.from_numpy(stats["counts"]).to(torch.float64)
    else:
        raise ValueError(
            f"Unsupported unigram baseline path {stat_path}; expected .pt, .npz, or a directory."
        )

    counts = counts.reshape(-1)
    if counts.numel() > vocab_size:
        raise ValueError(
            f"Unigram baseline vocab size {counts.numel()} exceeds runtime vocab "
            f"size {vocab_size}."
        )
    if counts.numel() < vocab_size:
        padded = torch.zeros(vocab_size, dtype=counts.dtype)
        padded[: counts.numel()] = counts
        counts = padded

    if smoothing == 0.0:
        log_prior = torch.full_like(counts, -torch.inf)
        nonzero = counts > 0
        if not bool(nonzero.any()):
            raise ValueError("Unigram baseline has no positive counts.")
        total = counts.sum()
        log_prior[nonzero] = torch.log(counts[nonzero]) - torch.log(total)
        floor = log_prior[nonzero].min()
        log_prior[~nonzero] = floor
    else:
        smoothed = counts + float(smoothing)
        log_prior = torch.log(smoothed) - torch.log(smoothed.sum())

    return log_prior.to(device=device, dtype=dtype).view(1, 1, vocab_size)


def parse_scalar_schedule(
    values: str | float | Iterable[float], steps: int, *, name: str
) -> list[float]:
    """Parse a scalar or comma-separated per-position scalar schedule."""
    if isinstance(values, str):
        parsed = [float(part.strip()) for part in values.split(",") if part.strip()]
    elif isinstance(values, (float, int)):
        parsed = [float(values)]
    else:
        parsed = [float(v) for v in values]

    if not parsed:
        raise ValueError(f"{name} schedule cannot be empty.")
    if len(parsed) == 1:
        parsed = parsed * steps
    if len(parsed) != steps:
        raise ValueError(
            f"{name} schedule length must be 1 or {steps}, got {len(parsed)}."
        )
    return parsed


def detect_local_head_checkpoint_kind(
    checkpoint_path: str,
) -> Literal["dels", "markov"]:
    """Return the local-head runtime kind for a SpecForge checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "embedding_weight" in checkpoint and "lm_head_weight" in checkpoint:
        return "markov"

    state_dict = checkpoint.get("model_state_dict")
    if isinstance(state_dict, dict):
        if "down_proj" in state_dict and "up_proj" in state_dict:
            return "markov"
        if any(name.startswith("prefix_gru.") for name in state_dict):
            return "dels"

    config = checkpoint.get("model_config")
    if isinstance(config, dict) and "gru_hidden_dim" in config:
        return "dels"

    raise ValueError(
        "Cannot identify local-head checkpoint type. Expected a SpecForge "
        "local_head.pt, markov_local_head.pt, or merged_markov_local_head.pt."
    )


class DeLSLocalHead(nn.Module):
    """Inference module for SpecForge local-head checkpoints."""

    def __init__(
        self,
        *,
        target_model: nn.Module,
        vocab_size: int,
        embed_dim: int,
        gru_hidden_dim: int,
        low_rank_dim: int,
        lm_head_mode: str,
        rank_activation: str = "identity",
        dtype: torch.dtype,
        device: torch.device,
        target_lm_head: Optional[nn.Module] = None,
    ):
        super().__init__()
        if rank_activation not in _VALID_RANK_ACTIVATIONS:
            raise ValueError(
                f"rank_activation={rank_activation!r}; must be one of "
                f"{_VALID_RANK_ACTIVATIONS}"
            )
        self.embed_tokens = target_model.model.embed_tokens
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.low_rank_dim = int(low_rank_dim)
        self.lm_head_mode = lm_head_mode
        self.rank_activation = rank_activation

        self.prefix_gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bias=False,
        )
        self.rank_proj = nn.Linear(self.gru_hidden_dim, self.low_rank_dim, bias=False)
        self.low_rank_lm_head = nn.Linear(self.low_rank_dim, self.vocab_size, bias=False)
        self.rank_to_hidden = None
        self.target_lm_head = None

        if lm_head_mode == "target_lm_head":
            if target_lm_head is None:
                raise ValueError("target_lm_head is required for target_lm_head mode.")
            hidden_size = int(target_lm_head.weight.shape[1])
            self.rank_to_hidden = nn.Linear(self.low_rank_dim, hidden_size, bias=False)
            self.target_lm_head = target_lm_head

        self.to(device=device, dtype=dtype)
        self.eval()

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint_path: str,
        target_model: nn.Module,
        dtype: torch.dtype,
        device: torch.device,
        merged_lm_head_path: Optional[str] = None,
    ) -> "DeLSLocalHead":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("model_config")
        state_dict = checkpoint.get("model_state_dict")
        if config is None or state_dict is None:
            raise ValueError(
                "Local-head checkpoint must contain model_config and model_state_dict."
            )

        model = cls(
            target_model=target_model,
            vocab_size=int(config["vocab_size"]),
            embed_dim=int(config["embed_dim"]),
            gru_hidden_dim=int(config["gru_hidden_dim"]),
            low_rank_dim=int(config["low_rank_dim"]),
            lm_head_mode=config.get("lm_head_mode", "low_rank"),
            rank_activation=config.get("rank_activation", "identity"),
            dtype=dtype,
            device=device,
            target_lm_head=target_model.lm_head,
        )
        model.pure_draft_prefix_len = int(config.get("pure_draft_prefix_len", 1))
        model.shift_label = bool(config.get("shift_label", False))

        own_state = model.state_dict()
        filtered = {}
        skipped = []
        for name, tensor in state_dict.items():
            if name in own_state and own_state[name].shape == tensor.shape:
                filtered[name] = tensor.to(dtype=own_state[name].dtype)
            elif name.startswith(("embed_tokens.", "target_lm_head.")):
                skipped.append(name)
            else:
                raise ValueError(
                    f"Unexpected local-head tensor {name} with shape {tuple(tensor.shape)}."
                )

        if merged_lm_head_path is not None:
            merged = torch.load(
                merged_lm_head_path, map_location="cpu", weights_only=False
            )
            weight = merged["weight"] if isinstance(merged, dict) else merged
            filtered["low_rank_lm_head.weight"] = weight.to(
                dtype=own_state["low_rank_lm_head.weight"].dtype
            )
            model.lm_head_mode = "low_rank"

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        missing = [
            name
            for name in missing
            if not name.startswith(("embed_tokens.", "target_lm_head."))
            and not (
                model.lm_head_mode == "low_rank" and name.startswith("rank_to_hidden.")
            )
            and not (
                model.lm_head_mode == "target_lm_head"
                and name.startswith("low_rank_lm_head.")
            )
        ]
        if missing or unexpected:
            raise ValueError(
                f"Local-head checkpoint mismatch: missing={missing}, "
                f"unexpected={unexpected}, skipped_frozen={skipped}"
            )
        if model.lm_head_mode == "target_lm_head":
            model.fuse_target_lm_head_()

        return model

    @torch.inference_mode()
    def fuse_target_lm_head_(self) -> None:
        if self.lm_head_mode != "target_lm_head":
            return
        if self.rank_to_hidden is None or self.target_lm_head is None:
            raise ValueError("Cannot fuse local head without rank_to_hidden/target_lm_head.")

        target_weight = self.target_lm_head.weight.detach().to(torch.float32)
        up_weight = self.rank_to_hidden.weight.detach().to(torch.float32)
        fused_weight = target_weight.matmul(up_weight).to(
            self.low_rank_lm_head.weight.dtype
        )
        self.low_rank_lm_head.weight.copy_(fused_weight)
        self.rank_to_hidden = None
        self.target_lm_head = None
        self.lm_head_mode = "low_rank"

    @torch.inference_mode()
    def init_empty_hidden(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            1,
            batch_size,
            self.gru_hidden_dim,
            device=device,
            dtype=self.rank_proj.weight.dtype,
        )

    @torch.inference_mode()
    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        rank_state = self.rank_proj(hidden.transpose(0, 1))
        if self.rank_activation == "silu":
            rank_state = F.silu(rank_state)
        return self.low_rank_lm_head(rank_state)

    @torch.inference_mode()
    def advance(self, token_ids: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        token_embeds = self.embed_tokens(token_ids)
        _, hidden = self.prefix_gru(token_embeds, hidden)
        return hidden


class MarkovDeLSLocalHead(nn.Module):
    """Inference module for dense Markov local-head checkpoints."""

    def __init__(
        self,
        *,
        embedding_weight: torch.Tensor,
        lm_head_weight: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        if embedding_weight.ndim != 2 or lm_head_weight.ndim != 2:
            raise ValueError("Markov DeLS weights must be rank-2 tensors.")
        if embedding_weight.shape[1] != lm_head_weight.shape[1]:
            raise ValueError(
                "Markov DeLS merged embedding/lm-head rank mismatch: "
                f"embedding={tuple(embedding_weight.shape)}, "
                f"lm_head={tuple(lm_head_weight.shape)}"
            )

        self.embedding_weight = nn.Parameter(
            embedding_weight.to(device=device, dtype=dtype), requires_grad=False
        )
        self.lm_head_weight = nn.Parameter(
            lm_head_weight.to(device=device, dtype=dtype), requires_grad=False
        )
        self.vocab_size = int(lm_head_weight.shape[0])
        self.rank = int(lm_head_weight.shape[1])
        self.eval()

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint_path: str,
        target_model: nn.Module,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "MarkovDeLSLocalHead":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "embedding_weight" in checkpoint and "lm_head_weight" in checkpoint:
            embedding_weight = checkpoint["embedding_weight"]
            lm_head_weight = checkpoint["lm_head_weight"]
        else:
            config = checkpoint.get("model_config")
            state_dict = checkpoint.get("model_state_dict")
            if config is None or state_dict is None:
                raise ValueError(
                    "Markov DeLS checkpoint must be either "
                    "merged_markov_local_head.pt or markov_local_head.pt from "
                    "SpecForge scripts/train_markov_local_head.py."
                )
            if "down_proj" not in state_dict or "up_proj" not in state_dict:
                raise ValueError(
                    "Markov local-head checkpoint must contain down_proj and up_proj."
                )

            target_embedding = target_model.model.embed_tokens.weight.detach().cpu()
            target_lm_head = target_model.lm_head.weight.detach().cpu()
            down_proj = state_dict["down_proj"].to(torch.float32)
            up_proj = state_dict["up_proj"].to(torch.float32)
            embedding_weight = target_embedding.to(torch.float32).matmul(down_proj)
            lm_head_weight = target_lm_head.to(torch.float32).matmul(up_proj.t())

            expected_rank = int(config.get("rank", down_proj.shape[1]))
            if embedding_weight.shape[1] != expected_rank:
                raise ValueError(
                    f"Unexpected Markov rank: merged={embedding_weight.shape[1]}, "
                    f"config={expected_rank}"
                )

        return cls(
            embedding_weight=embedding_weight,
            lm_head_weight=lm_head_weight,
            dtype=dtype,
            device=device,
        )

    @torch.inference_mode()
    def logits_from_token(self, token_ids: torch.Tensor) -> torch.Tensor:
        rank_states = torch.nn.functional.embedding(token_ids, self.embedding_weight)
        return torch.matmul(rank_states, self.lm_head_weight.t())


class PositionMixJointHead(nn.Module):
    """Runtime for block-prefix position joint finetune checkpoints."""

    def __init__(
        self,
        *,
        target_model: nn.Module,
        vocab_size: int,
        embed_dim: int,
        gru_hidden_dim: int,
        low_rank_dim: int,
        suffix_len: int,
        shift_label: bool,
        rank_activation: str,
        baseline_logits: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        if rank_activation not in _VALID_RANK_ACTIVATIONS:
            raise ValueError(
                f"rank_activation={rank_activation!r}; must be one of "
                f"{_VALID_RANK_ACTIVATIONS}"
            )
        self.embed_tokens = target_model.model.embed_tokens
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.low_rank_dim = int(low_rank_dim)
        self.pure_draft_prefix_len = 1
        self.shift_label = bool(shift_label)
        self.rank_activation = rank_activation
        self.rnn_input = "block_prefix"
        self.mix_form = "block_prefix_position"

        self.prefix_gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bias=False,
        )
        self.rank_proj = nn.Linear(self.gru_hidden_dim, self.low_rank_dim, bias=False)
        self.low_rank_lm_head = nn.Linear(self.low_rank_dim, self.vocab_size, bias=False)
        self.position_gates = nn.Parameter(
            torch.zeros(int(suffix_len), 2, dtype=dtype, device=device)
        )
        self.register_buffer(
            "baseline_logits",
            baseline_logits.to(device=device, dtype=dtype).view(1, 1, -1),
            persistent=True,
        )
        self.to(device=device, dtype=dtype)
        self.eval()

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint_path: str,
        target_model: nn.Module,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "PositionMixJointHead":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("model_config")
        state_dict = checkpoint.get("model_state_dict")
        if config is None or state_dict is None:
            raise ValueError(
                "Block-prefix position checkpoint must contain model_config and "
                "model_state_dict."
            )
        if config.get("method") != "block_prefix_position_mix_joint":
            raise ValueError(
                "Only block_prefix_position_mix_joint checkpoints are supported."
            )
        if config.get("rnn_input", "block_prefix") != "block_prefix":
            raise ValueError("Only block_prefix position-joint checkpoints are supported.")

        local_config = config.get("local_head_config")
        if local_config is None:
            raise ValueError("Checkpoint is missing local_head_config.")
        if "baseline_logits" not in state_dict:
            raise ValueError("Block-prefix position checkpoint is missing baseline_logits.")

        shift_label = bool(config.get("dflash_shift_label", False))
        suffix_start = 1 if shift_label else 2
        suffix_len = int(block_size) - suffix_start
        if suffix_len <= 0:
            raise ValueError(
                "Block-prefix position checkpoint has no suffix positions: "
                f"block_size={block_size}, suffix_start={suffix_start}."
            )

        model = cls(
            target_model=target_model,
            vocab_size=int(local_config["vocab_size"]),
            embed_dim=int(local_config["embed_dim"]),
            gru_hidden_dim=int(local_config["gru_hidden_dim"]),
            low_rank_dim=int(local_config["low_rank_dim"]),
            suffix_len=suffix_len,
            shift_label=shift_label,
            rank_activation=local_config.get("rank_activation", "identity"),
            baseline_logits=state_dict["baseline_logits"],
            dtype=dtype,
            device=device,
        )

        own_state = model.state_dict()
        filtered = {}
        skipped = []
        for name, tensor in state_dict.items():
            runtime_name = None
            if name.startswith("local_head_model."):
                runtime_name = name[len("local_head_model.") :]
            elif name in {"position_gates", "baseline_logits"}:
                runtime_name = name

            if runtime_name is None:
                skipped.append(name)
                continue
            if runtime_name.startswith(("embed_tokens.", "target_lm_head.")):
                skipped.append(name)
                continue
            if runtime_name == "baseline_logits":
                filtered[runtime_name] = tensor.reshape_as(own_state[runtime_name]).to(
                    dtype=own_state[runtime_name].dtype
                )
            elif runtime_name in own_state and own_state[runtime_name].shape == tensor.shape:
                filtered[runtime_name] = tensor.to(dtype=own_state[runtime_name].dtype)
            elif runtime_name == "rank_to_hidden.weight":
                local_mode = local_config.get("lm_head_mode", "low_rank")
                if local_mode != "target_lm_head":
                    raise ValueError(
                        "Unexpected rank_to_hidden.weight for block-prefix joint "
                        f"local lm_head_mode={local_mode!r}."
                    )
                target_weight = target_model.lm_head.weight.detach().to(torch.float32)
                fused_weight = target_weight.matmul(tensor.to(torch.float32))
                filtered["low_rank_lm_head.weight"] = fused_weight.to(
                    dtype=own_state["low_rank_lm_head.weight"].dtype
                )
            else:
                raise ValueError(
                    f"Unexpected block-prefix joint tensor {name} mapped to "
                    f"{runtime_name} with shape {tuple(tensor.shape)}."
                )

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        missing = [
            name
            for name in missing
            if not name.startswith(("embed_tokens.", "target_lm_head."))
        ]
        if missing or unexpected:
            raise ValueError(
                f"Block-prefix position checkpoint mismatch: missing={missing}, "
                f"unexpected={unexpected}, skipped={skipped}"
            )
        return model

    @torch.inference_mode()
    def init_empty_hidden(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            1,
            batch_size,
            self.gru_hidden_dim,
            device=device,
            dtype=self.rank_proj.weight.dtype,
        )

    @torch.inference_mode()
    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        rank_state = self.rank_proj(hidden.transpose(0, 1))
        if self.rank_activation == "silu":
            rank_state = F.silu(rank_state)
        return self.low_rank_lm_head(rank_state)

    @torch.inference_mode()
    def alpha_beta_schedules(self) -> tuple[list[float], list[float]]:
        gates = self.position_gates.detach().to(device="cpu", dtype=torch.float32)
        return gates[:, 0].tolist(), gates[:, 1].tolist()

    @torch.inference_mode()
    def advance(self, token_ids: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        token_embeds = self.embed_tokens(token_ids)
        _, hidden = self.prefix_gru(token_embeds, hidden)
        return hidden
