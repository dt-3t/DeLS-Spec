import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _fused_markov_argmax_kernel(
        rank_ptr,
        lm_head_ptr,
        base_ptr,
        baseline_ptr,
        out_val_ptr,
        out_idx_ptr,
        R,
        V,
        stride_rank_b,
        stride_rank_r,
        stride_lm_v,
        stride_lm_r,
        stride_base_b,
        stride_base_v,
        stride_base_t,
        stride_baseline_v,
        stride_out_b,
        stride_out_v,
        STEP: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_R: tl.constexpr,
        ALPHA: tl.constexpr,
        BETA: tl.constexpr,
        HAS_BASELINE: tl.constexpr,
    ):
        pid_v = tl.program_id(0)
        pid_b = tl.program_id(1)

        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V
        acc = tl.zeros([BLOCK_V], dtype=tl.float32)

        for r_start in range(0, R, BLOCK_R):
            offs_r = r_start + tl.arange(0, BLOCK_R)
            mask_r = offs_r < R
            rank_vals = tl.load(
                rank_ptr + pid_b * stride_rank_b + offs_r * stride_rank_r,
                mask=mask_r,
                other=0.0,
            ).to(tl.float32)
            weight_vals = tl.load(
                lm_head_ptr
                + offs_v[:, None] * stride_lm_v
                + offs_r[None, :] * stride_lm_r,
                mask=mask_v[:, None] & mask_r[None, :],
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
            acc += tl.sum(weight_vals * rank_vals[None, :], axis=1)

        acc *= ALPHA

        base_vals = tl.load(
            base_ptr
            + pid_b * stride_base_b
            + STEP * stride_base_t
            + offs_v * stride_base_v,
            mask=mask_v,
            other=-float("inf"),
            cache_modifier=".cg",
        ).to(tl.float32)
        acc += base_vals

        if HAS_BASELINE:
            baseline_vals = tl.load(
                baseline_ptr + offs_v * stride_baseline_v,
                mask=mask_v,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
            acc -= BETA * baseline_vals

        acc = tl.where(mask_v, acc, -float("inf"))

        local_max_val = tl.max(acc, axis=0)
        local_max_idx = tl.argmax(acc, axis=0)
        global_max_idx = pid_v * BLOCK_V + local_max_idx

        out_offs = pid_b * stride_out_b + pid_v * stride_out_v
        tl.store(out_val_ptr + out_offs, local_max_val)
        tl.store(out_idx_ptr + out_offs, global_max_idx)

    @triton.jit
    def _reduce_argmax_kernel(
        out_val_ptr,
        out_idx_ptr,
        final_token_ptr,
        N,
        stride_val_b,
        stride_val_n,
        stride_idx_b,
        stride_idx_n,
        stride_final_b,
        BLOCK_N: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        best_val = -float("inf")
        best_idx = 0

        for start_n in range(0, N, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N
            vals = tl.load(
                out_val_ptr + pid_b * stride_val_b + offs_n * stride_val_n,
                mask=mask_n,
                other=-float("inf"),
            )
            local_pos = tl.argmax(vals, axis=0)
            local_val = tl.max(vals, axis=0)
            local_idx = tl.load(
                out_idx_ptr
                + pid_b * stride_idx_b
                + (start_n + local_pos) * stride_idx_n
            )

            take = local_val > best_val
            best_val = tl.where(take, local_val, best_val)
            best_idx = tl.where(take, local_idx, best_idx)

        tl.store(final_token_ptr + pid_b * stride_final_b, best_idx)


class MarkovDeLSGraphRunner:
    """CUDA Graph runner for greedy Markov DeLS suffix rollout."""

    def __init__(
        self,
        *,
        local_head,
        batch_size: int,
        steps: int,
        vocab_size: int,
        alpha_values: list[float],
        beta_values: list[float],
        baseline_logits: torch.Tensor | None,
        device: torch.device,
    ):
        if steps <= 0:
            raise ValueError(
                f"MarkovDeLSGraphRunner requires steps > 0, got {steps}."
            )
        if len(alpha_values) != steps or len(beta_values) != steps:
            raise ValueError(
                "Markov DeLS graph alpha/beta schedules must match steps: "
                f"steps={steps}, alpha={len(alpha_values)}, beta={len(beta_values)}."
            )
        if int(local_head.vocab_size) != int(vocab_size):
            raise ValueError(
                "Markov DeLS local-head vocab size mismatch: "
                f"local_head={local_head.vocab_size}, vocab={vocab_size}."
            )
        if any(float(beta) != 0.0 for beta in beta_values) and baseline_logits is None:
            raise ValueError(
                "Markov DeLS graph requires baseline_logits when any beta is nonzero."
            )

        self.batch_size = int(batch_size)
        self.steps = int(steps)
        self.vocab_size = int(vocab_size)
        self.alpha_values = [float(x) for x in alpha_values]
        self.beta_values = [float(x) for x in beta_values]
        self.embedding_weight = (
            local_head.embedding_weight.detach().to(device=device).contiguous()
        )
        self.lm_head_weight = (
            local_head.lm_head_weight.detach().to(device=device).contiguous()
        )
        self.rank = int(self.lm_head_weight.shape[1])
        self._can_triton_fuse = HAS_TRITON
        dtype = self.lm_head_weight.dtype

        baseline_flat = torch.zeros(self.vocab_size, dtype=dtype, device=device)
        if baseline_logits is not None:
            baseline_numel = int(baseline_logits.numel())
            if baseline_numel != self.vocab_size:
                raise ValueError(
                    "Markov DeLS graph baseline vocab size mismatch: "
                    f"baseline={baseline_numel}, vocab={self.vocab_size}."
                )
            baseline_flat.copy_(
                baseline_logits.to(device=device, dtype=dtype).reshape(-1)
            )
        self.baseline_logits = baseline_flat.contiguous()
        self.has_baseline = baseline_logits is not None

        self.static_prev_ids = torch.zeros(
            self.batch_size, 1, dtype=torch.long, device=device
        )
        self.static_base_logits = torch.zeros(
            self.batch_size, self.steps, self.vocab_size, dtype=dtype, device=device
        )
        self.static_out = torch.zeros(
            self.batch_size, self.steps, dtype=torch.long, device=device
        )

        self._BLOCK_V = 512
        self._num_v_blocks = (self.vocab_size + self._BLOCK_V - 1) // self._BLOCK_V
        self._triton_out_val = torch.zeros(
            self.batch_size, self._num_v_blocks, dtype=torch.float32, device=device
        )
        self._triton_out_idx = torch.zeros(
            self.batch_size, self._num_v_blocks, dtype=torch.int32, device=device
        )
        self._triton_final = torch.zeros(
            self.batch_size, dtype=torch.int64, device=device
        )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(stream):
            for _ in range(3):
                self._forward_impl()
        torch.cuda.current_stream().wait_stream(stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self._forward_impl()

    def _argmax_step_triton(
        self,
        rank_state: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        alpha = self.alpha_values[int(step_index)]
        beta = self.beta_values[int(step_index)]
        has_baseline = self.has_baseline and beta != 0.0

        grid = (self._num_v_blocks, self.batch_size)
        _fused_markov_argmax_kernel[grid](
            rank_state,
            self.lm_head_weight,
            self.static_base_logits,
            self.baseline_logits,
            self._triton_out_val,
            self._triton_out_idx,
            self.rank,
            self.vocab_size,
            rank_state.stride(0),
            rank_state.stride(1),
            self.lm_head_weight.stride(0),
            self.lm_head_weight.stride(1),
            self.static_base_logits.stride(0),
            self.static_base_logits.stride(2),
            self.static_base_logits.stride(1),
            self.baseline_logits.stride(0),
            self._triton_out_val.stride(0),
            self._triton_out_val.stride(1),
            STEP=int(step_index),
            BLOCK_V=self._BLOCK_V,
            BLOCK_R=32,
            ALPHA=alpha,
            BETA=beta,
            HAS_BASELINE=has_baseline,
        )
        _reduce_argmax_kernel[(self.batch_size,)](
            self._triton_out_val,
            self._triton_out_idx,
            self._triton_final,
            self._num_v_blocks,
            self._triton_out_val.stride(0),
            self._triton_out_val.stride(1),
            self._triton_out_idx.stride(0),
            self._triton_out_idx.stride(1),
            self._triton_final.stride(0),
            BLOCK_N=512,
        )
        return self._triton_final

    def _forward_impl(self):
        current_token = self.static_prev_ids[:, 0]
        for step_index in range(self.steps):
            rank_state = F.embedding(current_token, self.embedding_weight)
            if self._can_triton_fuse:
                current_token = self._argmax_step_triton(rank_state, step_index)
            else:
                short_logits = F.linear(rank_state, self.lm_head_weight, None)
                logits = (
                    self.static_base_logits[:, step_index, :]
                    + self.alpha_values[step_index] * short_logits
                )
                beta = self.beta_values[step_index]
                if self.has_baseline and beta != 0.0:
                    logits = logits - beta * self.baseline_logits.view(1, -1)
                current_token = torch.argmax(logits, dim=-1)
            self.static_out[:, step_index] = current_token

    @torch.inference_mode()
    def __call__(
        self,
        prev_token_ids: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        self.static_prev_ids.copy_(prev_token_ids)
        self.static_base_logits.copy_(base_logits)
        self.graph.replay()
        return self.static_out.clone()
