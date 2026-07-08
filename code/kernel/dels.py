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
    def _fused_low_rank_argmax_kernel(
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

    @triton.jit
    def _fused_gru_cell_kernel(
        gi_ptr,
        gh_ptr,
        h_ptr,
        h_out_ptr,
        B,
        G,
        stride_gi_b,
        stride_gi_g,
        stride_gh_b,
        stride_gh_g,
        stride_h_b,
        stride_h_g,
        stride_hout_b,
        stride_hout_g,
        BLOCK_G: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_g = tl.program_id(1)

        offs_g = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
        mask_g = offs_g < G

        h_state = tl.load(
            h_ptr + pid_b * stride_h_b + offs_g * stride_h_g,
            mask=mask_g,
            other=0.0,
        ).to(tl.float32)

        gi_r = tl.load(
            gi_ptr + pid_b * stride_gi_b + offs_g * stride_gi_g,
            mask=mask_g,
            other=0.0,
        ).to(tl.float32)
        gi_z = tl.load(
            gi_ptr + pid_b * stride_gi_b + (G + offs_g) * stride_gi_g,
            mask=mask_g,
            other=0.0,
        ).to(tl.float32)
        gi_n = tl.load(
            gi_ptr + pid_b * stride_gi_b + (2 * G + offs_g) * stride_gi_g,
            mask=mask_g,
            other=0.0,
        ).to(tl.float32)

        gh_r = tl.load(
            gh_ptr + pid_b * stride_gh_b + offs_g * stride_gh_g,
            mask=mask_g,
            other=0.0,
        ).to(tl.float32)
        gh_z = tl.load(
            gh_ptr + pid_b * stride_gh_b + (G + offs_g) * stride_gh_g,
            mask=mask_g,
            other=0.0,
        ).to(tl.float32)
        gh_n = tl.load(
            gh_ptr + pid_b * stride_gh_b + (2 * G + offs_g) * stride_gh_g,
            mask=mask_g,
            other=0.0,
        ).to(tl.float32)

        reset_gate = tl.sigmoid(gi_r + gh_r)
        update_gate = tl.sigmoid(gi_z + gh_z)
        new_gate = 2.0 * tl.sigmoid(2.0 * (gi_n + reset_gate * gh_n)) - 1.0
        h_new = (1.0 - update_gate) * new_gate + update_gate * h_state

        tl.store(
            h_out_ptr + pid_b * stride_hout_b + offs_g * stride_hout_g,
            h_new.to(h_ptr.dtype.element_ty),
            mask=mask_g,
        )


class DeLSGraphRunner:
    """
    CUDA Graph runner for block-prefix DeLS greedy correction.

    Captures the fixed-shape suffix rollout used by HF benchmark. The Triton
    path fuses low-rank LM head projection, long-logit/baseline mixing, and
    argmax for each suffix step.
    """

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
            raise ValueError(f"DeLSGraphRunner requires steps > 0, got {steps}.")
        if len(alpha_values) != steps or len(beta_values) != steps:
            raise ValueError(
                "DeLS graph alpha/beta schedules must match steps: "
                f"steps={steps}, alpha={len(alpha_values)}, beta={len(beta_values)}."
            )
        if any(float(beta) != 0.0 for beta in beta_values) and baseline_logits is None:
            raise ValueError("DeLS graph requires baseline_logits when any beta is nonzero.")

        self.local_head = local_head
        self.batch_size = int(batch_size)
        self.steps = int(steps)
        self.vocab_size = int(vocab_size)
        self.alpha_values = [float(x) for x in alpha_values]
        self.beta_values = [float(x) for x in beta_values]

        dtype = local_head.rank_proj.weight.dtype
        self.rank_dim = int(local_head.low_rank_lm_head.weight.shape[1])
        self._can_triton_fuse = HAS_TRITON

        gru = local_head.prefix_gru
        if gru.num_layers != 1 or gru.bidirectional:
            raise ValueError("DeLS graph runner supports only a single unidirectional GRU.")
        self.gru_hidden_dim = int(gru.hidden_size)
        self.gru_w_ih = gru.weight_ih_l0.detach().contiguous()
        self.gru_w_hh = gru.weight_hh_l0.detach().contiguous()
        self.gru_b_ih = gru.bias_ih_l0.detach().contiguous() if gru.bias else None
        self.gru_b_hh = gru.bias_hh_l0.detach().contiguous() if gru.bias else None

        embed_weight = local_head.embed_tokens.weight
        self._gru_input_proj_table = F.linear(
            embed_weight, self.gru_w_ih, self.gru_b_ih
        ).contiguous()

        self.rank_weight = local_head.rank_proj.weight.detach().contiguous()
        self.rank_bias = (
            local_head.rank_proj.bias.detach().contiguous()
            if local_head.rank_proj.bias is not None
            else None
        )
        self.lm_head_weight = local_head.low_rank_lm_head.weight.detach().contiguous()

        baseline_flat = torch.zeros(self.vocab_size, dtype=dtype, device=device)
        if baseline_logits is not None:
            baseline_numel = int(baseline_logits.numel())
            if baseline_numel != self.vocab_size:
                raise ValueError(
                    "DeLS graph baseline vocab size mismatch: "
                    f"baseline={baseline_numel}, vocab={self.vocab_size}."
                )
            baseline_flat.copy_(
                baseline_logits.to(device=device, dtype=dtype).reshape(-1)
            )
        self.baseline_logits = baseline_flat.contiguous()
        self.has_baseline = baseline_logits is not None

        self.static_prefix_ids = torch.zeros(
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
        self._gru_h_out_a = torch.zeros(
            self.batch_size, self.gru_hidden_dim, dtype=dtype, device=device
        )
        self._gru_h_out_b = torch.zeros(
            self.batch_size, self.gru_hidden_dim, dtype=dtype, device=device
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

    def _fused_gru_cell(
        self,
        gi: torch.Tensor,
        gh: torch.Tensor,
        h_state: torch.Tensor,
        h_out: torch.Tensor,
    ) -> torch.Tensor:
        block_g = 256
        grid = (self.batch_size, (self.gru_hidden_dim + block_g - 1) // block_g)
        _fused_gru_cell_kernel[grid](
            gi,
            gh,
            h_state,
            h_out,
            self.batch_size,
            self.gru_hidden_dim,
            gi.stride(0),
            gi.stride(1),
            gh.stride(0),
            gh.stride(1),
            h_state.stride(0),
            h_state.stride(1),
            h_out.stride(0),
            h_out.stride(1),
            BLOCK_G=block_g,
        )
        return h_out

    def _gru_cell(
        self,
        gi: torch.Tensor,
        h_state: torch.Tensor,
        h_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gh = F.linear(h_state, self.gru_w_hh, self.gru_b_hh)
        if self._can_triton_fuse and h_out is not None:
            return self._fused_gru_cell(gi, gh, h_state, h_out)
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        return (1.0 - update_gate) * new_gate + update_gate * h_state

    def _argmax_step_triton(
        self,
        rank_state: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        alpha = self.alpha_values[int(step_index)]
        beta = self.beta_values[int(step_index)]
        has_baseline = self.has_baseline and beta != 0.0

        grid = (self._num_v_blocks, self.batch_size)
        _fused_low_rank_argmax_kernel[grid](
            rank_state,
            self.lm_head_weight,
            self.static_base_logits,
            self.baseline_logits,
            self._triton_out_val,
            self._triton_out_idx,
            self.rank_dim,
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
        h_state = torch.zeros(
            self.batch_size,
            self.gru_hidden_dim,
            dtype=self.static_base_logits.dtype,
            device=self.static_base_logits.device,
        )
        gi = self._gru_input_proj_table[self.static_prefix_ids[:, 0]]
        h_state = self._gru_cell(gi, h_state, self._gru_h_out_a)
        next_h_out = self._gru_h_out_b

        for step_index in range(self.steps):
            rank_state = F.linear(h_state, self.rank_weight, self.rank_bias)
            if self.local_head.rank_activation == "silu":
                rank_state = F.silu(rank_state)
            elif self.local_head.rank_activation != "identity":
                raise ValueError(
                    f"Unsupported DeLS rank_activation={self.local_head.rank_activation!r}."
                )

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

            if step_index + 1 < self.steps:
                gi = self._gru_input_proj_table[current_token]
                h_state = self._gru_cell(gi, h_state, next_h_out)
                next_h_out = (
                    self._gru_h_out_b
                    if next_h_out is self._gru_h_out_a
                    else self._gru_h_out_a
                )

    @torch.inference_mode()
    def __call__(
        self,
        prefix_ids: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        self.static_prefix_ids.copy_(prefix_ids)
        self.static_base_logits.copy_(base_logits)
        self.graph.replay()
        return self.static_out.clone()
