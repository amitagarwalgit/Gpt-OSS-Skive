"""Fused (Triton) eviction scoring — from the team's ``attention.py`` notebook.

Two things live here:

1. ``evict_fwd_kernel`` / ``triton_evict_attention_forward`` — the team's
   original fused kernel, kept as provided. It computes the decode attention
   output AND a per-token eviction loss ``|p_t * v_t|`` in one pass over a dense
   ``[B, H_kv, S, D]`` cache, then returns ``argmin`` as the token to evict.
   NOTE: it takes ``exp(score)`` without subtracting the running max, so it can
   overflow float32 for large logits; use it for reference / HF-style dense
   caches. It has no attention-sink term.

2. ``skive_token_score_kernel`` / ``score_tokens_fused`` — a numerically-stable,
   sink-aware *scoring-only* variant that SKIVE uses inside vLLM. It returns the
   normalized per-(batch, head, token) value-attention score
   ``p_t * mean_d |v_{t,d}|`` (the SKIVE metric) so the caller can apply any
   head aggregation and block aggregation. Two passes (max, then exp/sum) so it
   never overflows; the optional sink logit joins the softmax denominator
   exactly as gpt-oss's attention does (``s_aux`` in FlashAttention-3).

Both operate on DENSE tensors; ``integration._batch_score_va`` gathers the
request's paged KV blocks into that layout before calling ``score_tokens_fused``.
Triton is optional -- importing this module without Triton is fine, only the
kernels become unavailable (``HAS_TRITON`` is False).
"""

from __future__ import annotations

import torch

try:  # Triton is a vLLM dependency on GPU, but keep CPU/test imports working.
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - CPU-only environments
    HAS_TRITON = False


if HAS_TRITON:

    # ------------------------------------------------------------------
    # 1. The team's original fused attention + eviction-loss kernel (as given).
    # ------------------------------------------------------------------
    @triton.jit
    def evict_fwd_kernel(
        # Inputs
        Q_ptr, K_ptr, V_ptr, Mask_ptr,
        # Outputs
        AttnOutput_ptr, TempLoss_ptr, L_ptr,
        # Parameters
        scaling,
        N_REP,
        N_Q_HEADS,  # Total number of query heads
        # Strides
        stride_q_b, stride_q_h, stride_q_d,
        stride_k_b, stride_k_h_kv, stride_k_s, stride_k_d,
        stride_v_b, stride_v_h_kv, stride_v_s, stride_v_d,
        stride_mask_b, stride_mask_s,
        stride_ao_b, stride_ao_h, stride_ao_d,
        stride_loss_b, stride_loss_h, stride_loss_s,
        stride_l_b, stride_l_h, stride_l_s,
        # Compile-time constants for tensor dimensions
        D_HEAD: tl.constexpr,
        SEQ_LEN_KV: tl.constexpr,
        BLOCK_S: tl.constexpr,  # Block size for SEQ_LEN_KV dimension
    ):
        # Program IDs for batch and query head
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        # --- Common offsets for D_HEAD dimension ---
        d_offs = tl.arange(0, D_HEAD)

        # --- Load Query vector ---
        # Q is (B, N_Q_H, 1, D_HEAD). We only care about the single sequence element.
        q_base_ptr = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
        q_vec = tl.load(q_base_ptr + d_offs * stride_q_d)  # Shape: (D_HEAD,)

        # --- Determine Key/Value head index ---
        kv_head_idx = pid_h // N_REP

        # --- Base pointers for K, V, Mask for the current batch item and KV head ---
        k_batch_head_base_ptr = K_ptr + pid_b * stride_k_b + kv_head_idx * stride_k_h_kv
        v_batch_head_base_ptr = V_ptr + pid_b * stride_v_b + kv_head_idx * stride_v_h_kv
        mask_batch_base_ptr = Mask_ptr + pid_b * stride_mask_b
        loss_base_ptr = TempLoss_ptr + pid_b * stride_loss_b + pid_h * stride_loss_h

        # --- Accumulators for Online Softmax and Attention Output ---
        acc_o = tl.zeros((D_HEAD,), dtype=tl.float32)
        l_i = tl.zeros((1,), dtype=tl.float32)

        # === PASS 1: Compute Attention Output using Online Softmax ===
        for s_start_offset in range(0, SEQ_LEN_KV, BLOCK_S):
            s_block_offs = s_start_offset + tl.arange(0, BLOCK_S)
            s_boundary_mask = s_block_offs < SEQ_LEN_KV
            k_s_ptrs = k_batch_head_base_ptr + s_block_offs[:, None] * stride_k_s
            k_block_ptrs = k_s_ptrs + d_offs[None, :] * stride_k_d
            k_block = tl.load(k_block_ptrs, mask=s_boundary_mask[:, None], other=0.0)
            scores_block = tl.sum(
                q_vec[None, :].to(tl.float32) * k_block.to(tl.float32), axis=1
            ) * scaling

            mask_s_ptrs = mask_batch_base_ptr + s_block_offs * stride_mask_s
            attn_mask_block_vec = tl.load(mask_s_ptrs, mask=s_boundary_mask, other=False)
            scores_block = tl.where(
                attn_mask_block_vec & s_boundary_mask, scores_block, float(-1e35)
            )

            p_block_numerators = tl.exp(scores_block)
            l_i += tl.sum(p_block_numerators, axis=0)

            v_s_ptrs = v_batch_head_base_ptr + s_block_offs[:, None] * stride_v_s
            v_block_ptrs = v_s_ptrs + d_offs[None, :] * stride_v_d
            v_block = tl.load(v_block_ptrs, mask=s_boundary_mask[:, None], other=0.0)

            attn_output_tmp = p_block_numerators[:, None] * v_block.to(tl.float32)
            acc_o += tl.sum(attn_output_tmp, axis=0)
            attn_output_loss = tl.sum(tl.abs(attn_output_tmp), axis=1) / D_HEAD
            tl.store(
                loss_base_ptr + s_block_offs * stride_loss_s,
                attn_output_loss,
                mask=s_boundary_mask,
            )

        attn_output_vec = (acc_o / l_i).to(Q_ptr.dtype.element_ty)

        ao_base_ptr = AttnOutput_ptr + pid_b * stride_ao_b + pid_h * stride_ao_h
        tl.store(ao_base_ptr + d_offs * stride_ao_d, attn_output_vec)

        tl.store(
            L_ptr + pid_b * stride_l_b + pid_h * stride_l_h + tl.arange(0, 1) * stride_l_s,
            l_i,
        )

    # ------------------------------------------------------------------
    # 2. Stable, sink-aware SCORING-only kernel used by SKIVE inside vLLM.
    # ------------------------------------------------------------------
    @triton.jit
    def skive_token_score_kernel(
        Q_ptr, K_ptr, V_ptr, Mask_ptr, Sink_ptr,
        Score_ptr, L_ptr,
        scaling,
        N_REP,
        SEQ_LEN_KV,  # runtime (not constexpr): avoids a recompile per padded length
        stride_q_b, stride_q_h, stride_q_d,
        stride_k_b, stride_k_h, stride_k_s, stride_k_d,
        stride_v_b, stride_v_h, stride_v_s, stride_v_d,
        stride_mask_b, stride_mask_s,
        stride_sc_b, stride_sc_h, stride_sc_s,
        stride_l_b, stride_l_h,
        D_HEAD: tl.constexpr,
        BLOCK_S: tl.constexpr,
        HAS_SINK: tl.constexpr,
    ):
        """Per (batch, q-head): score[b,h,s] = exp(q.k_s*scale - m) * mean_d|v_s|,
        L[b,h] = sum_s exp(...) (+ exp(sink_h - m) if HAS_SINK). Caller divides."""
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        d_offs = tl.arange(0, D_HEAD)

        q_base = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
        q_vec = tl.load(q_base + d_offs * stride_q_d).to(tl.float32)
        kv_h = pid_h // N_REP
        k_base = K_ptr + pid_b * stride_k_b + kv_h * stride_k_h
        v_base = V_ptr + pid_b * stride_v_b + kv_h * stride_v_h
        m_base = Mask_ptr + pid_b * stride_mask_b

        # ---- pass 1: running max of the masked logits (numerical stability) ----
        m_i = tl.full((1,), float("-inf"), dtype=tl.float32)
        for s0 in range(0, SEQ_LEN_KV, BLOCK_S):
            s_offs = s0 + tl.arange(0, BLOCK_S)
            in_b = s_offs < SEQ_LEN_KV
            k_blk = tl.load(
                k_base + s_offs[:, None] * stride_k_s + d_offs[None, :] * stride_k_d,
                mask=in_b[:, None], other=0.0,
            ).to(tl.float32)
            sc = tl.sum(q_vec[None, :] * k_blk, axis=1) * scaling
            keep = tl.load(m_base + s_offs * stride_mask_s, mask=in_b, other=False)
            sc = tl.where(keep & in_b, sc, float("-inf"))
            m_i = tl.maximum(m_i, tl.max(sc, axis=0))
        if HAS_SINK:
            sink = tl.load(Sink_ptr + pid_h).to(tl.float32)
            m_i = tl.maximum(m_i, sink)

        # ---- pass 2: exp(sc - m), per-token score, denominator ----
        l_i = tl.zeros((1,), dtype=tl.float32)
        if HAS_SINK:
            sink = tl.load(Sink_ptr + pid_h).to(tl.float32)
            l_i += tl.exp(sink - m_i)
        sc_base = Score_ptr + pid_b * stride_sc_b + pid_h * stride_sc_h
        for s0 in range(0, SEQ_LEN_KV, BLOCK_S):
            s_offs = s0 + tl.arange(0, BLOCK_S)
            in_b = s_offs < SEQ_LEN_KV
            k_blk = tl.load(
                k_base + s_offs[:, None] * stride_k_s + d_offs[None, :] * stride_k_d,
                mask=in_b[:, None], other=0.0,
            ).to(tl.float32)
            sc = tl.sum(q_vec[None, :] * k_blk, axis=1) * scaling
            keep = tl.load(m_base + s_offs * stride_mask_s, mask=in_b, other=False)
            valid = keep & in_b
            p = tl.where(valid, tl.exp(sc - m_i), 0.0)
            l_i += tl.sum(p, axis=0)
            v_blk = tl.load(
                v_base + s_offs[:, None] * stride_v_s + d_offs[None, :] * stride_v_d,
                mask=in_b[:, None], other=0.0,
            ).to(tl.float32)
            vmean = tl.sum(tl.abs(v_blk), axis=1) / D_HEAD
            tl.store(sc_base + s_offs * stride_sc_s, p * vmean, mask=in_b)
        tl.store(L_ptr + pid_b * stride_l_b + pid_h * stride_l_h + tl.arange(0, 1), l_i)


def triton_evict_attention_forward(
    query: torch.Tensor,          # [B, N_Q_HEADS, 1, D_HEAD]
    key: torch.Tensor,            # [B, N_KV_HEADS, S_KV, D_HEAD]
    value: torch.Tensor,          # [B, N_KV_HEADS, S_KV, D_HEAD]
    attention_mask: torch.Tensor, # [B, S_KV], dtype=torch.bool (True means keep)
    num_key_value_groups: int,    # This is N_REP
    scaling: float,
    dropout: float = 0.0,
    block_s: int = 64,            # Block size for SEQ_LEN_KV, can be tuned
):
    """The team's original wrapper (dense cache): returns (attn_output, evict_idx)."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    assert query.is_cuda and key.is_cuda and value.is_cuda and attention_mask.is_cuda
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    attention_mask = attention_mask.contiguous()
    if dropout != 0.0:
        print("Warning: Dropout is not implemented in the Triton kernel for evict_attention_forward.")

    BATCH_SIZE, N_Q_HEADS, _, D_HEAD = query.shape
    _, N_KV_HEADS, SEQ_LEN_KV, _ = key.shape
    if N_Q_HEADS // num_key_value_groups != N_KV_HEADS:
        raise ValueError(
            f"N_Q_HEADS ({N_Q_HEADS}) // num_key_value_groups ({num_key_value_groups}) "
            f"must be equal to N_KV_HEADS ({N_KV_HEADS})"
        )

    attn_output = torch.empty_like(query)
    temp_loss = torch.zeros((BATCH_SIZE, N_Q_HEADS, SEQ_LEN_KV), dtype=torch.float32, device=query.device)
    tmp_l = torch.zeros((BATCH_SIZE, N_Q_HEADS, 1), dtype=torch.float32, device=query.device)

    stride_q_b, stride_q_h, _, stride_q_d = query.stride()
    stride_k_b, stride_k_h_kv, stride_k_s, stride_k_d = key.stride()
    stride_v_b, stride_v_h_kv, stride_v_s, stride_v_d = value.stride()
    stride_mask_b, stride_mask_s = attention_mask.stride()
    stride_ao_b, stride_ao_h, _, stride_ao_d = attn_output.stride()
    stride_loss_b, stride_loss_h, stride_loss_s = temp_loss.stride()
    stride_l_b, stride_l_h, stride_l_s = tmp_l.stride()

    grid = (BATCH_SIZE, N_Q_HEADS)
    num_warps = 8 if D_HEAD >= 128 else 4
    evict_fwd_kernel[grid](
        Q_ptr=query, K_ptr=key, V_ptr=value, Mask_ptr=attention_mask,
        AttnOutput_ptr=attn_output, TempLoss_ptr=temp_loss, L_ptr=tmp_l,
        scaling=scaling, N_REP=num_key_value_groups, N_Q_HEADS=N_Q_HEADS,
        stride_q_b=stride_q_b, stride_q_h=stride_q_h, stride_q_d=stride_q_d,
        stride_k_b=stride_k_b, stride_k_h_kv=stride_k_h_kv, stride_k_s=stride_k_s, stride_k_d=stride_k_d,
        stride_v_b=stride_v_b, stride_v_h_kv=stride_v_h_kv, stride_v_s=stride_v_s, stride_v_d=stride_v_d,
        stride_mask_b=stride_mask_b, stride_mask_s=stride_mask_s,
        stride_ao_b=stride_ao_b, stride_ao_h=stride_ao_h, stride_ao_d=stride_ao_d,
        stride_loss_b=stride_loss_b, stride_loss_h=stride_loss_h, stride_loss_s=stride_loss_s,
        stride_l_b=stride_l_b, stride_l_h=stride_l_h, stride_l_s=stride_l_s,
        D_HEAD=D_HEAD, SEQ_LEN_KV=SEQ_LEN_KV, BLOCK_S=block_s, num_warps=num_warps,
    )
    final_loss = (temp_loss / tmp_l).mean(dim=1).to(query.dtype)
    attn_output_final = attn_output.transpose(1, 2).contiguous()
    evict_idx = torch.argmin(final_loss, dim=-1)
    return attn_output_final, evict_idx


def score_tokens_fused(
    q: torch.Tensor,        # [B, Hq, D]
    k: torch.Tensor,        # [B, Hkv, S, D]
    v: torch.Tensor,        # [B, Hkv, S, D]
    mask: torch.Tensor,     # [B, S] bool, True = valid token
    scaling: float,
    sinks: torch.Tensor | None = None,  # [Hq] per-head sink logits (gpt-oss) or None
    block_s: int = 64,
) -> torch.Tensor:
    """Per-(batch, head, token) normalized SKIVE score ``p_t * mean_d|v_{t,d}|``.

    Returns float32 ``[B, Hq, S]``; masked / padded tokens are exactly 0.
    Equivalent to the torch path in ``integration._batch_score_va`` up to the
    constant ``1/D`` (mean vs. L1 over head dim), so block rankings match.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    B, Hq, D = q.shape
    _, Hkv, S, _ = k.shape
    if Hq % Hkv != 0 or (D & (D - 1)) != 0:
        raise ValueError(f"unsupported shapes Hq={Hq} Hkv={Hkv} D={D}")
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    mask = mask.contiguous()
    score = torch.zeros((B, Hq, S), dtype=torch.float32, device=q.device)
    l = torch.zeros((B, Hq), dtype=torch.float32, device=q.device)
    has_sink = sinks is not None
    sink_t = sinks.to(q.device, torch.float32).contiguous() if has_sink else l  # dummy ptr
    grid = (B, Hq)
    skive_token_score_kernel[grid](
        q, k, v, mask, sink_t, score, l,
        float(scaling), Hq // Hkv, S,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        mask.stride(0), mask.stride(1),
        score.stride(0), score.stride(1), score.stride(2),
        l.stride(0), l.stride(1),
        D_HEAD=D, BLOCK_S=block_s, HAS_SINK=has_sink,
        num_warps=8 if D >= 128 else 4,
    )
    return score / l.clamp_min(1e-30)[:, :, None]
