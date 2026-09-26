"""vLLM-side integration glue for KV-eviction (Stages 4b/4c).

This module is imported *inside* the running vLLM worker process. It must not
import vLLM at module load time (it's called from within vLLM), only torch.

Stage 4b: worker-side per-block scoring from the live KV cache, plus a
read-only debug logger used to validate the slicing/layout on real data
WITHOUT mutating anything.

gpt-oss / hybrid-attention support (multi KV-cache group): models such as
gpt-oss-20b/120b alternate sliding-window (window 128) and full-attention
layers, and vLLM gives each attention type its OWN KV-cache group with its own
block table and manager. SKIVE therefore (1) resolves the true full-attention
group and evicts/reclaims only there, (2) scores only that group's layers,
reading each layer's own cache tensor (``layer.kv_cache``) rather than
assuming a single shared block table, and (3) folds gpt-oss's learned
per-head attention *sinks* into the softmax denominator so scores match the
model's real attention distribution. Sliding-window layers are left to vLLM's
own window handling (they already drop out-of-window blocks).
"""

from __future__ import annotations

import numpy as np
import torch

import os

from .manager import EvictionConfig
from .scoring import DEFAULT_EPS, score_blocks_value_attention
from .selection import choose_lowest as _choose_lowest
from .selection import plan_eviction as _plan_eviction

# SKIVE: the cumulative evict/reclaim counters below are always maintained
# (a free int add). The per-step stdout print is gated behind this flag, read
# once at import time, so a default (clean) run pays zero print/flush I/O on
# the decode path. Set SKIVE_LOG_EVICT=1 to emit the counts for parsing.
_SKIVE_LOG_EVICT = os.environ.get("SKIVE_LOG_EVICT") == "1"

# SKIVE_PROTECT_PROMPT=1: never evict the prompt's blocks (sink = prompt blocks + 1,
# per request). For generation-heavy tasks (math / reasoning traces) the prompt is
# the problem statement itself; with a fixed 2-block sink it was evictable once
# the trace exceeded the budget. Long-prompt tasks keep the fixed sink (the prompt
# is what must be evicted there).
_SKIVE_PROTECT_PROMPT = os.environ.get("SKIVE_PROTECT_PROMPT") == "1"

# Trajectory scoring: SKIVE_QHIST=N scores blocks against the last N captured
# decode queries (captured every SKIVE_QHIST_STRIDE steps) instead of the single
# current query -- the observation-window idea of SnapKV / R-KV / TriAttention.
# A single query is a noisy estimate of what the next few hundred tokens will
# read; averaging the attention over a window keeps blocks the trace keeps
# returning to. 0 = off (previous behaviour).
_SKIVE_QHIST = int(os.environ.get("SKIVE_QHIST", "0"))
_SKIVE_QHIST_STRIDE = max(1, int(os.environ.get("SKIVE_QHIST_STRIDE", "2")))
# Redundancy-aware selection (R-KV style): final = z(importance) - lam * z(redundancy)
# where redundancy(block) = max cosine similarity of its mean key to any other
# retained block. Reasoning traces restate themselves; this evicts duplicates
# before unique intermediate results. 0 = off.
_SKIVE_REDUNDANCY = float(os.environ.get("SKIVE_REDUNDANCY", "0"))


def _kv_blocks_f32(kc, idx_flat, layer=None):
    """(K, V) blocks of a cache tensor as float32 [n, bs, Hkv, D]; decodes fp8
    caches (stored as uint8) and applies the layer's per-tensor scales."""
    kb = kc[idx_flat, 0]
    vb = kc[idx_flat, 1]
    if kb.dtype == torch.uint8:  # fp8 KV cache
        f8 = torch.float8_e4m3fn
        ks = float(getattr(layer, "_k_scale_float", 1.0) or 1.0) if layer is not None else 1.0
        vs = float(getattr(layer, "_v_scale_float", 1.0) or 1.0) if layer is not None else 1.0
        return kb.view(f8).to(torch.float32) * ks, vb.view(f8).to(torch.float32) * vs
    return kb.to(torch.float32), vb.to(torch.float32)


def _score_multi_query(qm, k_all, v_all, tok_valid, g, scale, sinks, metric, agg):
    """Per-token importance [R, T] for a query window.

    qm: [R, Nq, Hq, D] (Nq recent queries per row), k_all/v_all: [R, T, Hkv, D],
    tok_valid: [R, T]. Attention is computed per query (sink-aware softmax) and
    averaged over the window; rows are processed in chunks so the
    [Rc, Nq, Hq, T] logits stay under ~256 MB."""
    R, Nq, Hq, D = qm.shape
    T = k_all.shape[1]
    chunk = max(1, int(256e6 // max(1, Nq * Hq * T * 4)))
    out = []
    for s in range(0, R, chunk):
        e = min(R, s + chunk)
        k_h = k_all[s:e].permute(0, 2, 1, 3).repeat_interleave(g, dim=1)        # [Rc,Hq,T,D]
        logits = torch.einsum("rnhd,rhtd->rnht", qm[s:e], k_h) * scale            # [Rc,Nq,Hq,T]
        valid = tok_valid[s:e]
        logits = logits.masked_fill(~valid[:, None, None, :], float("-inf"))
        p = _softmax_with_sink(logits.reshape(-1, Hq, T), sinks).reshape(e - s, Nq, Hq, T)
        p = p.mean(dim=1)                                                          # [Rc,Hq,T]
        if metric in ("h2o", "snapkv"):
            per_head = p
        else:  # value_attention (SKIVE): attention x ||v||_1
            v_h = v_all[s:e].permute(0, 2, 1, 3).repeat_interleave(g, dim=1)
            per_head = p * torch.linalg.vector_norm(v_h, ord=1, dim=3)
        out.append(_agg_heads(per_head, valid, agg))
    return torch.cat(out, 0)


def _redundancy(k_all, R, max_nb, bs, nb_per_row):
    """[R, max_nb] max cosine similarity of each block's mean key to any other
    real block of the same request (0 for padded blocks)."""
    km = k_all.reshape(R, max_nb, bs, -1).mean(dim=2)                              # [R,nb,Hkv*D]
    km = torch.nn.functional.normalize(km, dim=2)
    sim = torch.einsum("rid,rjd->rij", km, km)                                     # [R,nb,nb]
    nb_t = torch.as_tensor(nb_per_row, device=k_all.device)
    valid = torch.arange(max_nb, device=k_all.device)[None, :] < nb_t[:, None]     # [R,nb]
    mask = valid[:, :, None] & valid[:, None, :]
    mask &= ~torch.eye(max_nb, dtype=torch.bool, device=k_all.device)[None]
    sim = sim.masked_fill(~mask, -1.0)
    red = sim.max(dim=2).values.clamp_min(0.0)
    return red.masked_fill(~valid, 0.0)


def _zscore_rows(x, valid):
    vm = valid.to(x.dtype)
    n = vm.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (x * vm).sum(dim=1, keepdim=True) / n
    var = (((x - mean) * vm) ** 2).sum(dim=1, keepdim=True) / n
    return ((x - mean) / (var.sqrt() + 1e-6)).masked_fill(~valid, 0.0)


def _skive_capture_qhist(model_runner, layers, stride, num_reqs):
    """Append this step's decode queries (all rows, scored layers) to the
    per-runner window used by trajectory scoring. One index_select per layer."""
    from collections import deque
    hist = getattr(model_runner, "_skive_qhist", None)
    if hist is None:
        hist = model_runner._skive_qhist = deque(maxlen=_SKIVE_QHIST)
    q_rows = model_runner.query_start_loc.np[1:num_reqs + 1] - 1
    idx = None
    per_layer = {}
    for li in range(0, len(layers), stride):
        q = getattr(layers[li], "_skive_q", None)
        if q is None:
            continue
        if idx is None:
            idx = torch.as_tensor(q_rows, dtype=torch.long, device=q.device)
        per_layer[li] = q.index_select(0, idx).to(torch.bfloat16)               # [R, Hq, D]
    if per_layer:
        hist.append((tuple(model_runner.input_batch.req_ids[:num_reqs]), per_layer))

# vLLM's reserved placeholder block (block_pool.py: "placeholder block with
# block_id=0"). Pointing a block-table entry here frees the real block while
# reads return zeros (the cache is .zero_()'d at init and block 0 is never
# written). Under full attention this is a bounded zero-value attention sink --
# it does NOT shift positions, so RoPE stays exact for every retained token.
NULL_BLOCK_ID = 0

# Query-dependent block-selection policies, all evaluated in one common harness
# (same budget / sinks / sparse-gather) so only the SCORING SIGNAL differs:
#   value_attention (SKIVE) : sum_t softmax(q.k)*||v||_1      (attention x value)
#   h2o                     : sum_t softmax(q.k)              (summed attention mass)
#   snapkv                  : max_t softmax(q.k)              (max-pooled attention)
#   streaming               : block position                 (keep sinks + recent)
# NOTE: in-framework reimplementations capturing each method's core signal, NOT
# the authors' official code. Per-layer methods (PyramidKV) are infeasible on
# vLLM's shared block table; Quest is compute-sparsity (no memory saving).
_QDEP_METRICS = ("value_attention", "h2o", "snapkv", "streaming")


# --------------------------------------------------------------------------
# Multi KV-cache-group resolution (gpt-oss & other hybrid models).
# --------------------------------------------------------------------------
_NON_FULL_SPEC_HINTS = ("SlidingWindow", "Mamba", "Cross", "EncoderOnly",
                        "ChunkedLocal", "UniformType")


def _is_full_attention_spec(spec) -> bool:
    """Duck-typed test for a full-attention KV spec (no window / chunk), by
    class name + attributes, so it needs no vLLM import (and is unit-testable)."""
    name = type(spec).__name__
    if any(h in name for h in _NON_FULL_SPEC_HINTS):
        return False
    return (getattr(spec, "sliding_window", None) is None
            and getattr(spec, "attention_chunk_size", None) is None)


def _skive_kv_groups(model_runner):
    """``[(gid, spec, layer_names)]`` from ``kv_cache_config``, else from the
    runner's ``attn_groups``; ``None`` if neither can be inspected."""
    try:
        return [(i, g.kv_cache_spec, list(g.layer_names))
                for i, g in enumerate(model_runner.kv_cache_config.kv_cache_groups)]
    except Exception:
        pass
    try:
        by_gid = {}
        for lst in model_runner.attn_groups:
            for g in lst:
                by_gid.setdefault(int(g.kv_cache_group_id),
                                  (g.kv_cache_spec, list(g.layer_names)))
        return [(gid, spec, names) for gid, (spec, names) in sorted(by_gid.items())]
    except Exception:
        return None


def _skive_full_group(model_runner):
    """Resolve ``(gid, layer_names)`` of the TRUE full-attention KV-cache group.

    ``layer_names`` is ``None`` for single-group models (=> use every attention
    layer, the pre-existing behaviour). On a multi-group model whose groups
    cannot be inspected we set ``_skive_multigroup_unresolved`` so eviction is
    REFUSED instead of guessing group 0 (which on gpt-oss could be the
    sliding-window table). Cached on the runner after first use.
    """
    cached = getattr(model_runner, "_skive_full_group_cache", None)
    if cached is not None:
        return cached
    gid, names = 0, None
    groups = _skive_kv_groups(model_runner)
    if groups is not None and len(groups) > 1:
        # Prefer a pure full-attention group (no sliding window at all).
        pick = next((k for k, (_, spec, _) in enumerate(groups)
                     if _is_full_attention_spec(spec)), None)
        if pick is None:  # e.g. non-hybrid allocator: FullAttentionSpec w/ window
            pick = next((k for k, (_, spec, _) in enumerate(groups)
                         if any(h in type(spec).__name__
                                for h in ("FullAttention", "MLAAttention"))), None)
        if pick is not None:
            gid, names = groups[pick][0], set(groups[pick][2])
        else:
            model_runner._skive_multigroup_unresolved = True
    elif groups is None:
        try:
            n_bt = len(model_runner.input_batch.block_table.block_tables)
        except Exception:
            n_bt = 1
        if n_bt > 1:
            model_runner._skive_multigroup_unresolved = True
    model_runner._skive_full_group_cache = (gid, names)
    return gid, names


def _skive_layers(model_runner):
    """Attention layer modules SKIVE may score, in forward order, restricted to
    the full-attention group on hybrid models. Cached on the runner."""
    cached = getattr(model_runner, "_skive_layers_cache", None)
    if cached is not None:
        return cached
    _, names = _skive_full_group(model_runner)
    layers = []
    for name, m in model_runner.compilation_config.static_forward_context.items():
        if type(m).__name__ not in ("Attention", "MLAAttention"):
            continue
        if names is not None and name not in names:
            continue
        layers.append(m)
    model_runner._skive_layers_cache = layers
    return layers


def _layer_kv(layer, model_runner, li):
    """A layer's own KV tensor (bound by vLLM's bind_kv_cache). The positional
    fallback into ``model_runner.kv_caches`` is only valid when scoring every
    layer (single-group models), never on a filtered multi-group layer list."""
    kc = getattr(layer, "kv_cache", None)
    if isinstance(kc, (list, tuple)):
        kc = kc[0] if kc else None
    if kc is None:
        _, names = _skive_full_group(model_runner)
        if names is None and li < len(model_runner.kv_caches):
            kc = model_runner.kv_caches[li]
    return kc


def _skive_full_caches(model_runner):
    """KV tensors of the scorable (full-attention) layers, in forward order."""
    out = []
    for li, layer in enumerate(_skive_layers(model_runner)):
        kc = _layer_kv(layer, model_runner, li)
        if kc is not None and kc.ndim == 5 and kc.shape[1] == 2:
            out.append(kc)
    return out


def _skive_block_table(model_runner):
    """The worker BlockTable of the full-attention group (or None if unsafe)."""
    gid, _ = _skive_full_group(model_runner)
    if getattr(model_runner, "_skive_multigroup_unresolved", False):
        if not getattr(model_runner, "_skive_unresolved_warned", False):
            model_runner._skive_unresolved_warned = True
            print("[SKIVE] multiple KV-cache groups but the full-attention group is "
                  "unresolved; eviction disabled for safety", flush=True)
        return None
    bt = model_runner.input_batch.block_table[gid]
    if getattr(bt, "use_hybrid_blocks", False):
        # Kernel blocks != manager blocks: worker logical index != manager
        # index, so reclaim would free the wrong block. Refuse (log once).
        if not getattr(model_runner, "_skive_hybrid_warned", False):
            model_runner._skive_hybrid_warned = True
            print("[SKIVE] hybrid kernel/manager block sizes detected; eviction "
                  "disabled for safety (set block_size to a kernel-supported size)",
                  flush=True)
        return None
    return bt


def score_request_blocks(
    kv_caches: list[torch.Tensor],
    block_ids,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Per-block importance for one request, aggregated over all layers.

    Each entry of ``kv_caches`` is a FlashAttention KV tensor of shape
    ``(num_blocks, 2, block_size, num_kv_heads, head_size)`` (NHD layout; dim 1
    index 0 = K, 1 = V). For physical blocks ``block_ids`` we compute

        score(b) = ||V_b||_2 / (||K_b||_2 + eps)

    where the norm is the L2 over every element of that block across ALL
    attention layers (i.e. the layers' contributions are concatenated). Lower
    score = evict first. Returns a 1-D float32 tensor of length len(block_ids),
    on the kv-cache device.
    """
    idx = torch.as_tensor(block_ids, dtype=torch.long)
    n = idx.numel()
    if n == 0:
        return torch.empty(0, dtype=torch.float32)

    k_terms: list[torch.Tensor] = []
    v_terms: list[torch.Tensor] = []
    for kc in kv_caches:
        # Only full-attention KV tensors (skip e.g. mamba state tensors).
        if kc is None or kc.ndim != 5 or kc.shape[1] != 2:
            continue
        i = idx.to(kc.device)
        kb, vb = _kv_blocks_f32(kc, i)
        k_blocks = kb.reshape(n, -1)
        v_blocks = vb.reshape(n, -1)
        k_terms.append((k_blocks * k_blocks).sum(dim=1))
        v_terms.append((v_blocks * v_blocks).sum(dim=1))

    if not k_terms:  # no attention KV caches found
        return torch.empty(0, dtype=torch.float32)
    sum_k = torch.stack(k_terms).sum(dim=0)
    sum_v = torch.stack(v_terms).sum(dim=0)
    return torch.sqrt(sum_v) / (torch.sqrt(sum_k) + eps)


def _row_block_ids(model_runner, req_index: int):
    """Physical block ids currently held by request row `req_index` (full group)."""
    bt = _skive_block_table(model_runner)
    if bt is None:
        return []
    n = int(bt.num_blocks_per_row[req_index])
    if n <= 0:
        return []
    return bt.block_table.np[req_index, :n].tolist()


def debug_log_scores(model_runner) -> bool:
    """Stage 4b validation: score the first request that has >= 2 blocks and
    log the result. READ-ONLY -- mutates nothing. Returns True once it has
    logged (so the caller can stop invoking it).
    """
    ib = model_runner.input_batch
    num_reqs = ib.num_reqs if hasattr(ib, "num_reqs") else len(ib.req_ids)
    for ri in range(num_reqs):
        block_ids = _row_block_ids(model_runner, ri)
        if len(block_ids) < 2:
            continue
        scores = score_request_blocks(_skive_full_caches(model_runner), block_ids)
        vals = [round(x, 4) for x in scores.tolist()]
        finite = bool(torch.isfinite(scores).all().item())
        msg = (
            f"[SKIVE 4b] req_index={ri} blocks={len(block_ids)} "
            f"finite={finite} scores={vals}"
        )
        print(msg, flush=True)
        try:
            from vllm.logger import init_logger

            init_logger(__name__).info(msg)
        except Exception:
            pass
        return True
    return False


# --------------------------------------------------------------------------
# Stage 4c: actual eviction via null-block replacement (worker-side).
# --------------------------------------------------------------------------
def build_eviction_config(cache_config) -> EvictionConfig | None:
    """Construct an EvictionConfig from the vLLM cache_config, or None if
    eviction is disabled / misconfigured (in which case we never evict)."""
    if not getattr(cache_config, "kv_evict_enabled", False):
        return None
    budget = getattr(cache_config, "kv_evict_budget", None)
    if budget is None:
        return None
    try:
        return EvictionConfig(
            block_size=cache_config.block_size,
            kv_budget=int(budget),
            num_sink_blocks=int(getattr(cache_config, "kv_evict_num_sink_blocks", 0)),
            num_local_blocks=int(getattr(cache_config, "kv_evict_num_local_blocks", 0)),
            metric=os.environ.get("SKIVE_METRIC", "vk_ratio"),
        )
    except ValueError:
        return None


def _layer_sinks(layer):
    """gpt-oss learned per-head sink logits (``s_aux``) for this layer, or None."""
    s = getattr(getattr(layer, "impl", None), "sinks", None)
    return s if isinstance(s, torch.Tensor) else None


def _softmax_with_sink(logits: torch.Tensor, sinks: torch.Tensor | None):
    """softmax over tokens; if ``sinks`` [Hq] is given, the sink logit joins
    the denominator exactly like gpt-oss / FA3 ``s_aux`` (then is dropped).
    logits: [R, Hq, T] (masked entries are -inf)."""
    if sinks is None:
        return torch.softmax(logits, dim=2)
    R, Hq, _ = logits.shape
    s = sinks.to(logits.device, torch.float32).view(1, Hq, 1).expand(R, Hq, 1)
    return torch.softmax(torch.cat([logits, s], dim=2), dim=2)[:, :, :-1]


def _agg_heads(x: torch.Tensor, valid: torch.Tensor, mode: str) -> torch.Tensor:
    """Aggregate per-head token scores [R,H,T] -> [R,T].

    sum  : plain sum over heads (default, previous behaviour).
    max  : max over heads.
    zmax : TriAttention (arXiv:2604.04921 §4.3) normalize-then-aggregate for
           GQA -- z-score each head over its valid tokens, then max over heads,
           so a token is kept if ANY query head deems it important. Recommended
           for gpt-oss (64 q-heads / 8 kv-heads)."""
    if mode == "max":
        return x.max(dim=1).values
    if mode == "zmax":
        vm = valid[:, None, :].to(x.dtype)
        n = vm.sum(dim=2, keepdim=True).clamp_min(1.0)
        mean = (x * vm).sum(dim=2, keepdim=True) / n
        var = (((x - mean) * vm) ** 2).sum(dim=2, keepdim=True) / n
        z = (x - mean) / (var.sqrt() + 1e-6)
        z = z.masked_fill(~valid[:, None, :], float("-inf"))
        return z.max(dim=1).values.masked_fill(~valid, 0.0)
    return x.sum(dim=1)


def score_request_blocks_va(model_runner, req_index, real_block_ids,
                            eps: float = DEFAULT_EPS):
    """SKIVE-style value x attention score for a request's real blocks,
    aggregated over layers. Returns a 1-D tensor (len == #real blocks) or None
    if the per-layer query wasn't captured (caller falls back to vk_ratio)."""
    try:
        import torch
        bt = _skive_block_table(model_runner)
        if bt is None:
            return None
        bs = int(bt.block_size)
        q_row = int(model_runner.query_start_loc.np[req_index + 1]) - 1
        layers = _skive_layers(model_runner)
        # Layer-stride scoring (Zipage): aggregate the value_attention score over
        # every stride-th layer instead of all layers -> ~stride x cheaper. The
        # relative block ranking is preserved, so eviction quality is ~unchanged.
        stride = max(1, int(os.environ.get("SKIVE_SCORE_LAYER_STRIDE", "1")))
        total = None
        for li in range(0, len(layers), stride):
            layer = layers[li]
            q = getattr(layer, "_skive_q", None)
            if q is None:
                return None
            kc = _layer_kv(layer, model_runner, li)
            if kc is None or kc.ndim != 5 or kc.shape[1] != 2:
                continue
            dev = kc.device
            idx = torch.as_tensor(real_block_ids, dtype=torch.long, device=dev)
            qr = q[q_row].to(dev)                       # [num_heads, head_size]
            kb = kc[idx, 0]                              # [nb, bs, Hkv, D]
            vb = kc[idx, 1]
            nb = kb.shape[0]
            k_all = kb.reshape(nb * bs, kb.shape[2], kb.shape[3])
            v_all = vb.reshape(nb * bs, vb.shape[2], vb.shape[3])
            nqpk = getattr(layer.impl, "num_queries_per_kv",
                           qr.shape[0] // kb.shape[2])
            scale = getattr(layer.impl, "scale", None)
            s = score_blocks_value_attention(qr, k_all, v_all, bs, nqpk, scale, eps)
            total = s if total is None else total + s
        return total
    except Exception:
        return None


def _ensure_null_zeroed(model_runner) -> None:
    """One-time defensive zero of the null block's KV across all layers."""
    if getattr(model_runner, "_skive_null_zeroed", False):
        return
    for kc in model_runner.kv_caches:
        if kc is not None and kc.ndim == 5 and kc.shape[1] == 2:
            kc[NULL_BLOCK_ID].zero_()
    model_runner._skive_null_zeroed = True


def _vk_score_cached(model_runner, req_id, real_pos, real_block_ids,
                     eps: float = DEFAULT_EPS):
    """vk_ratio scores for a request's real blocks, cached per (req, logical
    position).

    A block's K/V are frozen once the block is full, so its ``||V||/||K||``
    score never changes. We compute each block's score once and reuse it,
    turning the per-step scoring cost from O(all blocks) into O(newly-filled
    blocks) -- the dominant eviction-hot-path overhead. The cache entry stores
    the physical block id so a reused slot is re-scored rather than trusted.
    """
    cache = getattr(model_runner, "_skive_score_cache", None)
    if cache is None:
        cache = model_runner._skive_score_cache = {}
    rc = cache.setdefault(req_id, {})
    scores = [0.0] * len(real_block_ids)
    todo = []  # (k, logical_j, block_id) for uncached blocks
    for k, (j, bid) in enumerate(zip(real_pos, real_block_ids)):
        ent = rc.get(j)
        if ent is not None and ent[0] == bid:
            scores[k] = ent[1]
        else:
            todo.append((k, j, bid))
    if todo:
        new = score_request_blocks(
            _skive_full_caches(model_runner), [bid for _, _, bid in todo], eps
        ).tolist()
        for (k, j, bid), s in zip(todo, new):
            rc[j] = (bid, s)
            scores[k] = s
    return scores


def _batch_prescore(model_runner, over_rows, ib, npbt, nrows,
                    eps: float = DEFAULT_EPS) -> None:
    """Populate the vk_ratio score cache for every cache-miss block across ALL
    over-budget rows in a SINGLE batched GPU call.

    This is the key eviction-hot-path optimization: scoring per request meant
    one GPU->CPU ``.tolist()`` sync per over-budget row every step (~6/step),
    and each sync stalls the async decode pipeline. Batching collapses that to
    one sync per step regardless of how many rows evict.
    """
    cache = getattr(model_runner, "_skive_score_cache", None)
    if cache is None:
        cache = model_runner._skive_score_cache = {}
    misses = []       # (req_id, logical_j, block_id) needing a score
    miss_bids = []
    for ri in over_rows.tolist():
        req_id = ib.req_ids[ri]
        if req_id is None:
            continue
        rc = cache.setdefault(req_id, {})
        n = int(nrows[ri])
        row = npbt[ri]
        for j in range(n):
            bid = int(row[j])
            if bid == NULL_BLOCK_ID:
                continue
            ent = rc.get(j)
            if ent is None or ent[0] != bid:
                misses.append((req_id, j, bid))
                miss_bids.append(bid)
    if miss_bids:
        s = score_request_blocks(_skive_full_caches(model_runner), miss_bids, eps).tolist()
        for (req_id, j, bid), sc in zip(misses, s):
            cache[req_id][j] = (bid, sc)


def _batch_score_va(model_runner, over_rows, ib, npbt, nrows,
                    eps: float = DEFAULT_EPS):
    """Batched query-dependent block scores for ALL over-budget rows at once.

    Replaces the per-request scoring loop (one q@K matmul chain + one GPU->CPU
    sync per request) with a few fused ops per sampled layer and a SINGLE sync
    per step. Rows are padded to the max block count; padded tokens are masked
    to -inf before the softmax, so results match the per-request path exactly.
    Honors SKIVE_SCORE_LAYER_STRIDE, SKIVE_HEAD_AGG (sum|max|zmax) and the
    gpt-oss attention sinks. SKIVE_SCORER=triton routes value_attention through
    the fused Triton kernel in ``fused_attention``. Returns
    {req_index: [block scores]} or None (caller falls back)."""
    try:
        import torch
        bt = _skive_block_table(model_runner)
        if bt is None:
            return None
        bs = int(bt.block_size)
        stride = max(1, int(os.environ.get("SKIVE_SCORE_LAYER_STRIDE", "1")))
        agg = os.environ.get("SKIVE_HEAD_AGG", "sum")
        use_fused = os.environ.get("SKIVE_SCORER", "torch") == "triton"
        layers = _skive_layers(model_runner)
        rows = over_rows.tolist()
        bids_per_row = []
        for ri in rows:
            n = int(nrows[ri]); row = npbt[ri]
            bids_per_row.append([int(row[j]) for j in range(n)
                                 if int(row[j]) != NULL_BLOCK_ID])
        R = len(rows)
        nb_per_row = [len(b) for b in bids_per_row]
        max_nb = max(nb_per_row)

        metric = os.environ.get("SKIVE_METRIC", "value_attention")
        # StreamingLLM: purely positional (keep sinks + most-recent window). No
        # attention/query needed -- score = block position, oldest evicted first.
        if metric == "streaming":
            return {ri: [float(j) for j in range(nb_per_row[r])]
                    for r, ri in enumerate(rows)}

        idx = torch.zeros((R, max_nb), dtype=torch.long)
        for r, bids in enumerate(bids_per_row):
            idx[r, :len(bids)] = torch.as_tensor(bids, dtype=torch.long)
        q_rows = [int(model_runner.query_start_loc.np[ri + 1]) - 1 for ri in rows]

        total = None
        red_first = None
        qhist = getattr(model_runner, "_skive_qhist", None) if _SKIVE_QHIST > 0 else None
        req_ids = [ib.req_ids[ri] for ri in rows]
        for li in range(0, len(layers), stride):
            layer = layers[li]
            q = getattr(layer, "_skive_q", None)
            if q is None:
                return None
            kc = _layer_kv(layer, model_runner, li)
            if kc is None or kc.ndim != 5 or kc.shape[1] != 2:
                continue
            dev = kc.device
            idx_d = idx.to(dev)
            qr = q[q_rows].to(dev).to(torch.float32)            # [R, Hq, D]
            kb, vb = _kv_blocks_f32(kc, idx_d.reshape(-1), layer)   # [R*max_nb, bs, Hkv, D]
            Hkv, D = kb.shape[2], kb.shape[3]
            T = max_nb * bs
            k_all = kb.reshape(R, T, Hkv, D)
            v_all = vb.reshape(R, T, Hkv, D)
            Hq = qr.shape[1]
            g = getattr(layer.impl, "num_queries_per_kv", Hq // Hkv)
            scale = getattr(layer.impl, "scale", None) or (1.0 / D ** 0.5)
            sinks = _layer_sinks(layer)
            tok_valid = (torch.arange(T, device=dev)[None, :] <
                         torch.as_tensor([n * bs for n in nb_per_row],
                                         device=dev)[:, None])           # [R,T]

            # Trajectory window: the current query plus the captured history of
            # each over-budget request (rows missing from an older step reuse
            # the current query, so the window is always well defined).
            window = [qr]
            if qhist:
                for rids, per_layer in qhist:
                    ht = per_layer.get(li)
                    if ht is None:
                        continue
                    pos = {rid: i for i, rid in enumerate(rids)}
                    sel = [pos.get(rid, -1) for rid in req_ids]
                    if all(s_ < 0 for s_ in sel):
                        continue
                    sel_t = torch.as_tensor([max(s_, 0) for s_ in sel], device=dev)
                    miss = torch.as_tensor([s_ < 0 for s_ in sel], device=dev)
                    h = ht.to(dev).index_select(0, sel_t).to(torch.float32)
                    window.append(torch.where(miss[:, None, None], qr, h))

            per_head = None
            token = None
            if len(window) > 1:
                qm = torch.stack(window, dim=1)                                # [R,Nq,Hq,D]
                token = _score_multi_query(qm, k_all, v_all, tok_valid, g, scale, sinks,
                                           metric, agg)                        # [R,T]
            elif use_fused and metric == "value_attention":
                try:
                    from .fused_attention import score_tokens_fused
                    per_head = score_tokens_fused(
                        qr, k_all.permute(0, 2, 1, 3), v_all.permute(0, 2, 1, 3),
                        tok_valid, scale, sinks)                       # [R,Hq,T]
                except Exception:
                    per_head = None  # fall back to the torch path below
            if token is None and per_head is None:
                k_h = k_all.permute(0, 2, 1, 3).repeat_interleave(g, dim=1)  # [R,Hq,T,D]
                logits = torch.einsum("rhd,rhtd->rht", qr, k_h) * scale      # [R,Hq,T]
                logits = logits.masked_fill(~tok_valid[:, None, :], float("-inf"))
                p = _softmax_with_sink(logits, sinks)                          # [R,Hq,T]
                if metric in ("h2o", "snapkv"):
                    per_head = p
                else:  # value_attention (SKIVE): attention x ||v||_1
                    v_h = v_all.permute(0, 2, 1, 3).repeat_interleave(g, dim=1)
                    per_head = p * torch.linalg.vector_norm(v_h, ord=1, dim=3)
            if token is None:
                token = _agg_heads(per_head, tok_valid, agg)                    # [R,T]
            if metric == "snapkv":   # max-pooled attention over block tokens
                s = token.view(R, max_nb, bs).max(dim=2).values
            else:                    # summed over block tokens
                s = token.view(R, max_nb, bs).sum(dim=2)                        # [R,max_nb]
            total = s if total is None else total + s
            if _SKIVE_REDUNDANCY > 0 and red_first is None:
                red_first = _redundancy(k_all, R, max_nb, bs, nb_per_row)
        if total is None:
            return None
        if red_first is not None:
            blk_valid = (torch.arange(max_nb, device=total.device)[None, :] <
                         torch.as_tensor(nb_per_row, device=total.device)[:, None])
            total = _zscore_rows(total, blk_valid) - _SKIVE_REDUNDANCY * _zscore_rows(red_first, blk_valid)
        flat = total.tolist()  # the ONE GPU->CPU sync for the whole step
        return {ri: flat[r][:nb_per_row[r]] for r, ri in enumerate(rows)}
    except Exception:
        if os.environ.get("SKIVE_DBG2") == "1":
            import traceback
            traceback.print_exc()
        return None


def evict_request_blocks(
    model_runner, cfg: EvictionConfig, req_index: int
) -> list[int]:
    """Null-replace the lowest-importance non-protected real blocks of one
    request until it is back within budget. Returns the logical row indices
    that were nulled (so the scheduler can free the same physical blocks).

    Positions/length are left UNCHANGED (RoPE stays exact); only the block
    table entry is repointed to the null block.
    """
    bt = _skive_block_table(model_runner)  # full-attention group
    if bt is None:
        return []
    n = int(bt.num_blocks_per_row[req_index])
    if n == 0:
        return []
    row = bt.block_table.np[req_index]

    sink = cfg.num_sink_blocks
    if _SKIVE_PROTECT_PROMPT:
        try:  # V1 InputBatch keeps the prompt length per row
            npt = int(model_runner.input_batch.num_prompt_tokens[req_index])
            sink = max(sink, -(-npt // int(bt.block_size)) + 1)
        except Exception:  # noqa: BLE001
            pass
    # Real (non-null) blocks + evictable candidates, vectorized in NumPy
    # (replaces the O(n) Python list-comp + protected-set scan -- IMP-10).
    real_indices, candidates_k, num_to_evict = _plan_eviction(
        row[:n], sink, cfg.num_local_blocks, cfg.kv_budget
    )
    if num_to_evict == 0:
        return []
    num_real = int(real_indices.size)
    real_pos = real_indices.tolist()

    # Score the real blocks. value_attention is query-dependent (recomputed);
    # vk_ratio is cached per block (frozen K/V) to avoid re-scoring every step.
    real_block_ids = row[real_indices].tolist()
    req_id = model_runner.input_batch.req_ids[req_index]
    scores = None
    if os.environ.get("SKIVE_NOSCORE") == "1":  # diagnostic: skip GPU scoring
        scores = list(range(num_real))
    elif getattr(cfg, "metric", "vk_ratio") in _QDEP_METRICS:
        # Prefer the whole-batch pre-scored result (one sync/step); fall back
        # to the per-request path if it's missing or desynced.
        vb = getattr(model_runner, "_skive_va_batch", None)
        if vb is not None and req_index in vb and len(vb[req_index]) == num_real:
            scores = vb[req_index]
        else:
            sv = score_request_blocks_va(model_runner, req_index, real_block_ids)
            if sv is not None:
                scores = sv.tolist()
    if scores is None:  # default proxy (cached), or fallback if query missing
        scores = _vk_score_cached(model_runner, req_id, real_pos, real_block_ids)

    # Lowest-importance candidates first (stable on ties via index) -- replaces
    # the Python sorted(key=lambda) (IMP-10).
    chosen = _choose_lowest(candidates_k, scores, num_to_evict)
    rc = getattr(model_runner, "_skive_score_cache", {}).get(req_id)
    nulled_rows = []
    for k in chosen.tolist():
        j = real_pos[k]
        row[j] = NULL_BLOCK_ID  # repoint worker table to null
        nulled_rows.append(j)
        if rc is not None:
            rc.pop(j, None)  # evicted block's cached score no longer needed
    return nulled_rows


def skive_post_step(model_runner) -> None:
    """Stable post-decode entrypoint called from the patched gpu_model_runner.

    Nulls evicted entries in the worker block table and records
    (req_id, j, group_id) triples on ``model_runner._skive_pending_free`` for
    EngineCore to free on the scheduler side (4c-ii). Flag-gated upstream; a
    None config => no-op."""
    cfg = build_eviction_config(model_runner.cache_config)
    if cfg is None:
        return
    _ensure_null_zeroed(model_runner)
    _dbg = os.environ.get("SKIVE_TIME") == "1"
    if _dbg:
        import time as _t
        if os.environ.get("SKIVE_TIME_SYNC") == "1":
            torch.cuda.synchronize()  # drain GPU so timer measures pure CPU work
        _t0 = _t.perf_counter()

    ib = model_runner.input_batch
    num_reqs = ib.num_reqs if hasattr(ib, "num_reqs") else len(ib.req_ids)
    bt = _skive_block_table(model_runner)
    if bt is None:
        return
    gid, _ = _skive_full_group(model_runner)

    # Trajectory scoring: keep a window of recent decode queries for every row.
    if _SKIVE_QHIST > 0 and getattr(cfg, "metric", "vk_ratio") in _QDEP_METRICS and num_reqs:
        qstep = getattr(model_runner, "_skive_qstep", 0) + 1
        model_runner._skive_qstep = qstep
        if qstep % _SKIVE_QHIST_STRIDE == 0:
            try:
                _skive_capture_qhist(
                    model_runner, _skive_layers(model_runner),
                    max(1, int(os.environ.get("SKIVE_SCORE_LAYER_STRIDE", "1"))), num_reqs)
            except Exception as e:  # noqa: BLE001
                if os.environ.get("SKIVE_DBG2") == "1":
                    print(f"[SKIVE] qhist capture failed: {e!r}", flush=True)

    # Vectorized over-budget check for the whole batch in one NumPy op. This
    # replaces the per-request Python scan that ran every step: eviction only
    # actually fires when a block fills (~every block_size steps), so on most
    # steps this finds nothing over budget and we skip all per-request work.
    npbt = bt.block_table.np[:num_reqs]
    nrows = np.asarray(bt.num_blocks_per_row[:num_reqs])
    cols = np.arange(npbt.shape[1])[None, :]
    real_count = ((cols < nrows[:, None]) & (npbt != NULL_BLOCK_ID)).sum(1)
    over_rows = np.nonzero(real_count > cfg.kv_budget)[0]

    # Deferred eviction: block scores are static, so we don't need to pay the
    # GPU->CPU scoring sync every step. Only score+evict every SKIVE_EVICT_EVERY
    # steps, unless a row is >= MARGIN over budget (safety valve against pool
    # exhaustion). This cuts the per-step sync-stall frequency ~N-fold; the cost
    # is a few extra retained blocks briefly (peak ~budget + MARGIN).
    step = getattr(model_runner, "_skive_step", 0) + 1
    model_runner._skive_step = step
    if over_rows.size:
        excess = int((real_count[over_rows] - cfg.kv_budget).max())
        _every = int(os.environ.get("SKIVE_EVICT_EVERY", "8"))
        _margin = int(os.environ.get("SKIVE_EVICT_MARGIN", "4"))
        if step % _every != 0 and excess < _margin:
            over_rows = over_rows[:0]  # defer to a later step

    if _dbg:
        import time as _t
        _tc = _t.perf_counter()

    if os.environ.get("SKIVE_DBG2") == "1":
        _s = getattr(model_runner, "_skive_dbgstep", 0) + 1
        model_runner._skive_dbgstep = _s
        if _s <= 12 or _s % 50 == 0:
            print(f"[SKIVE DBG] step={_s} nreq={num_reqs} gid={gid} "
                  f"n_max={int(nrows[:num_reqs].max())} "
                  f"real_max={int(real_count.max())} real_min={int(real_count.min())} "
                  f"over={len(over_rows)} budget={cfg.kv_budget}", flush=True)

    # Batch-score every over-budget row in one GPU pass (one sync/step) before
    # the per-row eviction loop, which then runs entirely on CPU-held scores --
    # no per-row GPU sync. vk_ratio: cache-miss blocks only (scores static).
    # query-dependent metrics: fused whole-batch q@K scoring.
    model_runner._skive_va_batch = None
    if over_rows.size:
        if getattr(cfg, "metric", "vk_ratio") in _QDEP_METRICS:
            model_runner._skive_va_batch = _batch_score_va(
                model_runner, over_rows, ib, npbt, nrows)
        else:
            _batch_prescore(model_runner, over_rows, ib, npbt, nrows)

    pending = getattr(model_runner, "_skive_pending_free", [])
    # Per-request cumulative blocks evicted, keyed by req_id. Maintained
    # unconditionally (a free dict add, like _skive_evicted_total below) so the
    # eval harness can pull a MEASURED per-request eviction count after generate
    # (via _skive_get_evicted_by_req), rather than estimating it from the budget.
    by_req = getattr(model_runner, "_skive_evicted_by_req", None)
    if by_req is None:
        by_req = model_runner._skive_evicted_by_req = {}
    total = 0
    for ri in over_rows.tolist():
        req_id = ib.req_ids[ri]
        if req_id is None:
            continue
        n_ri = 0
        for j in evict_request_blocks(model_runner, cfg, ri):
            pending.append((req_id, j, gid))
            total += 1
            n_ri += 1
        if n_ri:
            by_req[req_id] = by_req.get(req_id, 0) + n_ri
    model_runner._skive_pending_free = pending

    # Drop score-cache entries for requests no longer in the batch (bounded).
    cache = getattr(model_runner, "_skive_score_cache", None)
    if cache:
        live = {rid for rid in ib.req_ids if rid is not None}
        for rid in [r for r in cache if r not in live]:
            del cache[rid]

    if _dbg:
        import time as _t
        _tend = _t.perf_counter()
        acc = getattr(model_runner, "_skive_t", 0.0) + (_tend - _t0)
        acc_chk = getattr(model_runner, "_skive_tchk", 0.0) + (_tc - _t0)
        acc_loop = getattr(model_runner, "_skive_tloop", 0.0) + (_tend - _tc)
        cnt = getattr(model_runner, "_skive_n", 0) + 1
        model_runner._skive_t, model_runner._skive_n = acc, cnt
        model_runner._skive_tchk, model_runner._skive_tloop = acc_chk, acc_loop
        fired = getattr(model_runner, "_skive_fired", 0) + (1 if total else 0)
        model_runner._skive_fired = fired
        if cnt % 200 == 0:
            print(f"[SKIVE TIME] post_step={1000 * acc / cnt:.3f} ms/step "
                  f"(check={1000 * acc_chk / cnt:.3f} loop={1000 * acc_loop / cnt:.3f}) "
                  f"fired {fired}/{cnt}", flush=True)
    if total:
        c = getattr(model_runner, "_skive_evicted_total", 0) + total
        model_runner._skive_evicted_total = c
        if _SKIVE_LOG_EVICT:
            print(f"[SKIVE 4c] evicted {total} block(s) this step; cumulative={c}",
                  flush=True)


def _skive_pop_pending(worker):
    """Worker-side: return and clear the pending (req_id, logical_index, gid) frees.

    Used via collective_rpc from EngineCore. ``worker`` is the (wrapped) worker;
    attribute access delegates to the real worker, so .model_runner resolves."""
    mr = getattr(worker, "model_runner", None)
    if mr is None:
        return []
    pending = getattr(mr, "_skive_pending_free", None)
    if not pending:
        return []
    mr._skive_pending_free = []
    if os.environ.get("SKIVE_NO_RECLAIM") == "1":  # diagnostic: skip physical free
        return []
    return list(pending)


def _skive_get_evicted_by_req(worker):
    """Worker-side: snapshot of per-request cumulative blocks evicted.

    Returns ``{req_id: blocks_evicted}``. Pulled via collective_rpc from the eval
    harness after ``generate()`` to record a MEASURED per-request eviction count
    (not a budget-derived estimate). Read-only -- it does NOT clear the counter,
    so it is safe to call at any time. Mirrors ``_skive_pop_pending``'s worker
    delegation (attribute access resolves to the real worker.model_runner)."""
    mr = getattr(worker, "model_runner", None)
    if mr is None:
        return {}
    return dict(getattr(mr, "_skive_evicted_by_req", {}) or {})


def skive_reclaim(kv_cache_manager, pending) -> int:
    """Scheduler-side: free the physical blocks the worker evicted, mirroring
    them to null_block in req_to_blocks of the SAME KV-cache group the worker
    evicted from. Returns #blocks freed.

    This mirrors vLLM's own sliding-window path (remove_skipped_blocks): null
    substitution + BlockPool.free_blocks. Prefix caching is OFF in this stage,
    so there are no shared/cached-block ref-count complications. Accepts both
    legacy ``(req_id, j)`` pairs (group 0) and ``(req_id, j, gid)`` triples.
    """
    block_pool = kv_cache_manager.block_pool
    null_block = block_pool.null_block
    managers = kv_cache_manager.coordinator.single_type_managers
    if not managers:
        return 0
    freed = 0
    for item in pending:
        if len(item) == 3:
            req_id, j, gid = item
        else:
            req_id, j = item
            gid = 0
        if gid >= len(managers):
            continue
        mgr = managers[gid]
        blocks = mgr.req_to_blocks.get(req_id)
        if not blocks or j >= len(blocks):
            continue
        old = blocks[j]
        if getattr(old, "is_null", False):
            continue
        blocks[j] = null_block
        block_pool.free_blocks([old])
        freed += 1
    if freed:
        c = getattr(kv_cache_manager, "_skive_freed_total", 0) + freed
        kv_cache_manager._skive_freed_total = c
        if _SKIVE_LOG_EVICT:
            print(f"[SKIVE 4c-ii] freed {freed} physical block(s); cumulative={c}; "
                  f"pool_free={block_pool.get_num_free_blocks()}", flush=True)
    return freed
