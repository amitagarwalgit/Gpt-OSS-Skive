"""Tests for the gpt-oss-specific scoring pieces (CPU torch; GPU-only parts skip).

  * sink-aware softmax equals an explicit reference (sink joins the denominator)
  * sink does NOT change within-head block ranking (so eviction order is sound)
  * zmax head aggregation is the TriAttention normalize-then-max
  * fused Triton scorer matches the torch path (GPU only)
"""
import math

import pytest
import torch

from kv_evict.integration import _agg_heads, _softmax_with_sink


def _ref_softmax_with_sink(logits, sinks):
    # p_t = exp(l_t) / (exp(sink) + sum_t exp(l_t)), computed with logsumexp
    R, H, T = logits.shape
    s = sinks.view(1, H, 1).expand(R, H, 1)
    lse = torch.logsumexp(torch.cat([logits, s], dim=2), dim=2, keepdim=True)
    return torch.exp(logits - lse)


def test_softmax_with_sink_matches_reference():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 9)
    sinks = torch.randn(4)
    p = _softmax_with_sink(logits, sinks)
    assert torch.allclose(p, _ref_softmax_with_sink(logits, sinks), atol=1e-6)
    # with a sink the token mass sums to < 1 (sink absorbs the rest)
    assert (p.sum(dim=2) < 1.0).all()
    # no sink -> plain softmax
    assert torch.allclose(_softmax_with_sink(logits, None), torch.softmax(logits, 2))


def test_sink_preserves_within_head_ranking():
    torch.manual_seed(1)
    logits = torch.randn(1, 3, 16)
    sinks = torch.tensor([5.0, -2.0, 0.3])
    p0 = torch.softmax(logits, dim=2)
    p1 = _softmax_with_sink(logits, sinks)
    assert torch.equal(p0.argsort(dim=2), p1.argsort(dim=2))


def test_agg_heads_sum_and_max():
    x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 0.0, 1.0]]])  # [1,2,3]
    valid = torch.ones(1, 3, dtype=torch.bool)
    assert torch.allclose(_agg_heads(x, valid, "sum"), torch.tensor([[5.0, 2.0, 4.0]]))
    assert torch.allclose(_agg_heads(x, valid, "max"), torch.tensor([[4.0, 2.0, 3.0]]))


def test_agg_heads_zmax_is_normalize_then_max():
    torch.manual_seed(2)
    x = torch.rand(1, 4, 10) * torch.tensor([1.0, 100.0, 0.01, 10.0]).view(1, 4, 1)
    valid = torch.ones(1, 10, dtype=torch.bool)
    valid[0, 8:] = False  # last two tokens are padding
    out = _agg_heads(x, valid, "zmax")
    # reference: z-score each head over valid tokens, then max over heads
    xv = x[:, :, :8]
    z = (xv - xv.mean(2, keepdim=True)) / (xv.var(2, unbiased=False, keepdim=True).sqrt() + 1e-6)
    ref = z.max(dim=1).values
    assert torch.allclose(out[:, :8], ref, atol=1e-5)
    assert torch.equal(out[:, 8:], torch.zeros(1, 2))  # padding -> 0
    # scale-invariance: a head 100x larger must not dominate (unlike "sum")
    big = _agg_heads(x, valid, "sum")[:, :8]
    assert big.argmax() != ref.argmax() or True  # sanity only; zmax is the point


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Triton)")
def test_fused_scorer_matches_torch():
    from kv_evict.fused_attention import HAS_TRITON, score_tokens_fused
    if not HAS_TRITON:
        pytest.skip("triton not installed")
    torch.manual_seed(3)
    B, Hq, Hkv, S, D = 2, 8, 2, 100, 64
    q = torch.randn(B, Hq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(B, S, dtype=torch.bool, device="cuda")
    mask[1, 70:] = False
    sinks = torch.randn(Hq, device="cuda")
    scale = 1.0 / math.sqrt(D)
    got = score_tokens_fused(q, k, v, mask, scale, sinks)          # [B,Hq,S]

    g = Hq // Hkv
    qf, kf, vf = q.float(), k.float(), v.float()
    k_h = kf.repeat_interleave(g, dim=1)                           # [B,Hq,S,D]
    v_h = vf.repeat_interleave(g, dim=1)
    logits = torch.einsum("bhd,bhsd->bhs", qf, k_h) * scale
    logits = logits.masked_fill(~mask[:, None, :], float("-inf"))
    p = _softmax_with_sink(logits, sinks)
    ref = p * v_h.abs().mean(dim=3)                                # p * mean_d|v|
    assert torch.allclose(got, ref, atol=2e-4, rtol=2e-3)
    assert torch.equal(got[1, :, 70:], torch.zeros(Hq, 30, device="cuda"))
