"""Trajectory (multi-query) scoring, redundancy penalty and fp8 block decode --
pure torch, no GPU / vLLM."""
import torch

import kv_evict.integration as integ


def _rand_kv(R, T, Hkv, D, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(R, T, Hkv, D, generator=g), torch.randn(R, T, Hkv, D, generator=g)


def test_multi_query_equals_single_query_when_window_is_one():
    R, T, Hkv, Hq, D, bs = 2, 64, 2, 4, 8, 16
    k, v = _rand_kv(R, T, Hkv, D)
    q = torch.randn(R, Hq, D)
    valid = torch.ones(R, T, dtype=torch.bool); valid[1, 48:] = False
    multi = integ._score_multi_query(q[:, None], k, v, valid, Hq // Hkv, 0.3, None,
                                     "value_attention", "sum")
    # reference: the single-query path
    k_h = k.permute(0, 2, 1, 3).repeat_interleave(Hq // Hkv, dim=1)
    logits = torch.einsum("rhd,rhtd->rht", q, k_h) * 0.3
    logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
    p = torch.softmax(logits, dim=2)
    v_h = v.permute(0, 2, 1, 3).repeat_interleave(Hq // Hkv, dim=1)
    ref = (p * torch.linalg.vector_norm(v_h, ord=1, dim=3)).sum(dim=1)
    assert torch.allclose(multi, ref, atol=1e-5)
    assert torch.all(multi[1, 48:] == 0)


def test_multi_query_is_mean_over_window_and_chunks_rows():
    R, T, Hkv, Hq, D = 5, 32, 1, 2, 4
    k, v = _rand_kv(R, T, Hkv, D, seed=1)
    qm = torch.randn(R, 3, Hq, D)
    valid = torch.ones(R, T, dtype=torch.bool)
    out = integ._score_multi_query(qm, k, v, valid, 2, 1.0, None, "h2o", "sum")
    per_q = [integ._score_multi_query(qm[:, i:i + 1], k, v, valid, 2, 1.0, None, "h2o", "sum")
             for i in range(3)]
    assert torch.allclose(out, sum(per_q) / 3, atol=1e-5)
    assert out.shape == (R, T)


def test_sink_aware_multi_query_matches_softmax_with_sink():
    R, T, Hkv, Hq, D = 1, 16, 1, 2, 4
    k, v = _rand_kv(R, T, Hkv, D, seed=2)
    q = torch.randn(R, Hq, D)
    sinks = torch.tensor([0.5, -1.0])
    valid = torch.ones(R, T, dtype=torch.bool)
    out = integ._score_multi_query(q[:, None], k, v, valid, 2, 1.0, sinks, "h2o", "sum")
    logits = torch.einsum("rhd,rhtd->rht", q, k.permute(0, 2, 1, 3).repeat_interleave(2, dim=1))
    ref = integ._softmax_with_sink(logits, sinks).sum(dim=1)
    assert torch.allclose(out, ref, atol=1e-5)


def test_redundancy_flags_duplicate_blocks():
    R, max_nb, bs, Hkv, D = 1, 4, 4, 1, 3
    base = torch.randn(1, bs, Hkv, D)
    blocks = [base, torch.randn(1, bs, Hkv, D), base.clone(), -base]  # 0 and 2 identical
    k_all = torch.cat(blocks, dim=1)                                    # [1, 16, 1, 3]
    red = integ._redundancy(k_all, R, max_nb, bs, [3])                  # block 3 is padding
    assert red.shape == (1, 4)
    assert red[0, 0] > 0.99 and red[0, 2] > 0.99      # duplicates of each other
    assert red[0, 1] < red[0, 0]
    assert red[0, 3] == 0                              # padded block


def test_zscore_rows_ignores_padding():
    x = torch.tensor([[1.0, 2.0, 3.0, 99.0]])
    valid = torch.tensor([[True, True, True, False]])
    z = integ._zscore_rows(x, valid)
    assert abs(float(z[0, :3].mean())) < 1e-6 and z[0, 3] == 0


def test_fp8_blocks_decode_and_scale():
    kc = torch.zeros(4, 2, 16, 1, 8, dtype=torch.uint8)
    f8 = torch.tensor([[1.5]], dtype=torch.float8_e4m3fn)
    kc[2, 0] = f8.view(torch.uint8).item()
    kc[2, 1] = torch.tensor([[-2.0]], dtype=torch.float8_e4m3fn).view(torch.uint8).item()

    class L:
        _k_scale_float = 2.0
        _v_scale_float = 0.5
    kb, vb = integ._kv_blocks_f32(kc, torch.tensor([2]), L())
    assert kb.dtype == torch.float32 and float(kb[0, 0, 0, 0]) == 3.0
    assert float(vb[0, 0, 0, 0]) == -1.0
    kb2, _ = integ._kv_blocks_f32(kc, torch.tensor([2]))     # no layer -> scale 1
    assert float(kb2[0, 0, 0, 0]) == 1.5
