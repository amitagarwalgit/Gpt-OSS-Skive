"""Tests for the SKIVE-style value x attention block scorer."""

import torch

from kv_evict.scoring import score_blocks_value_attention


def test_shape_and_blockcount():
    torch.manual_seed(0)
    q = torch.randn(4, 8)            # 4 heads, head_size 8
    k = torch.randn(20, 4, 8)        # 20 tokens, 4 kv heads (MHA), head_size 8
    v = torch.randn(20, 4, 8)
    s = score_blocks_value_attention(q, k, v, block_size=8, num_queries_per_kv=1)
    assert s.shape == (3,)           # ceil(20/8) = 3
    assert s.dtype == torch.float32
    assert torch.isfinite(s).all()


def test_high_attention_high_value_block_scores_highest():
    # Build 2 blocks of 4 tokens, head_size 4, single head.
    # Block 1's keys align with the query AND have large values => should win.
    head_size = 4
    q = torch.zeros(1, head_size)
    q[0, 0] = 10.0                    # query points along axis 0
    k = torch.zeros(8, 1, head_size)
    v = torch.zeros(8, 1, head_size)
    # block 0 (tokens 0-3): keys orthogonal to q, small values
    k[0:4, 0, 1] = 1.0
    v[0:4, 0, :] = 0.1
    # block 1 (tokens 4-7): keys aligned with q, large values
    k[4:8, 0, 0] = 1.0
    v[4:8, 0, :] = 5.0
    s = score_blocks_value_attention(q, k, v, block_size=4, num_queries_per_kv=1,
                                     scale=1.0)
    assert s[1] > s[0]               # attended + high-value block ranks higher
    # argmin (evict first) should be the low-attention low-value block 0
    assert torch.argmin(s).item() == 0


def test_gqa_grouping_runs():
    # 8 query heads, 2 kv heads => num_queries_per_kv = 4
    torch.manual_seed(1)
    q = torch.randn(8, 16)
    k = torch.randn(33, 2, 16)
    v = torch.randn(33, 2, 16)
    s = score_blocks_value_attention(q, k, v, block_size=16, num_queries_per_kv=4)
    assert s.shape == (3,)           # ceil(33/16) = 3
    assert torch.isfinite(s).all()


def test_empty():
    q = torch.randn(4, 8)
    k = torch.empty(0, 4, 8)
    v = torch.empty(0, 4, 8)
    s = score_blocks_value_attention(q, k, v, block_size=8)
    assert s.shape == (0,)
