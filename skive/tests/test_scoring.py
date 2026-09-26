"""Stage 1 tests for kv_evict.scoring.score_blocks.

Pure CPU, no GPU, no vLLM. Every case is hand-built so the expected score is
known by arithmetic, not by trusting the implementation.
"""

import math

import pytest
import torch

from kv_evict.scoring import DEFAULT_EPS, score_blocks


def test_known_values_all_ones():
    # A block of N ones has L2 norm sqrt(N). With k == v, score ~= 1.
    n = 8  # block_size * heads * head_dim, flattened
    k = torch.ones(1, n)
    v = torch.ones(1, n)
    score = score_blocks(k, v)
    expected = math.sqrt(n) / (math.sqrt(n) + DEFAULT_EPS)
    assert score.shape == (1,)
    assert score.dtype == torch.float32
    assert score.item() == pytest.approx(expected, rel=1e-6)


def test_ratio_scales_with_value_magnitude():
    # Same K for both blocks; block 1 has 3x the value magnitude => 3x score.
    k = torch.ones(2, 4)
    v = torch.stack([torch.ones(4), 3.0 * torch.ones(4)])
    score = score_blocks(k, v)
    assert score[1].item() == pytest.approx(3.0 * score[0].item(), rel=1e-6)


def test_ordering_and_argmin_picks_lowest_ratio():
    # Build three blocks with deliberately different v/k ratios.
    # Block layout: [num_blocks=3, features=4]
    k = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],  # k_norm = 2
            [1.0, 1.0, 1.0, 1.0],  # k_norm = 2
            [2.0, 2.0, 2.0, 2.0],  # k_norm = 4
        ]
    )
    v = torch.tensor(
        [
            [4.0, 4.0, 4.0, 4.0],  # v_norm = 8  -> ratio ~ 4.0  (most important)
            [1.0, 1.0, 1.0, 1.0],  # v_norm = 2  -> ratio ~ 1.0
            [1.0, 1.0, 1.0, 1.0],  # v_norm = 2  -> ratio ~ 0.5  (least -> evict)
        ]
    )
    score = score_blocks(k, v)
    assert score[0] > score[1] > score[2]
    # "evict first" == argmin
    assert torch.argmin(score).item() == 2
    assert torch.argmax(score).item() == 0


def test_eps_guard_zero_key_no_inf_or_nan():
    # All-zero K block must not produce inf/nan thanks to the eps guard.
    k = torch.zeros(1, 4)
    v = torch.ones(1, 4)
    score = score_blocks(k, v)
    assert torch.isfinite(score).all()
    # v_norm = 2, denom = eps -> large but finite
    assert score.item() == pytest.approx(2.0 / DEFAULT_EPS, rel=1e-3)


def test_empty_block_zero_over_eps_is_zero():
    # A fully-zero block (k=v=0) scores exactly 0 -> evicted first, which is
    # the desired behaviour for an empty/unused block.
    k = torch.zeros(1, 4)
    v = torch.zeros(1, 4)
    score = score_blocks(k, v)
    assert score.item() == pytest.approx(0.0, abs=1e-9)


def test_zero_blocks_returns_empty():
    k = torch.empty(0, 16, 4, 8)
    v = torch.empty(0, 16, 4, 8)
    score = score_blocks(k, v)
    assert score.shape == (0,)
    assert score.dtype == torch.float32


def test_multidim_block_shape_matches_flattened():
    # The real cache shape [num_blocks, block_size, heads, head_dim] must give
    # the same scores as its flattened-per-block equivalent.
    torch.manual_seed(0)
    k = torch.randn(5, 16, 4, 8)
    v = torch.randn(5, 16, 4, 8)
    score_md = score_blocks(k, v)
    score_flat = score_blocks(k.reshape(5, -1), v.reshape(5, -1))
    assert torch.allclose(score_md, score_flat, rtol=1e-6, atol=1e-6)
    assert score_md.shape == (5,)


def test_fp16_input_returns_finite_float32():
    # Half precision in -> float32 out, no overflow, sane values.
    torch.manual_seed(1)
    k = torch.randn(3, 64).to(torch.float16)
    v = torch.randn(3, 64).to(torch.float16)
    score = score_blocks(k, v)
    assert score.dtype == torch.float32
    assert torch.isfinite(score).all()


def test_shape_mismatch_raises():
    k = torch.ones(2, 4)
    v = torch.ones(3, 4)
    with pytest.raises(ValueError):
        score_blocks(k, v)


def test_one_d_input_raises():
    with pytest.raises(ValueError):
        score_blocks(torch.ones(4), torch.ones(4))


def test_nonpositive_eps_raises():
    k = torch.ones(1, 4)
    v = torch.ones(1, 4)
    with pytest.raises(ValueError):
        score_blocks(k, v, eps=0.0)


def test_device_preserved_cpu():
    k = torch.ones(2, 4)
    v = torch.ones(2, 4)
    score = score_blocks(k, v)
    assert score.device == k.device
