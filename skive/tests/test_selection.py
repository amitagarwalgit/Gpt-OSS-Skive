"""Unit tests for kv_evict.selection (pure NumPy, no GPU)."""
import numpy as np

from kv_evict.selection import NULL_BLOCK_ID, choose_lowest, plan_eviction


def test_plan_under_budget_no_evict():
    row = np.array([5, 6, 7, 8])
    real, cand, n = plan_eviction(row, 1, 1, kv_budget=4)
    assert n == 0 and cand.size == 0
    assert real.tolist() == [0, 1, 2, 3]


def test_plan_excludes_null_and_protected():
    # logical: [5, NULL, 7, 8, 9, 10, 11] -> real positions [0,2,3,4,5,6]
    row = np.array([5, NULL_BLOCK_ID, 7, 8, 9, 10, 11])
    real, cand, n = plan_eviction(row, num_sink_blocks=1, num_local_blocks=2, kv_budget=4)
    assert real.tolist() == [0, 2, 3, 4, 5, 6]
    # 6 real - budget 4 = 2 to evict; candidates k in [1, 6-2) = [1,2,3]
    assert n == 2 and cand.tolist() == [1, 2, 3]


def test_plan_caps_to_candidates():
    row = np.arange(1, 11)  # 10 real blocks
    real, cand, n = plan_eviction(row, num_sink_blocks=4, num_local_blocks=4, kv_budget=2)
    # wants 8 evictions but only k in [4,6) = 2 candidates
    assert cand.tolist() == [4, 5] and n == 2


def test_plan_all_protected():
    row = np.arange(1, 6)
    _, cand, n = plan_eviction(row, num_sink_blocks=3, num_local_blocks=3, kv_budget=1)
    assert n == 0 and cand.size == 0


def test_choose_lowest_stable_ties():
    cand = np.array([1, 2, 3, 4])
    scores = [9.0, 0.5, 0.2, 0.5, 0.9]  # indexed by k
    chosen = choose_lowest(cand, scores, 2)
    # lowest among candidates: k=2 (0.2), then tie 0.5 -> lowest index k=1
    assert chosen.tolist() == [2, 1]
    # the tie-break must not hide a genuinely lower candidate
    assert choose_lowest(cand, [9.0, 0.5, 0.2, 0.5, 0.1], 2).tolist() == [4, 2]


def test_choose_lowest_matches_sorted_reference():
    rng = np.random.default_rng(0)
    for _ in range(50):
        nreal = int(rng.integers(3, 40))
        scores = rng.random(nreal).round(2).tolist()  # rounded -> ties
        lo, hi = 1, nreal - 1
        cand = np.arange(lo, hi)
        k = int(rng.integers(0, cand.size + 1))
        ref = sorted(cand.tolist(), key=lambda i: (scores[i], i))[:k]
        got = choose_lowest(cand, scores, k).tolist()
        assert got == ref
