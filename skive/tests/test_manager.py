"""Stage 2 tests for kv_evict.manager.BlockEvictionManager.

Pure CPU, no GPU, no vLLM. The two load-bearing invariants:
  * should_evict never fires mid-block.
  * pick_block never returns a protected (sink/local) block.
"""

import pytest
import torch

from kv_evict.manager import BlockEvictionManager, EvictionConfig
from kv_evict.scoring import score_blocks


def mgr(block_size=4, kv_budget=4, sink=1, local=1):
    m = BlockEvictionManager(
        EvictionConfig(
            block_size=block_size,
            kv_budget=kv_budget,
            num_sink_blocks=sink,
            num_local_blocks=local,
        )
    )
    m.register_seq(0)
    return m


# --- config validation ----------------------------------------------------
def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        EvictionConfig(block_size=0, kv_budget=4)
    with pytest.raises(ValueError):
        EvictionConfig(block_size=4, kv_budget=0)
    with pytest.raises(ValueError):
        EvictionConfig(block_size=4, kv_budget=4, num_sink_blocks=-1)


def test_config_rejects_budget_smaller_than_protected():
    # budget < sink + local => nothing is ever evictable.
    with pytest.raises(ValueError):
        EvictionConfig(block_size=4, kv_budget=2, num_sink_blocks=2, num_local_blocks=2)


# --- block accounting ------------------------------------------------------
def test_num_blocks_counts_full_blocks_minus_evicted():
    m = mgr(block_size=4)
    assert m.num_blocks(0, 0) == 0
    assert m.num_blocks(0, 1) == 1      # partial block still counts as 1
    assert m.num_blocks(0, 4) == 1
    assert m.num_blocks(0, 5) == 2
    assert m.num_blocks(0, 16) == 4
    m.record_eviction(0)
    assert m.num_blocks(0, 16) == 3     # one physical block freed


# --- invariant 1: never fires mid-block ------------------------------------
def test_should_evict_false_mid_block_even_when_over_budget():
    m = mgr(block_size=4, kv_budget=2, sink=1, local=1)
    # 13 tokens -> ceil(13/4)=4 blocks, budget 2 => over budget, but 13%4 != 0.
    assert m.num_blocks(0, 13) == 4
    assert m.should_evict(0, 13) is False
    assert m.should_evict(0, 14) is False
    assert m.should_evict(0, 15) is False
    # 16 tokens completes a block -> boundary AND over budget -> fires.
    assert m.should_evict(0, 16) is True


def test_should_evict_false_within_budget_at_boundary():
    m = mgr(block_size=4, kv_budget=4, sink=1, local=1)
    # 16 tokens -> 4 blocks == budget, not over.
    assert m.should_evict(0, 16) is False
    # 20 tokens -> 5 blocks > budget, at boundary -> fires.
    assert m.should_evict(0, 20) is True


def test_should_evict_zero_length_false():
    m = mgr()
    assert m.should_evict(0, 0) is False


# --- invariant 2: never picks a protected block ----------------------------
def test_pick_block_excludes_protected_exhaustive():
    # 6 blocks, sink=2, local=2 -> protected {0,1,4,5}, candidates {2,3}.
    m = mgr(block_size=4, kv_budget=4, sink=2, local=2)
    total = 24  # 6 blocks
    assert m.num_blocks(0, total) == 6
    protected = m.protected_indices(6)
    assert protected == {0, 1, 4, 5}
    # Try many score vectors, including ones whose global min is protected.
    torch.manual_seed(0)
    for _ in range(200):
        scores = torch.randn(6)
        picked = m.pick_block(0, total, scores)
        assert picked not in protected
        assert picked in {2, 3}


def test_pick_block_is_argmin_among_candidates_not_global():
    # Global min is a protected sink block; the answer must be the candidate
    # argmin, not the global argmin.
    m = mgr(block_size=4, kv_budget=4, sink=1, local=1)
    total = 24  # 6 blocks, protected {0, 5}
    scores = [-100.0, 5.0, 9.0, 2.0, 7.0, -200.0]
    #            ^prot                ^min-candidate(idx3)   ^prot(global min)
    assert m.pick_block(0, total, scores) == 3


def test_pick_block_ties_resolve_to_lowest_index():
    m = mgr(block_size=4, kv_budget=4, sink=1, local=1)
    total = 24  # 6 blocks, candidates {1,2,3,4}
    scores = [9.0, 1.0, 1.0, 1.0, 1.0, 9.0]
    assert m.pick_block(0, total, scores) == 1


# --- desync / safety guards ------------------------------------------------
def test_pick_block_rejects_score_length_mismatch():
    m = mgr(block_size=4, kv_budget=4, sink=1, local=1)
    total = 24  # expects 6 scores
    with pytest.raises(ValueError):
        m.pick_block(0, total, [1.0, 2.0, 3.0])


def test_pick_block_raises_when_all_protected():
    # n <= sink + local => no candidate. Use a config where this can happen
    # transiently (small block count early on).
    m = BlockEvictionManager(
        EvictionConfig(block_size=4, kv_budget=3, num_sink_blocks=2, num_local_blocks=1)
    )
    m.register_seq(0)
    total = 8  # 2 blocks, protected {0,1} -> no candidate
    with pytest.raises(RuntimeError):
        m.pick_block(0, total, [1.0, 2.0])


def test_unregistered_seq_raises():
    m = BlockEvictionManager(EvictionConfig(block_size=4, kv_budget=4))
    with pytest.raises(KeyError):
        m.should_evict(99, 16)


# --- end-to-end decode simulation ------------------------------------------
def test_decode_simulation_stays_at_budget_and_protects_sinks():
    # Simulate a long decode with the eviction loop the Stage 4 integration
    # will run. After every boundary, blocks_in_cache must be <= budget, and a
    # sink block must never be the one chosen for eviction.
    block_size, budget, sink, local = 4, 4, 1, 2
    m = mgr(block_size=block_size, kv_budget=budget, sink=sink, local=local)

    prompt_len = 4
    evicted_indices = []
    for step in range(1, 80):
        total = prompt_len + step
        # Stage-4-style loop: shed whole blocks until back within budget.
        while m.should_evict(0, total):
            n = m.num_blocks(0, total)
            # Deterministic fake scores: make middle block index 'sink' cheap
            # to confirm it is still never picked because it's protected.
            scores = [float(i) for i in range(n)]
            scores[0] = -1.0  # sink would be global min if not protected
            picked = m.pick_block(0, total, scores)
            assert picked >= sink, "sink block must never be evicted"
            assert picked < n - local, "local window must never be evicted"
            evicted_indices.append(picked)
            m.record_eviction(0)
        # Invariant: after shedding, never above budget at a boundary.
        if total % block_size == 0:
            assert m.num_blocks(0, total) <= budget

    assert evicted_indices, "expected at least one eviction over the run"


# --- integration with Stage 1 scoring --------------------------------------
def test_pick_block_consumes_real_scores_from_scoring():
    m = mgr(block_size=4, kv_budget=4, sink=1, local=1)
    total = 24  # 6 blocks
    n = m.num_blocks(0, total)
    torch.manual_seed(3)
    k = torch.randn(n, 4, 2, 8)
    v = torch.randn(n, 4, 2, 8)
    scores = score_blocks(k, v)  # real Stage 1 output, a torch tensor
    picked = m.pick_block(0, total, scores)
    assert picked not in m.protected_indices(n)
    assert 0 <= picked < n
