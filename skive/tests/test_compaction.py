"""Tests for kv_evict.compaction.compact_block_table (Phase 1 sparse-gather).

Pure CPU/NumPy, no GPU, no vLLM. Load-bearing invariants:
  * Retained (non-null) physical ids survive in their original logical order.
  * reduced_seq_len == seq_len - (#evicted full blocks) * block_size.
  * The read stays self-consistent: ceil(compact_seq / block_size) equals the
    compacted block count (so the kernel gathers exactly the retained blocks).
  * A row with no evictions is returned byte-for-byte unchanged.
"""

import math

import numpy as np
import pytest

from kv_evict.compaction import NULL_BLOCK_ID, compact_block_table

BS = 16  # block_size


def test_no_evictions_is_identity():
    bt = np.array([[7, 8, 9, 0, 0]], dtype=np.int32)  # 3 logical blocks, 2 pad
    n = np.array([3])
    sl = np.array([3 * BS])
    cbt, cn, csl = compact_block_table(bt, n, sl, BS)
    np.testing.assert_array_equal(cbt, bt)
    np.testing.assert_array_equal(cn, n)
    np.testing.assert_array_equal(csl, sl)


def test_single_middle_eviction_shifts_left():
    # logical blocks: [11, NULL, 13, 14]; block 1 evicted.
    bt = np.array([[11, NULL_BLOCK_ID, 13, 14, 0, 0]], dtype=np.int64)
    n = np.array([4])
    sl = np.array([4 * BS])  # all full blocks
    cbt, cn, csl = compact_block_table(bt, n, sl, BS)
    # retained ids, in order, left-packed; tail nulled.
    np.testing.assert_array_equal(cbt[0, :3], [11, 13, 14])
    assert cn[0] == 3
    assert csl[0] == 4 * BS - BS  # one full block removed


def test_multiple_scattered_evictions_preserve_order():
    # [21, N, 23, N, 25, 26]; evict logical 1 and 3.
    bt = np.array([[21, NULL_BLOCK_ID, 23, NULL_BLOCK_ID, 25, 26]], dtype=np.int64)
    n = np.array([6])
    sl = np.array([6 * BS])
    cbt, cn, csl = compact_block_table(bt, n, sl, BS)
    np.testing.assert_array_equal(cbt[0, :4], [21, 23, 25, 26])
    np.testing.assert_array_equal(cbt[0, 4:], [NULL_BLOCK_ID, NULL_BLOCK_ID])
    assert cn[0] == 4
    assert csl[0] == 6 * BS - 2 * BS


def test_rows_are_independent():
    bt = np.array(
        [
            [31, NULL_BLOCK_ID, 33, 0],   # 3 logical, evict middle -> 2
            [41, 42, 43, 44],             # 4 logical, no eviction
        ],
        dtype=np.int64,
    )
    n = np.array([3, 4])
    sl = np.array([3 * BS, 4 * BS])
    cbt, cn, csl = compact_block_table(bt, n, sl, BS)
    np.testing.assert_array_equal(cbt[0, :2], [31, 33])
    assert cn[0] == 2 and csl[0] == 2 * BS
    np.testing.assert_array_equal(cbt[1], [41, 42, 43, 44])  # unchanged
    assert cn[1] == 4 and csl[1] == 4 * BS


def test_partial_last_block_length_preserved():
    # last block partial (9 tokens), only a full middle block evicted.
    # L = 2 full blocks + 9 = 2*BS + 9.  Evict logical block 1 (full).
    bt = np.array([[51, NULL_BLOCK_ID, 53]], dtype=np.int64)
    n = np.array([3])
    sl = np.array([2 * BS + 9])
    cbt, cn, csl = compact_block_table(bt, n, sl, BS)
    np.testing.assert_array_equal(cbt[0, :2], [51, 53])
    assert cn[0] == 2
    assert csl[0] == (2 * BS + 9) - BS  # exactly one full block of tokens gone


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_read_is_self_consistent(dtype):
    # For every row, the kernel reads ceil(seq/bs) blocks == compact_num_blocks.
    bt = np.array(
        [
            [61, NULL_BLOCK_ID, 63, NULL_BLOCK_ID, 65, 0, 0],
            [71, 72, 73, 74, 75, 76, 0],
            [81, NULL_BLOCK_ID, 83, 0, 0, 0, 0],
        ],
        dtype=dtype,
    )
    n = np.array([5, 6, 3])
    # rows all-full except make row 1 last block partial.
    sl = np.array([5 * BS, 5 * BS + 7, 3 * BS])
    cbt, cn, csl = compact_block_table(bt, n, sl, BS)
    assert cbt.dtype == dtype
    for r in range(bt.shape[0]):
        assert math.ceil(csl[r] / BS) == cn[r], f"row {r} read mismatch"


def test_dtype_and_null_id_custom():
    bt = np.array([[91, 5, 93, 5]], dtype=np.int32)  # null id == 5 here
    n = np.array([4])
    sl = np.array([4 * BS])
    cbt, cn, csl = compact_block_table(bt, n, sl, BS, null_block_id=5)
    np.testing.assert_array_equal(cbt[0, :2], [91, 93])
    assert cn[0] == 2 and csl[0] == 2 * BS


def test_all_nonprotected_evicted_row():
    # Extreme: only first and last retained (rest nulled). Order preserved.
    bt = np.array([[1001, NULL_BLOCK_ID, NULL_BLOCK_ID, NULL_BLOCK_ID, 1005]],
                  dtype=np.int64)
    n = np.array([5])
    sl = np.array([5 * BS])
    cbt, cn, csl = compact_block_table(bt, n, sl, BS)
    np.testing.assert_array_equal(cbt[0, :2], [1001, 1005])
    assert cn[0] == 2 and csl[0] == 2 * BS


def test_torch_matches_numpy_random():
    """The GPU/torch variant must agree with the verified NumPy reference on
    the retained region and reduced seq_lens, for random decode batches."""
    import torch

    from kv_evict.compaction import compact_block_table_torch

    rng = np.random.default_rng(0)
    for _ in range(50):
        R = int(rng.integers(1, 6))
        maxb = int(rng.integers(2, 10))
        n = rng.integers(1, maxb + 1, size=R)
        bt = np.zeros((R, maxb), dtype=np.int64)
        seq = np.zeros(R, dtype=np.int64)
        for r in range(R):
            # unique real block ids >= 1 for the logical blocks
            ids = rng.choice(np.arange(1, 500), size=n[r], replace=False)
            # randomly null out some *interior* blocks (never the last -> full)
            for i in range(n[r]):
                if i < n[r] - 1 and rng.random() < 0.4:
                    ids[i] = NULL_BLOCK_ID
            bt[r, : n[r]] = ids
            # last block partial sometimes; evicted blocks are full blocks
            last_fill = int(rng.integers(1, BS + 1))
            seq[r] = (n[r] - 1) * BS + last_fill

        cbt_np, cn_np, csl_np = compact_block_table(bt, n, seq, BS)
        # query_lens all 1 (pure decode) so torch compacts every row
        ql = torch.ones(R, dtype=torch.long)
        cbt_t, csl_t = compact_block_table_torch(
            torch.from_numpy(bt), torch.from_numpy(seq), BS, query_lens=ql
        )
        cbt_t = cbt_t.numpy()
        csl_t = csl_t.numpy()
        np.testing.assert_array_equal(csl_t, csl_np)
        for r in range(R):
            k = int(cn_np[r])
            np.testing.assert_array_equal(cbt_t[r, :k], cbt_np[r, :k])
