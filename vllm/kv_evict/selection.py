"""Vectorized eviction planning + candidate selection (IMP-10).

Pure NumPy, no torch / vLLM. Replaces the per-request Python list-comprehension
and ``sorted(key=lambda)`` that used to run on the decode hot path.

Terminology (matches ``manager.BlockEvictionManager``):
  * a *real* block is a block-table entry that is not the null block;
  * indices ``k`` index the ordered list of a request's real blocks;
  * the first ``num_sink_blocks`` and last ``num_local_blocks`` real blocks are
    protected and never evicted.
"""

from __future__ import annotations

import numpy as np

# vLLM's reserved placeholder block (see integration.NULL_BLOCK_ID).
NULL_BLOCK_ID = 0


def plan_eviction(
    row: np.ndarray,
    num_sink_blocks: int,
    num_local_blocks: int,
    kv_budget: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Decide *how many* blocks to evict and *which are eligible*.

    Args:
        row: 1-D int array of one request's block-table entries, in logical
            order (may contain ``NULL_BLOCK_ID`` for already-evicted slots).
        num_sink_blocks: leading real blocks that are never evicted.
        num_local_blocks: trailing real blocks that are never evicted.
        kv_budget: max real blocks the request may keep.

    Returns:
        ``(real_indices, candidates_k, num_to_evict)`` where
        ``real_indices`` are the logical positions of real blocks,
        ``candidates_k`` are indices *into* ``real_indices`` that are evictable,
        and ``num_to_evict`` is how many of them to evict (0 => nothing to do).
    """
    row = np.asarray(row)
    real_indices = np.flatnonzero(row != NULL_BLOCK_ID)
    num_real = int(real_indices.size)
    num_to_evict = num_real - int(kv_budget)
    if num_to_evict <= 0:
        return real_indices, real_indices[:0], 0
    lo = int(num_sink_blocks)
    hi = num_real - int(num_local_blocks)
    if hi <= lo:  # everything is protected
        return real_indices, real_indices[:0], 0
    candidates_k = np.arange(lo, hi, dtype=np.int64)
    num_to_evict = min(num_to_evict, int(candidates_k.size))
    return real_indices, candidates_k, num_to_evict


def choose_lowest(
    candidates_k: np.ndarray, scores, num_to_evict: int
) -> np.ndarray:
    """Return the ``num_to_evict`` candidates with the lowest importance.

    ``scores[k]`` is the importance of the k-th real block (lower = evict
    first). Ties resolve to the lowest index (stable sort), matching the
    previous ``sorted(key=(score, k))`` behaviour exactly.
    """
    if num_to_evict <= 0 or len(candidates_k) == 0:
        return np.empty(0, dtype=np.int64)
    ck = np.asarray(candidates_k, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)[ck]
    order = np.argsort(s, kind="stable")
    return ck[order[:num_to_evict]]
