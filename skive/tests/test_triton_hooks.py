"""SKIVE hooks in vLLM's TRITON attention backend (vllm_patches/patch_vllm.py edit K).

gpt-oss needs attention sinks; on GPUs without FlashAttention-3 (RTX PRO 6000 /
SM120, L40S / L4 / Ada) vLLM serves it through TRITON_ATTN, so the query
capture and sparse-gather hooks must exist there too. These tests apply edit K
to (a) a minimal stub carrying the real anchors and (b) the fork's real file
when present, then exercise the patched code: compaction of evicted (null)
blocks on decode rows, the CUDA-graph and sliding-window guards, and the query
capture. Pure Python + torch, no GPU / vLLM.
"""
import importlib
import importlib.util
import os
import py_compile
import shutil
import sys
import tempfile
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
if ROOT not in sys.path:            # the patched code imports kv_evict.compaction
    sys.path.insert(0, ROOT)
STAGE4 = os.path.join(ROOT)                            # patch_vllm.py lives in skive/
# the shipped (already patched) overlay copy of vLLM's triton_attn.py
FORK_TA = os.path.join(ROOT, "..", "vllm", "v1", "attention", "backends",
                       "triton_attn.py")

# The exact anchor lines of vllm 0.23.0 triton_attn.py, in a runnable stub.
STUB = '''"""stub of vllm/v1/attention/backends/triton_attn.py (real anchors only)."""

from dataclasses import dataclass
from typing import ClassVar
from typing import Generic, TypeVar

import torch

M = TypeVar("M")


class AttentionMetadataBuilder(Generic[M]):
    pass


@dataclass
class TritonAttentionMetadata:
    num_actual_tokens: int


class TritonAttentionMetadataBuilder(AttentionMetadataBuilder[TritonAttentionMetadata]):
    def __init__(self, vllm_config, kv_cache_spec, block_size=16,
                 decode_cudagraph_enabled=False):
        self.vllm_config = vllm_config
        self.kv_cache_spec = kv_cache_spec
        self.block_size = block_size
        self.decode_cudagraph_enabled = decode_cudagraph_enabled

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        use_cascade = common_prefix_len > 0
        return block_table_tensor, seq_lens


class TritonAttentionImpl:
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for TritonAttentionImpl"
            )
        return output
'''


def _load_patcher(vroot):
    """Import vllm_patches/patch_vllm.py pointed at `vroot` (no vllm import needed)."""
    os.environ["SKIVE_VLLM_ROOT"] = vroot
    os.environ["SKIVE_PATCH_DIR"] = os.path.join(vroot, "_patches")
    os.environ["SKIVE_IMPORT_ROOT"] = "kv_evict"   # tests import kv_evict.* directly
    if STAGE4 not in sys.path:
        sys.path.insert(0, STAGE4)
    if "patch_vllm" in sys.modules:
        return importlib.reload(sys.modules["patch_vllm"])
    return importlib.import_module("patch_vllm")


def _apply_k(patcher):
    return [fn() for letter, fn in patcher.EDITS if letter == "K"]


def _ta_path(vroot):
    p = os.path.join(vroot, "v1", "attention", "backends")
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, "triton_attn.py")


@pytest.fixture
def stub_root(tmp_path):
    vroot = str(tmp_path / "vllm")
    with open(_ta_path(vroot), "w") as f:
        f.write(STUB)
    return vroot


@pytest.fixture
def fake_specs(monkeypatch):
    class FullAttentionSpec:
        sliding_window = None

    class SlidingWindowSpec:
        sliding_window = 128

    mod = types.ModuleType("vllm.v1.kv_cache_interface")
    mod.FullAttentionSpec, mod.SlidingWindowSpec = FullAttentionSpec, SlidingWindowSpec
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.v1", types.ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.kv_cache_interface", mod)
    return mod


def _import_patched(path, metric="value_attention"):
    os.environ["SKIVE_METRIC"] = metric
    os.environ["SKIVE_SPARSE_GATHER"] = "1"
    spec = importlib.util.spec_from_file_location("triton_attn_patched_" + str(abs(hash(path))), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- patch application ---------------------------------------------------------
def test_k_edits_apply_once_and_compile(stub_root):
    patcher = _load_patcher(stub_root)
    assert all(_apply_k(patcher)), "every K edit should apply on a pristine file"
    src = open(_ta_path(stub_root)).read()
    assert "import os  # SKIVE" in src
    assert "_SKIVE_CAPTURE_Q" in src and "_SKIVE_SPARSE" in src
    assert "def _skive_can_compact" in src
    assert "compact_block_table_torch" in src and "layer._skive_q = query.detach()" in src
    # import os must precede the first `from` import (ruff E402 / isort order)
    assert src.index("import os") < src.index("from dataclasses import dataclass")
    py_compile.compile(_ta_path(stub_root), doraise=True)
    # idempotent: a second run touches nothing
    assert not any(_apply_k(patcher))
    assert open(_ta_path(stub_root)).read() == src


@pytest.mark.skipif(not os.path.exists(FORK_TA), reason="overlay triton_attn.py missing")
def test_k_edits_on_real_fork_file(tmp_path):
    vroot = str(tmp_path / "fork")
    shutil.copy(FORK_TA, _ta_path(vroot))
    patcher = _load_patcher(vroot)
    _apply_k(patcher)                      # applies, or skips if the fork is already patched
    src = open(_ta_path(vroot)).read()
    for marker in ("_SKIVE_CAPTURE_Q = ", "_SKIVE_SPARSE = ", "def _skive_can_compact",
                   "import compact_block_table_torch", "layer._skive_q = query.detach()"):
        assert src.count(marker) == 1, f"{marker!r} should appear exactly once"
    py_compile.compile(_ta_path(vroot), doraise=True)
    assert not any(_apply_k(patcher))      # idempotent on the real file too


# ---- behaviour of the patched code ----------------------------------------------
def _cam(block_table, seq_lens, query_lens):
    qsl = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    qsl[1:] = torch.cumsum(torch.as_tensor(query_lens, dtype=torch.int32), 0)
    return types.SimpleNamespace(
        num_actual_tokens=int(qsl[-1]), max_query_len=int(max(query_lens)),
        query_start_loc=qsl, seq_lens=torch.as_tensor(seq_lens, dtype=torch.int32),
        block_table_tensor=torch.as_tensor(block_table, dtype=torch.int32),
        slot_mapping=torch.zeros(int(qsl[-1]), dtype=torch.int64), num_reqs=len(query_lens))


def test_sparse_gather_compacts_decode_rows(stub_root, fake_specs):
    _apply_k(_load_patcher(stub_root))
    ta = _import_patched(_ta_path(stub_root))
    cfg = types.SimpleNamespace(cache_config=types.SimpleNamespace(kv_evict_enabled=True))
    b = ta.TritonAttentionMetadataBuilder(cfg, fake_specs.FullAttentionSpec(), block_size=16)
    # row 0: decode, block 1 evicted (null=0) -> 3 blocks read, seq_len 64-16
    # row 1: prefill (query_len 5), has a null but must NOT be compacted
    bt, sl = b.build(0, _cam([[7, 0, 9, 10], [3, 0, 5, 0]], [64, 40], [1, 5]))
    assert bt[0].tolist() == [7, 9, 10, 0] and int(sl[0]) == 48
    assert bt[1].tolist() == [3, 0, 5, 0] and int(sl[1]) == 40


def test_sparse_gather_skipped_for_sliding_window_and_cudagraph(stub_root, fake_specs):
    _apply_k(_load_patcher(stub_root))
    ta = _import_patched(_ta_path(stub_root))
    cfg = types.SimpleNamespace(cache_config=types.SimpleNamespace(kv_evict_enabled=True))
    cam = _cam([[7, 0, 9, 10]], [64], [1])
    assert ta._skive_can_compact(fake_specs.SlidingWindowSpec()) is False
    assert ta._skive_can_compact(fake_specs.FullAttentionSpec()) is True
    # sliding-window group: vLLM's own null blocks live here -> untouched
    bt, sl = ta.TritonAttentionMetadataBuilder(cfg, fake_specs.SlidingWindowSpec()).build(0, cam)
    assert bt[0].tolist() == [7, 0, 9, 10] and int(sl[0]) == 64
    # full CUDA-graph decode capture: metadata buffers must stay persistent -> untouched
    bt, sl = ta.TritonAttentionMetadataBuilder(
        cfg, fake_specs.FullAttentionSpec(), decode_cudagraph_enabled=True).build(0, cam)
    assert bt[0].tolist() == [7, 0, 9, 10] and int(sl[0]) == 64
    # eviction off: untouched
    off = types.SimpleNamespace(cache_config=types.SimpleNamespace(kv_evict_enabled=False))
    bt, sl = ta.TritonAttentionMetadataBuilder(off, fake_specs.FullAttentionSpec()).build(0, cam)
    assert bt[0].tolist() == [7, 0, 9, 10] and int(sl[0]) == 64


def test_query_capture_only_for_query_dependent_metrics(stub_root, fake_specs):
    _apply_k(_load_patcher(stub_root))
    q = torch.randn(3, 64, 64)
    for metric, captured in (("value_attention", True), ("h2o", True),
                             ("snapkv", True), ("vk_ratio", False)):
        ta = _import_patched(_ta_path(stub_root), metric=metric)
        layer = types.SimpleNamespace()
        ta.TritonAttentionImpl().forward(layer, q, None, None, None, None, torch.zeros(1))
        assert hasattr(layer, "_skive_q") is captured, metric
        if captured:
            assert torch.equal(layer._skive_q, q)
