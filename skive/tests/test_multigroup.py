"""Multi KV-cache-group (gpt-oss) plumbing tests -- pure-Python stubs, no GPU/vLLM.

Covers the riskiest new code for hybrid models:
  * _skive_full_group picks the TRUE full-attention group (not group 0)
  * _skive_layers scores only that group's attention layers
  * skive_reclaim frees blocks in the group the worker evicted from
  * the hybrid kernel/manager block-size guard disables eviction safely
"""
import sys
import types

import pytest

import kv_evict.integration as integ


# ---- stand-ins for vllm.v1.kv_cache_interface specs -------------------------
class FullAttentionSpec:
    def __init__(self, sliding_window=None):
        self.sliding_window = sliding_window


class SlidingWindowSpec:
    def __init__(self, sliding_window=128):
        self.sliding_window = sliding_window


class _Group:
    def __init__(self, spec, names):
        self.kv_cache_spec = spec
        self.layer_names = list(names)


class _Runner:
    def __init__(self, groups):
        self.kv_cache_config = types.SimpleNamespace(kv_cache_groups=groups)


@pytest.fixture
def fake_vllm_specs(monkeypatch):
    """_skive_full_group imports FullAttentionSpec from vllm.v1.kv_cache_interface."""
    mod = types.ModuleType("vllm.v1.kv_cache_interface")
    mod.FullAttentionSpec = FullAttentionSpec
    mod.SlidingWindowSpec = SlidingWindowSpec
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.v1", types.ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.kv_cache_interface", mod)
    return mod


# ---- group resolution ----------------------------------------------------------
def test_full_group_prefers_pure_full_attention(fake_vllm_specs):
    # gpt-oss style: group 0 = sliding window (even layers), group 1 = full (odd)
    r = _Runner([_Group(SlidingWindowSpec(128), ["l0", "l2"]),
                 _Group(FullAttentionSpec(), ["l1", "l3"])])
    assert integ._skive_full_group(r) == (1, {"l1", "l3"})
    assert integ._skive_full_group(r) == (1, {"l1", "l3"})  # cached


def test_full_group_single_group_scores_all_layers(fake_vllm_specs):
    r = _Runner([_Group(FullAttentionSpec(), ["l0", "l1"])])
    assert integ._skive_full_group(r) == (0, None)


def test_full_group_falls_back_to_windowed_full_spec(fake_vllm_specs):
    # non-hybrid allocator: the only FullAttentionSpec carries a sliding_window
    r = _Runner([_Group(SlidingWindowSpec(128), ["a"]),
                 _Group(FullAttentionSpec(sliding_window=128), ["b"])])
    assert integ._skive_full_group(r) == (1, {"b"})


def test_full_group_no_groups_defaults_to_group_zero():
    assert integ._skive_full_group(_Runner([])) == (0, None)


def test_full_group_via_attn_groups_fallback():
    # no kv_cache_config on the runner -> resolve from attn_groups instead
    ag_sw = types.SimpleNamespace(kv_cache_spec=SlidingWindowSpec(),
                                  layer_names=["l0"], kv_cache_group_id=0)
    ag_full = types.SimpleNamespace(kv_cache_spec=FullAttentionSpec(),
                                    layer_names=["l1"], kv_cache_group_id=1)
    r = types.SimpleNamespace(attn_groups=[[ag_sw], [ag_full]])
    assert integ._skive_full_group(r) == (1, {"l1"})


def test_unresolved_multigroup_refuses_to_evict(capsys):
    # two block tables but no inspectable group specs -> refuse, don't guess 0
    r = types.SimpleNamespace()
    r.input_batch = types.SimpleNamespace(
        block_table=types.SimpleNamespace(block_tables=[object(), object()]))
    assert integ._skive_full_group(r) == (0, None)
    assert integ._skive_block_table(r) is None
    assert "unresolved" in capsys.readouterr().out
    assert integ._skive_block_table(r) is None  # warns only once
    assert capsys.readouterr().out == ""


# ---- layer filtering -----------------------------------------------------------
def test_skive_layers_filters_to_full_group(fake_vllm_specs):
    class Attention:  # type(m).__name__ == "Attention"
        pass

    class Other:
        pass

    l0, l1, l2, l3 = Attention(), Attention(), Attention(), Attention()
    r = _Runner([_Group(SlidingWindowSpec(), ["l0", "l2"]),
                 _Group(FullAttentionSpec(), ["l1", "l3"])])
    r.compilation_config = types.SimpleNamespace(
        static_forward_context={"l0": l0, "l1": l1, "x": Other(), "l2": l2, "l3": l3})
    assert integ._skive_layers(r) == [l1, l3]


# ---- reclaim routing -----------------------------------------------------------
class _Blk:
    def __init__(self, bid, is_null=False):
        self.block_id, self.is_null = bid, is_null


class _Pool:
    def __init__(self):
        self.null_block = _Blk(0, True)
        self.freed = []

    def free_blocks(self, blocks):
        self.freed.extend(b.block_id for b in blocks)

    def get_num_free_blocks(self):
        return 999


class _Mgr:
    def __init__(self, req_to_blocks):
        self.req_to_blocks = req_to_blocks


class _KVM:
    def __init__(self, managers):
        self.block_pool = _Pool()
        self.coordinator = types.SimpleNamespace(single_type_managers=tuple(managers))


def test_reclaim_routes_to_group_of_the_evicted_block():
    sw = _Mgr({"r": [_Blk(10), _Blk(11)]})
    full = _Mgr({"r": [_Blk(20), _Blk(21), _Blk(22)]})
    kvm = _KVM([sw, full])
    assert integ.skive_reclaim(kvm, [("r", 1, 1), ("r", 2, 1)]) == 2
    assert kvm.block_pool.freed == [21, 22]
    assert full.req_to_blocks["r"][1].is_null and full.req_to_blocks["r"][2].is_null
    assert not sw.req_to_blocks["r"][1].is_null  # sliding-window group untouched


def test_reclaim_legacy_pairs_use_group_zero():
    m0 = _Mgr({"r": [_Blk(5), _Blk(6)]})
    m1 = _Mgr({"r": [_Blk(7)]})
    kvm = _KVM([m0, m1])
    assert integ.skive_reclaim(kvm, [("r", 0)]) == 1
    assert kvm.block_pool.freed == [5]


def test_reclaim_skips_null_bad_index_bad_group_unknown_req():
    m0 = _Mgr({"r": [_Blk(0, True), _Blk(9)]})
    kvm = _KVM([m0])
    assert integ.skive_reclaim(
        kvm, [("r", 0, 0), ("r", 5, 0), ("r", 1, 7), ("zz", 0, 0)]) == 0
    assert kvm.block_pool.freed == []


# ---- hybrid block-size guard ---------------------------------------------------
def test_block_table_hybrid_guard_disables_eviction(fake_vllm_specs, capsys):
    r = _Runner([_Group(FullAttentionSpec(), ["l0"])])
    r.input_batch = types.SimpleNamespace(
        block_table=[types.SimpleNamespace(use_hybrid_blocks=True)])
    assert integ._skive_block_table(r) is None
    assert "hybrid" in capsys.readouterr().out
    assert integ._skive_block_table(r) is None      # second call: silent, no crash
    assert capsys.readouterr().out == ""


def test_block_table_uses_full_group_index(fake_vllm_specs):
    bt_sw = types.SimpleNamespace(use_hybrid_blocks=False, tag="sw")
    bt_full = types.SimpleNamespace(use_hybrid_blocks=False, tag="full")
    r = _Runner([_Group(SlidingWindowSpec(), ["a"]), _Group(FullAttentionSpec(), ["b"])])
    r.input_batch = types.SimpleNamespace(block_table=[bt_sw, bt_full])
    assert integ._skive_block_table(r).tag == "full"
