"""Stage 4 vLLM patcher (idempotent, assert-guarded).

Applies SKIVE KV-eviction changes to the *pinned* installed vLLM 0.23.0 tree.
site-packages is not a git repo, so this script is the source of truth for
every edit: it backs up each file to <file>.skive_orig (once), asserts each
anchor is present exactly once and not already patched, applies the edit, and
writes a unified diff to vllm_patches/patches/<name>.patch for review.

Run inside the `skive` conda env on the box. Re-runnable: already-applied
edits are detected and skipped.

Stage 4a (this revision): config + CLI flag plumbing ONLY. No eviction logic,
no behavior change when --kv-evict-enabled is off (the default).
"""

import difflib
import os
import sys

# Target vLLM tree: an explicit source checkout (SKIVE_VLLM_ROOT, e.g. when
# building a fork) takes precedence; otherwise the installed package.
VROOT = os.environ.get("SKIVE_VLLM_ROOT")
if not VROOT:
    import vllm

    VROOT = os.path.dirname(vllm.__file__)
PATCH_DIR = os.environ.get(
    "SKIVE_PATCH_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "patches"),
)
os.makedirs(PATCH_DIR, exist_ok=True)

# marker present in every inserted block, used for idempotency detection
MARKER = "SKIVE:"

# Python import root for the kv_evict package as seen from inside vLLM.
# Overlay install -> top-level "kv_evict"; vendored fork -> "vllm.kv_evict".
KVE = os.environ.get("SKIVE_IMPORT_ROOT", "kv_evict")


def _read(path):
    with open(path) as f:
        return f.read()


def _backup(path):
    if os.environ.get("SKIVE_NO_BACKUP") == "1":  # e.g. when patching a git fork
        return
    bak = path + ".skive_orig"
    if not os.path.exists(bak):
        with open(bak, "w") as f:
            f.write(_read(path))


def _emit_diff(name, before, after, path):
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{os.path.relpath(path, VROOT)}",
        tofile=f"b/{os.path.relpath(path, VROOT)}",
    )
    with open(os.path.join(PATCH_DIR, f"{name}.patch"), "w") as f:
        f.writelines(diff)


def apply_edit(path, anchor, insertion, name, before=False):
    """Insert `insertion` adjacent to the unique `anchor` (after by default,
    or before it when before=True)."""
    text = _read(path)
    if MARKER in text and insertion.strip().split("\n")[0] in text:
        print(f"  [skip] {name}: already applied")
        return False
    n = text.count(anchor)
    assert n == 1, f"{name}: anchor found {n} times (expected 1) in {path}"
    _backup(path)
    before_text = text
    repl = insertion + anchor if before else anchor + insertion
    after = text.replace(anchor, repl, 1)
    with open(path, "w") as f:
        f.write(after)
    _emit_diff(name, before_text, after, path)
    print(f"  [ok]   {name}")
    return True


CACHE = os.path.join(VROOT, "config", "cache.py")
ARGS = os.path.join(VROOT, "engine", "arg_utils.py")
RUNNER = os.path.join(VROOT, "v1", "worker", "gpu_model_runner.py")

# --- A: CacheConfig fields ---------------------------------------------------
A_ANCHOR = (
    '    enable_prefix_caching: bool = True\n'
    '    """Whether to enable prefix caching."""\n'
)
A_INSERT = (
    '    kv_evict_enabled: bool = False\n'
    '    """SKIVE: enable value-aware block-wise KV-cache eviction. Default\n'
    '    False; when False, behavior is identical to stock vLLM."""\n'
    '    kv_evict_budget: int | None = None\n'
    '    """SKIVE: max KV blocks retained per sequence before eviction fires.\n'
    '    None disables eviction even if kv_evict_enabled is True."""\n'
    '    kv_evict_num_sink_blocks: int = 0\n'
    '    """SKIVE: number of leading (attention-sink) blocks never evicted."""\n'
    '    kv_evict_num_local_blocks: int = 0\n'
    '    """SKIVE: number of trailing (recent/local) blocks never evicted."""\n'
)

# --- B: EngineArgs mirror fields --------------------------------------------
B_ANCHOR = "    enable_prefix_caching: bool | None = None\n"
B_INSERT = (
    "    kv_evict_enabled: bool = CacheConfig.kv_evict_enabled  # SKIVE:\n"
    "    kv_evict_budget: int | None = CacheConfig.kv_evict_budget\n"
    "    kv_evict_num_sink_blocks: int = CacheConfig.kv_evict_num_sink_blocks\n"
    "    kv_evict_num_local_blocks: int = CacheConfig.kv_evict_num_local_blocks\n"
)

# --- C: CLI add_argument calls ----------------------------------------------
C_ANCHOR = (
    '        cache_group.add_argument(\n'
    '            "--kv-offloading-backend", **cache_kwargs["kv_offloading_backend"]\n'
    '        )\n'
)
C_INSERT = (
    '        cache_group.add_argument(  # SKIVE:\n'
    '            "--kv-evict-enabled", **cache_kwargs["kv_evict_enabled"]\n'
    '        )\n'
    '        cache_group.add_argument(\n'
    '            "--kv-evict-budget", **cache_kwargs["kv_evict_budget"]\n'
    '        )\n'
    '        cache_group.add_argument(\n'
    '            "--kv-evict-num-sink-blocks",\n'
    '            **cache_kwargs["kv_evict_num_sink_blocks"],\n'
    '        )\n'
    '        cache_group.add_argument(\n'
    '            "--kv-evict-num-local-blocks",\n'
    '            **cache_kwargs["kv_evict_num_local_blocks"],\n'
    '        )\n'
)

# --- D: CacheConfig constructor kwargs --------------------------------------
D_ANCHOR = "            enable_prefix_caching=self.enable_prefix_caching,\n"
D_INSERT = (
    "            kv_evict_enabled=self.kv_evict_enabled,  # SKIVE:\n"
    "            kv_evict_budget=self.kv_evict_budget,\n"
    "            kv_evict_num_sink_blocks=self.kv_evict_num_sink_blocks,\n"
    "            kv_evict_num_local_blocks=self.kv_evict_num_local_blocks,\n"
)


# --- E: worker read-only scoring hook (Stage 4b) ----------------------------
# Inserted right after the ModelRunnerOutput is built, BEFORE the
# `if not self.use_async_scheduling:` branch, so it runs under async scheduling
# too. Flag-gated and wrapped in try/except: off => never runs => identical.
E_ANCHOR = (
    "                routed_experts=None,\n"
    "            )\n"
)
E_INSERT = (
    "\n"
    "        # SKIVE: stable post-decode KV-eviction hook. Flag-gated; off (or\n"
    "        # an unset/invalid budget) => skive_post_step is a no-op => stock.\n"
    "        if self.cache_config.kv_evict_enabled:\n"
    "            try:\n"
    f"                from {KVE}.integration import skive_post_step\n"
    "\n"
    "                skive_post_step(self)\n"
    "            except Exception as _skive_e:\n"
    '                logger.warning("SKIVE post-step hook error: %s", _skive_e)\n'
)


def revert():
    """Restore every patched file from its .skive_orig backup."""
    for path in (CACHE, ARGS, RUNNER, CORE, RUNNER_V2, FA, TA):
        bak = path + ".skive_orig"
        if os.path.exists(bak):
            with open(bak) as f:
                content = f.read()
            with open(path, "w") as f:
                f.write(content)
            print(f"  [revert] {os.path.relpath(path, VROOT)}")
        else:
            print(f"  [revert] no backup for {os.path.relpath(path, VROOT)}")
    return 0


CORE = os.path.join(VROOT, "v1", "engine", "core.py")
RUNNER_V2 = os.path.join(VROOT, "v1", "worker", "gpu", "model_runner.py")
FA = os.path.join(VROOT, "v1", "attention", "backends", "flash_attn.py")

# --- I: capture the decode query per layer for the value x attention metric --
# Kernel-free: module flag (read once from SKIVE_METRIC) + a single detach/store
# in forward. Off unless SKIVE_METRIC=value_attention, so zero cost otherwise.
# import os at the TOP of the file (Ruff E402: imports must be at top)
I_IMPORT_ANCHOR = "import copy\n"
I_IMPORT_INSERT = "import os  # SKIVE: query capture for value_attention metric\n"
# the capture flag (an assignment, not an import) just before the class
I_FLAG_ANCHOR = "class FlashAttentionBackend(AttentionBackend):\n"
I_FLAG_INSERT = (
    "_SKIVE_CAPTURE_Q = os.environ.get(\"SKIVE_METRIC\") in (\"value_attention\", \"h2o\", \"snapkv\")  # SKIVE:\n"
    "\n\n"
)
I_CAP_ANCHOR = (
    "        if output_scale is not None or output_block_scale is not None:\n"
    "            raise NotImplementedError(\n"
    "                \"fused output quantization is not yet supported for "
    "FlashAttentionImpl\"\n"
)
I_CAP_INSERT = (
    "        # SKIVE: capture current query for value x attention eviction scoring.\n"
    "        if _SKIVE_CAPTURE_Q:\n"
    "            layer._skive_q = query.detach()\n"
)

# --- J: sparse-gather -- skip evicted (null) blocks in the FA read (Phase 1) --
# When eviction is on, drop null-marked blocks from the attention gather list
# and shorten seqused_k so FlashAttention reads ONLY retained keys: fewer FLOPs,
# real working-set reduction, and no zero-key softmax dilution. Correct because
# eviction removes whole already-full blocks during decode (query_len==1), and
# logical length / RoPE positions are untouched. Toggle off with
# SKIVE_SPARSE_GATHER=0 to A/B against the null-block (read-zeros) behavior.
J_FLAG_ANCHOR = (
    '_SKIVE_CAPTURE_Q = os.environ.get("SKIVE_METRIC") in ("value_attention", "h2o", "snapkv")  # SKIVE:\n'
)
J_FLAG_INSERT = (
    '_SKIVE_SPARSE = os.environ.get("SKIVE_SPARSE_GATHER", "1") == "1"  # SKIVE: sparse-gather\n'
    "\n"
    "\n"
    "def _skive_can_compact(kv_cache_spec) -> bool:  # SKIVE: multi-group (gpt-oss)\n"
    '    """Only the full-attention KV group may be sparse-gathered. Sliding-window\n'
    "    groups (gpt-oss even layers) already hold vLLM's own null blocks for\n"
    "    out-of-window tokens and compacting them would shift the window; a\n"
    "    FullAttentionSpec carrying a sliding_window (non-hybrid allocator) is a\n"
    '    table shared with sliding-window layers, so it is skipped too."""\n'
    "    from vllm.v1.kv_cache_interface import SlidingWindowSpec\n"
    "\n"
    "    if isinstance(kv_cache_spec, SlidingWindowSpec):\n"
    "        return False\n"
    '    return getattr(kv_cache_spec, "sliding_window", None) is None\n'
)
J_BUILD_ANCHOR = (
    "        block_table_tensor = common_attn_metadata.block_table_tensor\n"
    "        slot_mapping = common_attn_metadata.slot_mapping\n"
    "        causal = common_attn_metadata.causal\n"
)
J_BUILD_INSERT = (
    "\n"
    "        # SKIVE: sparse-gather -- drop evicted (null) blocks from the read so\n"
    "        # FlashAttention skips them (fewer FLOPs + memory, no zero-key\n"
    "        # dilution). Pure-decode rows only; RoPE is unaffected (positions come\n"
    "        # from the model's positions tensor, not the gather list).\n"
    '        if (_SKIVE_SPARSE and getattr(self.cache_config, "kv_evict_enabled", False)\n'
    "                and _skive_can_compact(self.kv_cache_spec)):\n"
    f"            from {KVE}.compaction import compact_block_table_torch\n"
    "\n"
    "            _q_lens = query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]\n"
    "            block_table_tensor, seq_lens = compact_block_table_torch(\n"
    "                block_table_tensor, seq_lens, self.block_size, query_lens=_q_lens\n"
    "            )\n"
)

# --- K: the same two hooks (I + J) in the TRITON attention backend -------------
# Why: gpt-oss needs attention sinks. vLLM only serves sinks through
# FlashAttention-3/4 (Hopper / SM100 datacenter Blackwell) or FlashInfer's
# TRT-LLM kernels (SM100 only). On every other sink-capable GPU -- RTX PRO 6000
# Blackwell (SM120), L40S / L4 / RTX 4090 (Ada, SM89) -- the selector falls
# through to TRITON_ATTN, whose unified kernel takes `block_table` + `seqused_k`
# exactly like FlashAttention, so the sparse-gather compaction and the query
# capture port 1:1. Without these edits SKIVE on such GPUs would evict and
# reclaim (backend-independent) but never capture the query (value_attention
# silently falls back to vk_ratio) and never skip evicted blocks in the read.
TA = os.path.join(VROOT, "v1", "attention", "backends", "triton_attn.py")

K_IMPORT_ANCHOR = "from dataclasses import dataclass\nfrom typing import ClassVar\n"
K_IMPORT_INSERT = "import os  # SKIVE: query capture / sparse-gather flags\n"

K_FLAG_ANCHOR = (
    "class TritonAttentionMetadataBuilder("
    "AttentionMetadataBuilder[TritonAttentionMetadata]):\n"
)
K_FLAG_INSERT = (
    '_SKIVE_CAPTURE_Q = os.environ.get("SKIVE_METRIC") in ("value_attention", "h2o", "snapkv")  # SKIVE:\n'
    '_SKIVE_SPARSE = os.environ.get("SKIVE_SPARSE_GATHER", "1") == "1"  # SKIVE: sparse-gather\n'
    "\n"
    "\n"
    "def _skive_can_compact(kv_cache_spec) -> bool:  # SKIVE: multi-group (gpt-oss)\n"
    '    """Only the full-attention KV group may be sparse-gathered (see the\n'
    '    identical helper in flash_attn.py)."""\n'
    "    from vllm.v1.kv_cache_interface import SlidingWindowSpec\n"
    "\n"
    "    if isinstance(kv_cache_spec, SlidingWindowSpec):\n"
    "        return False\n"
    '    return getattr(kv_cache_spec, "sliding_window", None) is None\n'
    "\n"
    "\n"
)

K_BUILD_ANCHOR = (
    "        block_table_tensor = common_attn_metadata.block_table_tensor\n"
    "        slot_mapping = common_attn_metadata.slot_mapping\n"
)
K_BUILD_INSERT = (
    "\n"
    "        # SKIVE: sparse-gather -- drop evicted (null) blocks from the read so the\n"
    "        # Triton unified kernel skips them (same contract as FlashAttention:\n"
    "        # gather list + seqused_k). Pure-decode rows only; positions/RoPE are\n"
    "        # untouched. Skipped under full CUDA-graph decode capture, where the\n"
    "        # metadata tensors must stay the persistent buffers (enforce_eager or\n"
    "        # piecewise graphs are fine).\n"
    '        if (_SKIVE_SPARSE and getattr(self.vllm_config.cache_config, "kv_evict_enabled", False)\n'
    "                and not self.decode_cudagraph_enabled\n"
    "                and _skive_can_compact(self.kv_cache_spec)):\n"
    f"            from {KVE}.compaction import compact_block_table_torch\n"
    "\n"
    "            _n = common_attn_metadata.num_reqs\n"
    "            _q_lens = query_start_loc[1 : _n + 1] - query_start_loc[:_n]\n"
    "            block_table_tensor, seq_lens = compact_block_table_torch(\n"
    "                block_table_tensor, seq_lens, self.block_size, query_lens=_q_lens\n"
    "            )\n"
)

K_CAP_ANCHOR = (
    "        if output_block_scale is not None:\n"
    "            raise NotImplementedError(\n"
    '                "fused block_scale output quantization is not yet supported"\n'
    '                " for TritonAttentionImpl"\n'
)
K_CAP_INSERT = (
    "        # SKIVE: capture current query for value x attention eviction scoring.\n"
    "        if _SKIVE_CAPTURE_Q:\n"
    "            layer._skive_q = query.detach()\n"
)


# --- H: loud guard on the V2 model runner -----------------------------------
# Our hooks live in the V1 runner. Some architectures (e.g. Llama) default to
# the V2 runner, where eviction would silently no-op. Fail loudly instead.
H_ANCHOR = "        self.cache_config = vllm_config.cache_config\n"
H_INSERT = (
    "        # SKIVE: eviction is implemented only for the V1 model runner;\n"
    "        # fail loudly instead of silently no-op'ing on V2.\n"
    "        if getattr(self.cache_config, \"kv_evict_enabled\", False):\n"
    "            raise NotImplementedError(\n"
    "                \"SKIVE KV eviction requires the V1 model runner. Set \"\n"
    "                \"VLLM_USE_V2_MODEL_RUNNER=0 (this model defaulted to V2).\"\n"
    "            )\n"
)


def _reclaim_block(tag: str) -> str:
    """Scheduler-side reclaim, inserted BEFORE the relevant return so it runs
    every executed step. `tag` differentiates the step() vs batch-queue copy
    (distinct first line => correct per-edit idempotency)."""
    return (
        f"        # SKIVE: reclaim ({tag}) physical blocks evicted this step.\n"
        '        if getattr(self.vllm_config.cache_config, "kv_evict_enabled", False):\n'
        "            try:\n"
        f"                from {KVE}.integration import (\n"
        "                    _skive_pop_pending,\n"
        "                    skive_reclaim,\n"
        "                )\n"
        "\n"
        "                _skive_pending = self.model_executor.collective_rpc(\n"
        "                    _skive_pop_pending, single_value=True\n"
        "                )\n"
        "                if _skive_pending:\n"
        "                    sched = self.scheduler\n"
        "                    skive_reclaim(\n"
        "                        sched.kv_cache_manager,  # type: ignore[attr-defined]\n"
        "                        _skive_pending,\n"
        "                    )\n"
        "            except Exception as _skive_e:\n"
        '                logger.warning("SKIVE reclaim error: %s", _skive_e)\n'
        "\n"
    )


# F: step() -- inserted before its unique return line.
F_ANCHOR = (
    "        return engine_core_outputs, "
    "scheduler_output.total_num_scheduled_tokens > 0\n"
)
F_INSERT = _reclaim_block("step")

# G: step_with_batch_queue() -- inserted before its unique return line.
G_ANCHOR = "        return engine_core_outputs, model_executed\n"
G_INSERT = _reclaim_block("batched")


# Every edit, keyed by its letter, so `--only=K` (or `--only=IJK`) can apply a
# subset -- used to add the Triton hooks to an already-patched tree, which
# already carries A-J, without re-touching the other files.
EDITS = [
    ("A", lambda: apply_edit(CACHE, A_ANCHOR, A_INSERT, "A_cacheconfig_fields")),
    ("B", lambda: apply_edit(ARGS, B_ANCHOR, B_INSERT, "B_engineargs_fields")),
    ("C", lambda: apply_edit(ARGS, C_ANCHOR, C_INSERT, "C_cli_add_argument")),
    ("D", lambda: apply_edit(ARGS, D_ANCHOR, D_INSERT, "D_cacheconfig_ctor")),
    ("E", lambda: apply_edit(RUNNER, E_ANCHOR, E_INSERT, "E_worker_score_hook")),
    ("F", lambda: apply_edit(CORE, F_ANCHOR, F_INSERT, "F_reclaim_step", before=True)),
    ("G", lambda: apply_edit(CORE, G_ANCHOR, G_INSERT, "G_reclaim_batched",
                             before=True)),
    ("H", lambda: apply_edit(RUNNER_V2, H_ANCHOR, H_INSERT, "H_v2_guard")),
    ("I", lambda: apply_edit(FA, I_IMPORT_ANCHOR, I_IMPORT_INSERT, "I_capq_import")),
    ("I", lambda: apply_edit(FA, I_FLAG_ANCHOR, I_FLAG_INSERT, "I_capq_flag",
                             before=True)),
    ("I", lambda: apply_edit(FA, I_CAP_ANCHOR, I_CAP_INSERT, "I_capq_forward",
                             before=True)),
    ("J", lambda: apply_edit(FA, J_FLAG_ANCHOR, J_FLAG_INSERT, "J_sparse_flag")),
    ("J", lambda: apply_edit(FA, J_BUILD_ANCHOR, J_BUILD_INSERT, "J_sparse_build")),
    # K: Triton backend (gpt-oss on SM120 RTX PRO 6000 / Ada GPUs)
    ("K", lambda: apply_edit(TA, K_IMPORT_ANCHOR, K_IMPORT_INSERT, "K_triton_import",
                             before=True)),
    ("K", lambda: apply_edit(TA, K_FLAG_ANCHOR, K_FLAG_INSERT, "K_triton_flags",
                             before=True)),
    ("K", lambda: apply_edit(TA, K_BUILD_ANCHOR, K_BUILD_INSERT,
                             "K_triton_sparse_build")),
    ("K", lambda: apply_edit(TA, K_CAP_ANCHOR, K_CAP_INSERT, "K_triton_capq",
                             before=True)),
]


def main():
    if "--revert" in sys.argv:
        print(f"Reverting vLLM at {VROOT}")
        return revert()
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--only="):
            only = set(a[len("--only="):].upper())
    print(f"Patching vLLM at {VROOT}" + (f" (edits {''.join(sorted(only))})" if only else ""))
    changed = False
    for letter, fn in EDITS:
        if only is None or letter in only:
            changed |= fn()
    print("done." if changed else "nothing to do (already patched).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
