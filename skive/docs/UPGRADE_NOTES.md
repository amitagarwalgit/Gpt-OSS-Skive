# SKIVE Upgrade Notes — gpt-oss compatibility, fused scoring kernel, new selection module

**Date:** 2026-09-17
**Scope:** three requests — (1) integrate the team's `attention.py` Triton kernel,
(2) adopt the team's new `integration.py`, (3) make SKIVE run on **gpt-oss-20b /
gpt-oss-120b** with the team's inference recipe. Guided by the TriAttention paper
(arXiv 2604.04921), which evaluates on GPT-OSS-20B.

This document lists **every file added or changed, what changed, where, and why**.

---

## 1. Summary of the upgrade

| area | before | after |
| --- | --- | --- |
| KV-cache groups | assumed **one** group (`block_table[0]`, `managers[0]`); scored every layer with group-0 block ids | resolves the **true full-attention group** on hybrid models (gpt-oss has 2), evicts/reclaims/scores only there; refuses to evict if it can't resolve |
| attention sinks | ignored (plain softmax) | gpt-oss's learned per-head sinks join the softmax denominator, matching FA3 `s_aux` |
| GQA head aggregation | sum over heads only | `SKIVE_HEAD_AGG=sum|max|zmax` — `zmax` = TriAttention normalize-then-max |
| scoring kernel | torch einsum only | optional fused Triton scorer (`SKIVE_SCORER=triton`) from the team's `attention.py`, made numerically stable + sink-aware |
| eviction selection | Python list-comp + `sorted(key=lambda)` | vectorized NumPy `selection.py` (team's IMP-10), identical results |
| sparse-gather | applied to every attention-metadata builder | gated to the full-attention group only (sliding-window groups hold vLLM's own null blocks) |
| gpt-oss runner | none | `run_gptoss.py` (mirrors the team's recipe) + `overlay_skive.sh` (no-build install) + `GPTOSS.md` |

---

## 2. New files

### `kv_evict/selection.py` (new)
**Why:** the team's new `integration.py` imports `from .selection import choose_lowest, plan_eviction`, which did not exist.
- `plan_eviction(row, num_sink_blocks, num_local_blocks, kv_budget)` → `(real_indices, candidates_k, num_to_evict)`: finds the real (non-null) blocks, excludes the first `sink` and last `local` real blocks, caps the eviction count to the eligible candidates.
- `choose_lowest(candidates_k, scores, num_to_evict)` → the lowest-scoring candidates, **stable** on ties (exactly matches the previous `sorted(key=(score, k))`; `test_selection.py` checks this against a reference on random inputs).

### `kv_evict/fused_attention.py` (new — from the team's `attention.py` notebook)
Two kernels:
1. **`evict_fwd_kernel` / `triton_evict_attention_forward`** — the team's kernel, kept as provided (dense `[B, H_kv, S, D]` cache; computes attention output + per-token eviction loss `|p_t·v_t|`, returns `argmin`). Two review findings, documented in the file:
   - it takes `exp(score)` **without subtracting the running max** → can overflow fp32 on large logits;
   - it has **no attention-sink term** → does not match gpt-oss's real softmax.
2. **`skive_token_score_kernel` / `score_tokens_fused`** — the scoring-only kernel SKIVE actually uses: same `p_t · mean_d|v_t|` score, **two-pass (max, then exp/sum) for stability**, **sink-aware** (`l = exp(sink−m) + Σ exp(s−m)`), `SEQ_LEN_KV` as a *runtime* arg (no recompile per padded length). Returns normalized per-(batch, head, token) scores `[B, Hq, S]`; masked tokens are exactly 0.
   `integration.py` gathers a request's paged blocks into the dense layout the kernel expects and calls it when `SKIVE_SCORER=triton` (value_attention only), falling back to torch on any error. Importing the module without Triton is safe (`HAS_TRITON=False`).

### `stage6/run_gptoss.py` (new)
Mirrors the team's inference screenshot exactly — `dtype=bfloat16`, `max_model_len=32768`, `gpu_memory_utilization=0.90`, `tensor_parallel_size`, `trust_remote_code=True`, `enforce_eager=True`, `SamplingParams(temperature=0.6, top_p=0.95, max_tokens=2048)`, `llm.chat(...)`, `VLLM_USE_FLASHINFER_SAMPLER=0` — and adds the SKIVE flags. `fullkv` vs `skive` modes; prints tok/s, wall time, measured evicted blocks; saves `gptoss_<mode>_result.json`. Knobs via env: `SKIVE_MODEL`, `TP`, `BUDGET`, `SINK`, `LOCAL`, `MAXTOK`, `PROMPTS`.

### `stage6/overlay_skive.sh` (new)
Installs the **prebuilt** vLLM 0.23.0 wheel and overlays SKIVE's pure-Python files (6 patched vLLM files + `vllm/kv_evict/`) — no CUDA build. Resolves the installed package from `/tmp` so the repo's own `vllm/` tree never shadows it (a real bug hit earlier), and upgrades old `scipy` (its `numpy.Inf` import broke `transformers`).

### `stage6/finish_sync.sh` (new)
One idempotent command to finish the hand-off: sync the working tree into the `Skive-V` fork, apply the sparse-gather gate to the fork's `flash_attn.py` (asserts on exact anchors), compile-check, run the CPU tests.

### `GPTOSS.md` (new)
The gpt-oss guide: the three model facts and their fixes, hardware requirements, recommended settings, the fused-kernel notes, and what was verified where.

### Tests (new)
- `tests/test_selection.py` — planning/selection semantics; stable ties; random-input match to the old sorted logic.
- `tests/test_multigroup.py` — pure-Python stubs (no vLLM/GPU): full-group resolution prefers the pure full-attention group; falls back to a windowed `FullAttentionSpec`; resolves via `attn_groups` when `kv_cache_config` is absent; **refuses to evict** on an unresolvable multi-group model; layer filtering; reclaim routes to the evicted block's group (never the sliding-window group); legacy 2-tuples → group 0; hybrid-block guard (warns once).
- `tests/test_gptoss_scoring.py` — sink-aware softmax equals an explicit reference; the sink **does not change within-head ranking**; `zmax` == normalize-then-max with padding masked; fused Triton scorer matches the torch path (GPU-only, skips without CUDA).

---

## 3. Modified files

### `kv_evict/integration.py` — merged (team's version + gpt-oss support + baselines)
Base = the team's new version (selection-based eviction, `_SKIVE_LOG_EVICT` gating, `_skive_get_evicted_by_req`). Added:

**Multi KV-cache-group support (gpt-oss)**
- `_is_full_attention_spec(spec)` — duck-typed (class name + `sliding_window`/`attention_chunk_size` attrs), so it needs no vLLM import and is unit-testable.
- `_skive_kv_groups(runner)` — `[(gid, spec, layer_names)]` from `runner.kv_cache_config.kv_cache_groups`, else from `runner.attn_groups`.
- `_skive_full_group(runner)` — picks the pure full-attention group; falls back to a `FullAttentionSpec` that carries a window (non-hybrid allocator); on a multi-group model it cannot inspect, sets `_skive_multigroup_unresolved` so eviction is **refused instead of guessing group 0** (which on gpt-oss could be the sliding-window table). Cached per runner.
- `_skive_layers(runner)` — scorable attention layers restricted to that group.
- `_layer_kv(layer, runner, li)` — reads each layer's **own** cache (`layer.kv_cache`, set by vLLM's `bind_kv_cache`); the positional `kv_caches[li]` fallback is only used on single-group models.
- `_skive_full_caches(runner)` — the group's cache tensors (used by the static `vk_ratio` scorer).
- `_skive_block_table(runner)` — the group's worker block table; returns `None` (and warns once) if unresolved **or** if `use_hybrid_blocks` (kernel/manager block sizes differ → worker index ≠ manager index at reclaim).
- `evict_request_blocks`, `skive_post_step`, `_row_block_ids`, `debug_log_scores`, `_vk_score_cached`, `_batch_prescore` now use the resolved group instead of index 0.
- Pending frees are `(req_id, j, gid)` triples; `skive_reclaim` routes each to `managers[gid]` (still accepts legacy `(req_id, j)` pairs → group 0).

**Sink-aware, GQA-aware scoring**
- `_layer_sinks(layer)` — the layer's `impl.sinks` tensor (gpt-oss) or `None`.
- `_softmax_with_sink(logits, sinks)` — appends the per-head sink logit before the softmax and drops it after: `p_t = exp(l_t) / (exp(sink) + Σ exp(l))`, exactly FA3's `s_aux` semantics.
- `_agg_heads(x, valid, mode)` — `sum` (previous behaviour), `max`, or `zmax` (z-score each head over its valid tokens, then max over heads; padding masked). Selected by `SKIVE_HEAD_AGG`.
- `_batch_score_va` uses `bt.block_size` of the resolved group, the layer's own cache, the sink-aware softmax, the chosen head aggregation, and optionally the fused Triton scorer (`SKIVE_SCORER=triton`).

**Kept from the earlier comparison work:** `_QDEP_METRICS = (value_attention, h2o, snapkv, streaming)` and their scoring branches.

### `stage4/patch_vllm.py` — patcher kept in sync with the fork
- **I-flag insert** now `_SKIVE_CAPTURE_Q = os.environ.get("SKIVE_METRIC") in ("value_attention", "h2o", "snapkv")` (query capture for all query-dependent metrics).
- **J-flag anchor** updated to match; **J-flag insert** now also defines `_skive_can_compact(kv_cache_spec)`.
- **J-build insert** condition now `... and _skive_can_compact(self.kv_cache_spec)`.

### `Skive-V/vllm/v1/attention/backends/flash_attn.py` — sparse-gather gate (applied by `finish_sync.sh`)
- `_skive_can_compact(spec)`: returns `False` for `SlidingWindowSpec` and for any spec carrying a `sliding_window` (a table shared with sliding-window layers); the `build()` compaction only runs when it returns `True`.
- **Why:** vLLM's `SlidingWindowManager.remove_skipped_blocks` already fills out-of-window slots with null blocks; compacting those out would shift the window the kernel expects.

### `Skive-V/vllm/kv_evict/` — vendored copies of `integration.py`, `selection.py`, `fused_attention.py` (synced by `finish_sync.sh`).

---

## 4. New configuration knobs

| knob | values | default | meaning |
| --- | --- | --- | --- |
| `SKIVE_HEAD_AGG` | `sum` / `max` / `zmax` | `sum` | head aggregation; use `zmax` for GQA models like gpt-oss |
| `SKIVE_SCORER` | `torch` / `triton` | `torch` | fused Triton scoring kernel for `value_attention` |
| `SKIVE_METRIC` | + `h2o`, `snapkv`, `streaming` | `vk_ratio` | query capture now enabled for all query-dependent metrics |

Recommended gpt-oss settings (see `GPTOSS.md`): `kv_evict_budget=128` (2048 tokens at block 16), `kv_evict_num_sink_blocks=2`, **`kv_evict_num_local_blocks≥8`** (= the 128-token sliding window, never touched), `SKIVE_HEAD_AGG=zmax`, `SKIVE_SCORE_LAYER_STRIDE=4`, `SKIVE_EVICT_EVERY=16`, `enable_prefix_caching=False`, `VLLM_USE_V2_MODEL_RUNNER=0`.

---

## 5. Hard requirements discovered (not optional)

- **gpt-oss sinks need a sink-capable attention backend.** vLLM 0.23 has three: FlashAttention-3 (Hopper only), FlashInfer TRT-LLM (SM100 only), and the Triton unified backend (any GPU). Originally our capture and sparse-gather hooks lived only in `flash_attn.py`, which tied the full SKIVE path to an H100. **Update 2026-09-22 (RTX PRO 6000 96 GB, SM120):** the same two hooks are now applied to `triton_attn.py` (patcher edit K, §8), so the full path runs on the RTX PRO 6000, on L40S/L4/Ada cards, and on Hopper alike; the runners force the backend explicitly (`SKIVE_ATTN_BACKEND`).
- **V1 model runner** (`VLLM_USE_V2_MODEL_RUNNER=0`); the V2 runner raises loudly.
- Memory: gpt-oss-20b (MXFP4) ≈ 13 GB weights; 120b ≈ 63 GB → one 96 GB RTX PRO 6000 or one 80 GB H100 at `TP=1`; `TP=2` on 40–48 GB cards.

---

## 8. Addendum 2026-09-22 — RTX PRO 6000 (96 GB, SM120) support

**Finding.** On compute capability 12.0 vLLM rejects the FlashAttention backend for gpt-oss (`flash_attn_supports_sinks()` is true only for FA3/FA4, and FA4 is restricted to `major == 10`), FlashInfer's `supports_sink()` requires SM100 TRT-LLM kernels, so the selector falls through to **`TRITON_ATTN`**. SKIVE's eviction + reclaim + `vk_ratio` are backend-independent, but the query capture (needed by `value_attention`/`h2o`/`snapkv`) and the sparse-gather compaction lived only in `flash_attn.py`, so on this GPU `value_attention` would have silently fallen back to `vk_ratio` and evicted blocks would still have been read as zeros.

**Changes.**

| file | change |
| --- | --- |
| `stage4/patch_vllm.py` | new **edit K** (4 insertions in `v1/attention/backends/triton_attn.py`): `import os`; `_SKIVE_CAPTURE_Q` / `_SKIVE_SPARSE` flags + `_skive_can_compact` before the metadata builder; sparse-gather compaction in `TritonAttentionMetadataBuilder.build` (gated to the full-attention group and skipped under full CUDA-graph decode capture); `layer._skive_q = query.detach()` at the top of `TritonAttentionImpl.forward`. Also `--only=<letters>` to apply a subset, `SKIVE_NO_BACKUP=1` to skip `.skive_orig` files inside a git tree, and an `EDITS` table. |
| `stage6/finish_sync.sh` | step 2b applies edit K to the fork via `patch_vllm.py --only=K`; compiles `triton_attn.py`; copies `box_setup_and_test.sh`; runs the new test. |
| `stage6/overlay_skive.sh` | overlays **7** files (adds `triton_attn.py`) and asserts both hook files carry the hooks. |
| `stage6/run_gptoss.py`, `run_aime24_gptoss.py` | `SKIVE_ATTN_BACKEND` (auto: `FLASH_ATTN` on Hopper, `TRITON_ATTN` elsewhere) passed as `attention_backend=` so the backend carrying the hooks is chosen deterministically and logged; GPU name / capability printed; `MAXSEQS`, `RESULT_TAG`; default model id `openai/gpt-oss-20b`. |
| `stage6/box_setup_and_test.sh` (new) | runs **on** the box: overlay → unit tests → smoke (FullKV, SKIVE `vk_ratio`, SKIVE `value_attention`; `RUN_120B=1` adds gpt-oss-120b) with a tok/s + evicted + backend summary. |
| `stage6/deploy_and_test_gptoss.sh` | host-agnostic (`user@host` or `local`), tar-over-ssh copy with verification, ControlMaster so a password is asked once, env passthrough, custom remote base dir. |
| `tests/test_triton_hooks.py` (new) | applies edit K to a stub with the real anchors and to the fork's real file (idempotent, compiles); exercises compaction of decode rows, prefill/sliding-window/CUDA-graph/eviction-off guards, and query capture per metric. |

**Not changed:** `kv_evict/*` — the integration already reads `layer.kv_cache`, `layer.impl.sinks`, `layer.impl.scale`, `layer.impl.num_queries_per_kv` and `layer._skive_q`, all of which the Triton implementation exposes identically.

---

## 6. Verification status (honest)

- **Done:** careful read-through of every new/changed file (incl. the fused kernel's 28 positional args matching its call site); all tests written.
- **Not yet executed:** a tool-permission outage on the assistant's side refused every shell command and every write to the fork directory for the whole session, so nothing has been compiled, tested, synced, or committed yet. `bash stage6/finish_sync.sh` performs all of that in one step.
- **Not yet run end-to-end on gpt-oss:** the first `run_gptoss.py skive` on the H100 is the integration test. If the log shows `[SKIVE] ... unresolved`, eviction switched itself off by design — capture the log.

---

## 7. How to finish

```bash
cd ~/Desktop/SKIVE_VLLM
bash stage6/finish_sync.sh        # sync -> fork, apply gate, compile, CPU tests (needs numpy+torch)
cd ~/Desktop/Skive-V && git add -A && git commit -m "gpt-oss support: multi-KV-group eviction, sink-aware scoring, zmax GQA agg, fused Triton scorer, selection.py"
```

On the H100 gpt-oss box:
```bash
git clone <repo> skive && bash skive/benchmarks/skive/overlay_skive.sh skive
cd /tmp && export VLLM_USE_FLASHINFER_SAMPLER=0
python ~/skive/benchmarks/skive/run_gptoss.py fullkv
python ~/skive/benchmarks/skive/run_gptoss.py skive
SKIVE_MODEL=/home/jovyan/basemodels/openai_gpt-oss-120b TP=2 python ~/skive/benchmarks/skive/run_gptoss.py skive
```
