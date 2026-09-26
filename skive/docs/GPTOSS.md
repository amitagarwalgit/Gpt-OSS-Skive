# Running SKIVE on gpt-oss-20b / gpt-oss-120b

This documents what had to change to make SKIVE work on OpenAI's gpt-oss models,
why, and how to run it with the team's inference recipe.

## Why gpt-oss needed changes (three model facts)

| gpt-oss fact | consequence for SKIVE | what we did |
| --- | --- | --- |
| **Alternating attention:** even layers are *sliding-window* (128 tokens), odd layers are *full attention* (`gpt_oss.py`: `sliding_window if layer_idx % 2 == 0`) | vLLM puts each attention type in its **own KV-cache group** with its **own block table and manager**. Our code assumed one group (`block_table[0]`, `managers[0]`) and scored every layer's cache with group-0 block ids — wrong ids on the sliding layers. | Resolve the true full-attention group (`_skive_full_group`), evict/reclaim **only** there, and score only that group's layers reading each layer's own `layer.kv_cache`. Pending frees now carry the group id. |
| **Learned attention sinks:** every layer has a per-head sink logit passed to FlashAttention-3 as `s_aux` | The model's softmax is `exp(l_t) / (exp(sink) + Σ exp(l))`. Scoring with a plain softmax ignores the sink. | `_softmax_with_sink`: the sink joins the denominator exactly like FA3. (Ranking within a head is unchanged — verified in tests — so eviction stays sound; cross-head weighting is now faithful.) |
| **Heavy GQA:** 64 query heads / 8 KV heads | Summing raw scores over heads lets a few large-scale heads dominate. | `SKIVE_HEAD_AGG=zmax`: TriAttention's normalize-then-aggregate (z-score per head, then max) — "keep a token if *any* head wants it". |

Two more guards:
- **Sparse-gather is now gated to the full-attention group** (`_skive_can_compact`).
  vLLM's own sliding-window manager already fills out-of-window slots with null
  blocks; compacting those would shift the window.
- **Hybrid kernel/manager block sizes** (`use_hybrid_blocks`) disable eviction
  with a one-time warning, because worker indices would not match manager
  indices at reclaim.

## Hardware / backend requirements

gpt-oss needs **attention sinks**, and vLLM 0.23 serves sinks through only
three backends. Which one is picked depends on the GPU, and SKIVE's two
attention hooks (query capture for `value_attention`/`h2o`/`snapkv`, and
sparse-gather) must exist in that backend:

| GPU | compute cap. | backend vLLM uses for gpt-oss | SKIVE hooks |
| --- | --- | --- | --- |
| H100 / H200 | 9.0 | `FLASH_ATTN` (FlashAttention-3, `s_aux` sinks) | yes (`flash_attn.py`) |
| **RTX PRO 6000 Blackwell 96 GB** | **12.0** | **`TRITON_ATTN`** (FA3 is Hopper-only, FA4 restricted to SM100, FlashInfer sinks need SM100 TRT-LLM) | **yes (`triton_attn.py`, edit K)** |
| L40S / L4 / RTX 4090 (Ada) | 8.9 | `TRITON_ATTN` | yes (`triton_attn.py`, edit K) |
| B200 (SM100 datacenter) | 10.0 | `FLASHINFER` (TRT-LLM) by default | no → force `SKIVE_ATTN_BACKEND=TRITON_ATTN` |

`run_gptoss.py` / `run_aime24_gptoss.py` therefore **force the backend**
explicitly (`SKIVE_ATTN_BACKEND=auto`: `FLASH_ATTN` on Hopper, `TRITON_ATTN`
everywhere else) so the run is deterministic and the log names the backend.
Both hook files carry the same logic: the Triton unified kernel takes
`block_table` + `seqused_k` exactly like FlashAttention, so the compaction is a
1:1 port. On the Triton path sparse-gather is skipped only under *full*
CUDA-graph decode capture (our recipe uses `enforce_eager=True`).

Other requirements:
- **V1 model runner:** `VLLM_USE_V2_MODEL_RUNNER=0` (the V2 runner raises loudly).
- **Memory:** 20b (MXFP4) ≈ 13 GB weights; 120b (MXFP4) ≈ 63 GB. A **96 GB RTX
  PRO 6000 runs gpt-oss-120b at `TP=1`** with ~20 GB left for KV at
  `gpu_memory_utilization=0.90`; an 80 GB H100 also fits it; 40–48 GB cards need
  `TP=2`.
- **MXFP4 expert kernels:** on SM120 the pip wheel uses the Marlin MXFP4 MoE
  kernels (the TRT-LLM/CUTLASS FlashInfer variants are SM90/SM100-only). If the
  log reports the Mxfp4 MoE backend as unsupported, install `triton_kernels`
  (from `triton-lang/triton`, `python/triton_kernels`) to enable the Triton MoE
  backend and re-run.

### Deploy to the RTX PRO 6000 box in one command

```bash
cd ~/Desktop/SKIVE_VLLM
bash stage6/deploy_and_test_gptoss.sh user@rtx6000-host          # ssh box
bash stage6/deploy_and_test_gptoss.sh local                      # if you are on the box
RUN_120B=1 bash stage6/deploy_and_test_gptoss.sh user@rtx6000-host   # also gpt-oss-120b
```

This syncs the working tree into the fork (applying the FA gate and the Triton
hooks), copies fork + tests to the box, installs the prebuilt vLLM 0.23.0
wheel, overlays the 7 patched files + `vllm/kv_evict`, runs the unit tests, and
runs the gpt-oss smoke: FullKV, then SKIVE with `vk_ratio` and
`value_attention`. Logs land in `~/skive/logs/` on the box; a summary of tok/s,
evicted blocks and the backend is printed at the end.

## Run it (matches the team's recipe)

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
python run_gptoss.py fullkv                                   # baseline
python run_gptoss.py skive                                    # SKIVE on
SKIVE_MODEL=/home/jovyan/basemodels/openai_gpt-oss-120b TP=2 python run_gptoss.py skive
```

`run_gptoss.py` uses exactly the screenshot settings — `dtype=bfloat16`,
`max_model_len=32768`, `gpu_memory_utilization=0.90`, `trust_remote_code=True`,
`enforce_eager=True`, `SamplingParams(temperature=0.6, top_p=0.95, max_tokens=2048)`,
`llm.chat(...)` — and adds the SKIVE flags. It prints tok/s, wall time and the
measured number of evicted blocks, and saves `gptoss_<mode>_result.json`.

### Recommended SKIVE settings for gpt-oss

| knob | value | why |
| --- | --- | --- |
| `kv_evict_budget` | 128 blocks (= 2048 tokens at block_size 16) | TriAttention's default budget for gpt-oss reasoning |
| `kv_evict_num_local_blocks` | **≥ 8** | 8 × 16 = 128 tokens = the sliding window, so the recent window is never touched |
| `kv_evict_num_sink_blocks` | 2 | attention-sink tokens |
| `SKIVE_METRIC` | `value_attention` | SKIVE's value-aware signal |
| `SKIVE_HEAD_AGG` | `zmax` | GQA normalize-then-max (TriAttention §4.3) |
| `SKIVE_SCORE_LAYER_STRIDE` | 4 | score every 4th full-attention layer (~4× cheaper) |
| `SKIVE_EVICT_EVERY` | 16 | deferred eviction cadence (paper prunes every 128 tokens) |
| `SKIVE_SCORER` | `torch` (default) / `triton` | `triton` = fused scoring kernel from `attention.py` (see below) |
| `SKIVE_PROTECT_PROMPT` | `1` for generation-heavy tasks (math / reasoning traces) | sink = prompt blocks + 1 per request, so the problem statement is never evicted. AIME24 @2048: 49.2 -> 65.0 pass rate at the same 77-79% KV saved. Leave unset for long-prompt tasks (the prompt is what must be evicted there). |
| `enable_prefix_caching` | False | reclaim assumes no shared-block ref-counts |

## Result tables in the slide format

`stage6/run_gptoss_campaign.sh` (run on the box; `CAMPAIGN=1` on the deploy
script starts it detached) produces `gptoss_results/TABLES.md` with:

1. **`<dataset>: accuracy and latency (all budgets)`** — one row per config
   (FullKV, FullKV cg, `vk @budget`, `va @budget`) with `cross%`, `acc (dAcc)`,
   `judge`, `rouge2`, `MiniLM`, `rep4g`, `ttft (s)`, `tpot (ms)`, `e2e (s)`,
   `wall (s)`, `med gen tok`, `batch tok/s`, `KV saved`; deltas vs FullKV.
   Datasets: `gov_report`, `multi_news` (LongBench, ROUGE-L), `hotpotqa`
   (LongBench multi-hop QA, token-F1), `longalpaca` (LongAlpaca-12k long items,
   token-F1), `gsm8k` (exact match), `aime24` (pass rate over NSAMP samples).
   `csv --dataset X` exports every generated answer (one row per sample) to
   `X_answers.csv`.
2. **`vk_ratio vs value_attention`** at a representative budget per dataset
   (acc, judge, rouge2, batch tok/s, ttft, max concurrency).
3. **Throughput** — req/s per concurrency cap, peak-vs-peak and at the shared
   high cap, peak KV usage, preemptions, and the CUDA-graph vs eager rows.

Definitions match the earlier Llama/Qwen slides: latencies are vLLM's own
per-request stats (`first_token_latency`, decode time / (n-1)); `cross%` is the
share of requests that evicted at least one block (measured on the worker);
`KV saved` = evicted tokens / total tokens; `judge` is the same gpt-oss model
grading each answer 1-5 against the reference (`JUDGE=0` to skip); `MiniLM` is
all-MiniLM-L6-v2 cosine similarity to the reference; `rep4g` is the repeated
4-gram fraction. gpt-oss outputs are scored on the harmony `final` channel only,
with `reasoning_effort=low` by default (`REASONING=medium|high` to change).
Scripts: `gptoss_campaign.py` (run / judge / tables) and `gptoss_throughput.py`.

## The fused kernel from `attention.py`

`kv_evict/fused_attention.py` contains the team's Triton kernel as provided
(`evict_fwd_kernel` / `triton_evict_attention_forward`, dense-cache attention +
per-token eviction loss + `argmin`). Two honest notes on it:

1. It computes `exp(score)` **without subtracting the running max**, so it can
   overflow float32 on large logits. Fine as a reference; not safe as-is on the
   hot path.
2. It has no attention-sink term, so it does not match gpt-oss's real softmax.

So SKIVE uses a second kernel in the same file, `skive_token_score_kernel`
(via `score_tokens_fused`): the same `p_t · mean|v_t|` score, but **two-pass
(max, then exp/sum) for stability** and **sink-aware**. `integration.py`
gathers a request's paged blocks into the dense `[B, H_kv, S, D]` layout the
kernel expects and applies it when `SKIVE_SCORER=triton`, falling back to the
torch path on any error. `tests/test_gptoss_scoring.py` checks it against the
torch reference on a GPU.

## What was verified where

- **Unit-tested on CPU (no vLLM/GPU):** selection planning/choice (exact match
  to the previous sorted-based logic); sink-aware softmax vs. an explicit
  reference; sink preserves within-head ranking; `zmax` == normalize-then-max;
  multi-group resolution picks the full-attention group; reclaim frees blocks in
  the group the worker evicted from (never the sliding-window group); the
  hybrid-block guard; layer filtering to the full group.
- **GPU test (skips without CUDA):** fused Triton scorer vs. torch reference.
- **Triton-backend hooks (edit K):** `tests/test_triton_hooks.py` applies the
  edit to a stub carrying vLLM's real anchor lines and to the fork's real
  `triton_attn.py`, checks idempotency and compilation, and exercises the
  patched code: decode rows with evicted blocks are compacted (prefill rows are
  not), the sliding-window group and full CUDA-graph capture are left untouched,
  and the query is captured only for query-dependent metrics.
- **Not yet run end-to-end on gpt-oss:** the multi-group path was implemented
  against vLLM 0.23.0's KV-group/coordinator/block-table code. The first
  `deploy_and_test_gptoss.sh` run on the RTX PRO 6000 (or an H100) is the
  integration test; treat its logs as the evidence.
