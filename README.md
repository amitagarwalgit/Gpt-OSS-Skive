# Gpt-OSS-Skive: vLLM 0.23.0 with SKIVE KV-cache eviction and gpt-oss support

This is a complete fork of [vLLM](https://github.com/vllm-project/vllm) at v0.23.0 with **SKIVE**, a value-aware block-wise KV-cache eviction system, integrated into the engine, plus everything needed to run it on **openai/gpt-oss-20b** and reproduce the **AIME 2024** results below. The upstream README is kept as `README.vllm.md`; the upstream license (Apache 2.0) applies to the whole tree.

SKIVE keeps every request's KV cache under a fixed block budget while it decodes: it scores blocks of past tokens by how much they matter to the model's attention, evicts the least important ones, and hands the memory back to vLLM's block pool, so more requests fit on the same GPU.

## What is in the fork

| where | what |
| --- | --- |
| `vllm/kv_evict/` | the SKIVE package: block scoring (`vk_ratio`, `value_attention`, baselines), selection, sparse-gather compaction, eviction and reclaim, the gpt-oss integration (multi-KV-group resolution, sink-aware softmax, grouped-query head aggregation, prompt protection, trajectory scoring, redundancy penalty, FP8 cache decode), and a fused Triton scorer |
| 7 patched vLLM files | `vllm/config/cache.py`, `vllm/engine/arg_utils.py` (four engine arguments), `vllm/v1/worker/gpu_model_runner.py` (post-step eviction hook), `vllm/v1/engine/core.py` (block reclaim), `vllm/v1/attention/backends/flash_attn.py` and `triton_attn.py` (query capture and sparse-gather), `vllm/v1/worker/gpu/model_runner.py` (V2-runner guard). Every insertion is marked `SKIVE:`; `skive/patch_vllm.py` re-applies all 13 of them to a pristine v0.23.0 tree and `skive/patches/` holds the diffs |
| `skive/` | installer, run scripts, the AIME 2024 harness, unit tests, docs and every measured result |

## Run AIME 2024 in three commands

Needs Linux, one NVIDIA GPU with 24 GB or more (measured on an L40S 48 GB), Python 3.10 to 3.12, network access for the wheel and the model.

```bash
bash skive/install.sh                    # venv + official vLLM 0.23.0 wheel + this fork's SKIVE files + unit tests (~10 min)
source .venv/bin/activate
bash skive/run_aime24.sh                 # 30 problems x 8 samples, FullKV vs SKIVE (~1.5 h)
```

`NPROB=5 NSAMP=2 bash skive/run_aime24.sh` is the ten-minute version. The script prints the results table and writes `skive/results/aime24/new_run/TABLES.md` and `aime24_answers.csv` (one row per generated sample). `EXTRA=1` adds the exploratory rows. `bash skive/run_smoke.sh` is a three-minute check that eviction fires on this GPU.

The installer takes the fast path: the official wheel supplies the compiled kernels and this tree supplies the Python. SKIVE is pure Python, so the result is exactly this fork. To build the whole fork from source instead (CUDA toolkit, about an hour), follow `README.vllm.md` (`pip install -e .`) and run the same scripts.

## Results: AIME 2024, gpt-oss-20b, NVIDIA L40S 48 GB

All 30 problems, 8 samples each (240 concurrent requests), medium reasoning effort, up to 16,384 output tokens per sample, piecewise CUDA graphs. Pass rate is the fraction of samples with the correct boxed integer; one sample is 0.4 points, so differences under about 2.5 points are noise. "evict rate" is cache tokens evicted per generated token; "KV saved" is evicted tokens over all tokens. SKIVE rows use value_attention with the prompt protected and a 32-block local window.

| Config | acc | evict rate | KV saved | tpot (ms) | wall (s) | batch tok/s | max concurrency |
|---|---|---|---|---|---|---|---|
| FullKV | 68.8 | - | - | 87 | 788 | 2384 | 32.9 |
| FullKV, FP8 KV cache | 71.7 | - | - | 63 | 603 | 3089 | 65.0 |
| SKIVE, budget 4096 tokens | **72.1** | 0.56 | **56%** | 81 | 721 | 2565 | 32.9 |
| SKIVE, budget 2048 tokens | 63.8 | 0.78 | **78%** | 73 | 656 | 3085 | 32.9 |
| SKIVE, 2048, 64-token blocks | 60.0 | 0.79 | 79% | 58 | 572 | 3653 | 32.9 |

At a 4096-token budget SKIVE matches FullKV while reclaiming 56 percent of the cache; at 2048 it gives up 5 points for 78 percent reclaimed and, because 240 concurrent traces put FullKV under memory pressure, finishes 17 percent sooner. Two changes made this possible and are on by default: **protecting the prompt** (without it the problem statement itself was evicted mid-derivation and the 2048 budget scored 49.2) and **piecewise CUDA graphs** (without them eviction paid a decode-speed penalty). Every variant tried, including the ones that did not help, is in `skive/results/aime24/`; `skive/docs/GPTOSS_AIME24_EXPLAINED.md` (PDF in `skive/docs/pdf/`) explains each change and result.

## Using SKIVE in your own vLLM script

```python
from vllm import LLM, SamplingParams
llm = LLM(
    model="openai/gpt-oss-20b", dtype="bfloat16", max_model_len=32768,
    enable_prefix_caching=False,           # reclaim assumes no shared-block reference counts
    attention_backend="TRITON_ATTN",       # the sink-capable backend carrying SKIVE's hooks on non-Hopper GPUs
    kv_evict_enabled=True,
    kv_evict_budget=256,                   # blocks kept per request (256 x 16 = 4096 tokens)
    kv_evict_num_sink_blocks=2,
    kv_evict_num_local_blocks=32,          # never evict the last 512 tokens
)
```

with, before the engine starts:

```bash
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1
export SKIVE_METRIC=value_attention SKIVE_PROTECT_PROMPT=1 SKIVE_HEAD_AGG=zmax SKIVE_SCORE_LAYER_STRIDE=4
```

Knobs: `SKIVE_METRIC` (`vk_ratio`, `value_attention`, `h2o`, `snapkv`, `streaming`), `SKIVE_PROTECT_PROMPT` (`1` for reasoning and math, unset for long-document tasks), `SKIVE_HEAD_AGG` (`sum`, `max`, `zmax`), `SKIVE_SCORE_LAYER_STRIDE`, `SKIVE_EVICT_EVERY` / `SKIVE_EVICT_MARGIN`, `SKIVE_SPARSE_GATHER`, `SKIVE_QHIST` (trajectory window, off), `SKIVE_REDUNDANCY` (off). For graphs use `enforce_eager=False` with `CompilationConfig(cudagraph_mode=CUDAGraphMode.PIECEWISE)`; the harness does this with `CUDAGRAPH=piecewise`. Run Python from outside the repository root, otherwise `import vllm` resolves to this source tree instead of the installed package.

## What gpt-oss needed

gpt-oss alternates sliding-window and full-attention layers, so vLLM keeps two KV-cache groups with separate block tables; it has learned attention sinks in every layer; it runs 64 query heads on 8 KV heads; and on GPUs without FlashAttention-3 vLLM serves it through the Triton attention backend. SKIVE resolves the full-attention group and evicts only there, folds the sink into its softmax, aggregates heads with a z-score-then-max, and carries its hooks in the Triton backend as well as FlashAttention. `skive/docs/GPTOSS.md` and `skive/docs/GPTOSS_EXPLAINER.md` cover this in detail; `skive/docs/UPGRADE_NOTES.md` lists every file changed.

## Layout of `skive/`

```
install.sh              venv, vLLM 0.23.0 wheel, overlay from this tree, verification, unit tests
run_aime24.sh           the AIME 2024 evaluation (FullKV vs SKIVE)
run_smoke.sh            three-minute check that eviction works on this GPU
gptoss_campaign.py      the harness behind run_aime24.sh (run / csv / tables)
run_gptoss.py           one gpt-oss run with the inference recipe (fullkv | skive)
run_aime24_gptoss.py    the TriAttention paper protocol variant (8 samples, 32k tokens)
patch_vllm.py, patches/ the 13 hooks as reproducible anchored insertions, and their diffs
tests/                  SKIVE unit tests (run by the installer and by CI)
docs/                   write-ups and PDFs
results/aime24/         every measured AIME run: JSON per config, answers CSV, TABLES.md
```

CI (`.github/workflows/skive-tests.yml`) compiles the patched files and runs the unit tests on every push.
