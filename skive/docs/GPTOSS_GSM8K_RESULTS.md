# SKIVE on gpt-oss-20b: GSM8K, full test set, every lever on both metrics

Campaign run on 2026-09-28 on one NVIDIA L40S (48 GB). All 1,319 GSM8K test problems in every configuration; 53 configurations: FullKV three ways and, for each of the two SKIVE metrics and each of three budgets, the plain run plus seven levers. Every generated answer is in `skive/results/gsm8k/grid/gsm8k_answers.csv.gz` (one row per problem per configuration: question, reference, gold, answer, correct or not, generated tokens; 69,907 rows).

## Setup

| item | value |
| --- | --- |
| model / engine | openai/gpt-oss-20b on vLLM 0.23.0 with SKIVE; TRITON_ATTN attention backend, Marlin MXFP4 experts; max_model_len 32768, gpu_memory_utilization 0.90, prefix caching off, tensor parallel 1 |
| generation | greedy (temperature 0), reasoning_effort=medium, up to 2,048 output tokens; the answer is the last number in the harmony `final` channel, compared numerically with the reference |
| prompts | the GSM8K question plus "Please reason step by step and put your final answer after '####'."; about 100 to 200 tokens |
| traces | about 450 generated tokens on average under FullKV, so a request's cache is 550 to 650 tokens: a 1024 budget evicts on 8 to 10 percent of requests, 512 on about 42 percent, 256 on all of them |
| statistics | one problem is 0.08 points; FullKV eager vs graphs differ by 0.2 from run-to-run noise; treat differences under about 0.7 as noise |
| metrics | `vk` = vk_ratio (static, cached per block), `va` = value_attention (query-dependent) |
| levers (suffixes) | `plain`: fixed 2-block sink, nothing else; `pp`: prompt protected (sink = prompt blocks + 1); `traj`: 16-query trajectory window; `red`: redundancy penalty 0.3; `fp8`: FP8 KV cache; `e64`: evict every 64 steps (margin 16) instead of every 16; `blk64`: 64-token blocks. All rows use piecewise CUDA graphs and a 128-token local window (8 blocks at size 16; 4 blocks at the 256 budget so the window fits inside the budget; 2 blocks at size 64) |
| columns | dAcc vs FullKV with graphs; cross% = requests that evicted at least one block; evict rate = cache tokens evicted per generated token; KV saved = evicted tokens over all tokens; rep4g = repeated 4-gram fraction; ttft, tpot, e2e = vLLM per-request stats; wall = batch wall time for all 1,319 problems; gen = mean generated tokens |

<!-- pagebreak -->

## Verdict

- **At a 512-token budget SKIVE is lossless on GSM8K with value_attention and prompt protection**: 93.03 vs 93.10, with 42 percent of requests evicting and 26 percent of KV reclaimed. Adding the redundancy penalty and the trajectory window gives the best number of the whole grid, 93.63 (+0.5, at the edge of noise), at a decode-speed cost for trajectory (tpot 33 to 39 ms).
- **Prompt protection is again the single most important change**: +3.6 points at 512 (89.4 to 93.0) and +8 at 256 (69.3 to 77.2) for value_attention; +6.6 at 256 for vk_ratio. Without it the model loses the question and re-derives (rep4g doubles).
- **At a 256-token budget every request evicts and the budget is below the trace length**, so accuracy falls to 77 to 78 for most value_attention variants and traces grow from 453 to about 790 tokens. The lever that matters there is **slow eviction cadence** (`e64`): 83.4, plus 6 over the base row, with the same 67 to 73 percent of KV reclaimed and the shortest traces of the block. Evicting every 64 steps lets a request run up to 16 blocks over budget between rounds, so the working context of the derivation stays longer and the evictions happen in larger, better-informed batches.
- **value_attention beats vk_ratio wherever eviction is active**: at 512, 93.0 vs 92.0 (base rows) and 93.6 vs 92.3 (best rows); at 256, 77.2 vs 74.8 (base) and 83.4 vs 79.8 (best). At 1024 the two are indistinguishable because almost nothing is evicted.
- **Levers that cost accuracy on this task**: FP8 cache (about -0.8 for FullKV and -0.5 to -2.3 with eviction), 64-token blocks with vk_ratio at 512 (-2.6), and the trajectory window for vk_ratio (no gain, unlike for value_attention). Eviction cadence and blocks are within noise at 512 and 1024 for value_attention.
- **Latency**: with graphs on, SKIVE's per-token decode matches FullKV (tpot 31 to 33 ms vs 31) at 512 and 1024, and wall time is within 4 to 8 percent. At 256 wall time rises with the longer traces, not with the eviction itself. GSM8K never fills the cache at this concurrency, so there is no throughput gain to show here; the value is the memory reclaimed at zero accuracy cost at 512.
- **Recommendation for short-reasoning workloads like GSM8K**: value_attention, prompt protected, redundancy penalty on, budget 512 tokens (lossless, 26 percent reclaimed); if the budget must go below the trace length, add slow cadence (`SKIVE_EVICT_EVERY=64`). Compared with AIME (7,000-token traces, where 4096 was the lossless budget), the lossless budget scales with the trace length, roughly one budget of the mean trace.

<!-- pagebreak -->

## value_attention: accuracy and latency (all budgets, all levers)

FullKV acc 93.10 (graphs), 92.87 (eager), 92.27 (FP8). Deltas vs FullKV with graphs.

| Config | cross% | evict rate | acc (dAcc) | rep4g | ttft (s) | tpot (ms) | e2e (s) | wall (s) | gen tok | batch tok/s | KV saved |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FullKV cg | - | - | 93.10 | 0.023 | 27.5 | 31 | 41.4 | 104 | 453 | 5734 | - |
| FullKV eager | - | - | 92.87 (-0.23) | 0.020 | 35.1 | 42 | 53.5 | 153 | 443 | 3820 | - |
| FullKV cg fp8 | - | - | 92.27 (-0.83) | 0.023 | 27.0 | 29 | 40.7 | 104 | 475 | 6050 | - |
| va @1024 plain | 9% | 0.12 | 92.57 (-0.53) | 0.030 | 27.6 | 33 | 42.5 | 111 | 462 | 5503 | 9% |
| va @1024 pp | 8% | 0.10 | 93.56 (+0.45) | 0.022 | 27.7 | 33 | 42.4 | 108 | 454 | 5527 | 8% |
| va @1024 pp traj | 8% | 0.11 | 93.03 (-0.08) | 0.023 | 32.3 | 38 | 49.3 | 123 | 455 | 4866 | 8% |
| va @1024 pp red | 9% | 0.11 | 92.80 (-0.30) | 0.020 | 28.0 | 33 | 42.9 | 109 | 462 | 5562 | 9% |
| va @1024 pp traj red | 8% | 0.11 | 93.18 (+0.08) | 0.021 | 31.9 | 38 | 48.9 | 123 | 456 | 4901 | 8% |
| va @1024 pp fp8 | 9% | 0.12 | 92.04 (-1.06) | 0.022 | 28.1 | 31 | 42.4 | 110 | 471 | 5647 | 9% |
| va @1024 pp e64 | 8% | 0.10 | 92.87 (-0.23) | 0.021 | 27.3 | 32 | 41.6 | 105 | 452 | 5671 | 7% |
| va @1024 pp blk64 | 8% | 0.11 | 92.19 (-0.91) | 0.022 | 28.8 | 33 | 43.3 | 110 | 453 | 5429 | 8% |
| va @512 plain | 42% | 0.39 | 89.39 (-3.71) | 0.051 | 29.5 | 33 | 45.6 | 116 | 499 | 5649 | 31% |
| va @512 pp | 40% | 0.33 | 93.03 (-0.08) | 0.020 | 27.7 | 33 | 42.4 | 108 | 455 | 5541 | 26% |
| va @512 pp traj | 42% | 0.33 | 93.18 (+0.08) | 0.022 | 32.5 | 39 | 49.8 | 125 | 457 | 4806 | 26% |
| va @512 pp red | 42% | 0.33 | 93.25 (+0.15) | 0.019 | 28.2 | 33 | 42.9 | 108 | 455 | 5547 | 25% |
| va @512 pp traj red | 42% | 0.33 | 93.63 (+0.53) | 0.020 | 32.8 | 39 | 49.9 | 123 | 452 | 4865 | 25% |
| va @512 pp fp8 | 44% | 0.34 | 92.57 (-0.53) | 0.020 | 28.5 | 31 | 42.6 | 108 | 463 | 5656 | 26% |
| va @512 pp e64 | 38% | 0.31 | 92.49 (-0.61) | 0.020 | 27.4 | 32 | 41.6 | 105 | 454 | 5716 | 24% |
| va @512 pp blk64 | 42% | 0.38 | 92.49 (-0.61) | 0.024 | 29.4 | 32 | 44.4 | 112 | 473 | 5555 | 29% |

<!-- pagebreak -->

### value_attention, 256-token budget (below the trace length)

| Config | cross% | evict rate | acc (dAcc) | rep4g | ttft (s) | tpot (ms) | e2e (s) | wall (s) | gen tok | batch tok/s | KV saved |
|---|---|---|---|---|---|---|---|---|---|---|---|
| va @256 plain | 99% | 0.82 | 69.29 (-23.81) | 0.115 | 35.9 | 32 | 56.4 | 137 | 657 | 6325 | 68% |
| va @256 pp | 99% | 0.86 | 77.18 (-15.92) | 0.049 | 55.5 | 41 | 88.5 | 206 | 801 | 5130 | 73% |
| va @256 pp traj | 99% | 0.86 | 77.26 (-15.85) | 0.054 | 56.1 | 45 | 92.1 | 230 | 786 | 4498 | 73% |
| va @256 pp red | 99% | 0.85 | 78.32 (-14.78) | 0.050 | 44.6 | 37 | 72.7 | 182 | 757 | 5484 | 72% |
| va @256 pp traj red | 99% | 0.85 | 77.41 (-15.69) | 0.054 | 54.1 | 45 | 88.9 | 226 | 759 | 4420 | 72% |
| va @256 pp fp8 | 100% | 0.86 | 77.86 (-15.24) | 0.054 | 46.8 | 35 | 74.7 | 184 | 787 | 5647 | 73% |
| va @256 pp e64 | 96% | 0.80 | 83.40 (-9.70) | 0.034 | 41.7 | 33 | 64.6 | 148 | 701 | 6265 | 67% |

<!-- pagebreak -->

## vk_ratio: accuracy and latency (all budgets, all levers)

Deltas vs FullKV with graphs (93.10).

| Config | cross% | evict rate | acc (dAcc) | rep4g | ttft (s) | tpot (ms) | e2e (s) | wall (s) | gen tok | batch tok/s | KV saved |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vk @1024 plain | 8% | 0.11 | 93.18 (+0.08) | 0.020 | 27.0 | 32 | 41.4 | 106 | 457 | 5691 | 8% |
| vk @1024 pp | 8% | 0.11 | 92.57 (-0.53) | 0.022 | 27.0 | 32 | 41.4 | 106 | 459 | 5680 | 9% |
| vk @1024 pp traj | 8% | 0.11 | 92.87 (-0.23) | 0.019 | 27.4 | 32 | 41.8 | 106 | 456 | 5674 | 8% |
| vk @1024 pp red | 10% | 0.13 | 92.72 (-0.38) | 0.024 | 27.6 | 32 | 42.5 | 109 | 472 | 5700 | 10% |
| vk @1024 pp traj red | 9% | 0.12 | 92.80 (-0.30) | 0.022 | 27.4 | 32 | 42.0 | 106 | 461 | 5716 | 9% |
| vk @1024 pp fp8 | 9% | 0.11 | 92.12 (-0.99) | 0.019 | 26.3 | 30 | 40.2 | 103 | 473 | 6055 | 9% |
| vk @1024 pp e64 | 8% | 0.11 | 92.72 (-0.38) | 0.023 | 26.9 | 32 | 41.3 | 106 | 459 | 5728 | 8% |
| vk @1024 pp blk64 | 9% | 0.12 | 92.95 (-0.15) | 0.019 | 27.1 | 32 | 41.6 | 106 | 460 | 5734 | 9% |
| vk @512 plain | 41% | 0.37 | 91.66 (-1.44) | 0.028 | 28.1 | 32 | 43.1 | 110 | 480 | 5726 | 28% |
| vk @512 pp | 42% | 0.37 | 91.96 (-1.14) | 0.019 | 28.1 | 32 | 43.3 | 110 | 485 | 5797 | 29% |
| vk @512 pp traj | 41% | 0.38 | 91.96 (-1.14) | 0.024 | 28.3 | 32 | 43.6 | 111 | 493 | 5851 | 30% |
| vk @512 pp red | 43% | 0.36 | 92.34 (-0.76) | 0.021 | 28.0 | 32 | 43.0 | 108 | 479 | 5839 | 28% |
| vk @512 pp traj red | 42% | 0.38 | 91.66 (-1.44) | 0.022 | 28.5 | 32 | 43.8 | 111 | 490 | 5813 | 29% |
| vk @512 pp fp8 | 44% | 0.39 | 90.83 (-2.27) | 0.023 | 27.3 | 30 | 42.0 | 107 | 505 | 6236 | 31% |
| vk @512 pp e64 | 37% | 0.37 | 91.28 (-1.82) | 0.021 | 28.1 | 31 | 43.1 | 108 | 493 | 6017 | 28% |
| vk @512 pp blk64 | 42% | 0.44 | 90.52 (-2.58) | 0.024 | 29.3 | 31 | 45.5 | 115 | 528 | 6068 | 35% |

<!-- pagebreak -->

### vk_ratio, 256-token budget (below the trace length)

| Config | cross% | evict rate | acc (dAcc) | rep4g | ttft (s) | tpot (ms) | e2e (s) | wall (s) | gen tok | batch tok/s | KV saved |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vk @256 plain | 99% | 0.84 | 68.16 (-24.94) | 0.095 | 39.5 | 31 | 61.3 | 143 | 728 | 6698 | 70% |
| vk @256 pp | 99% | 0.87 | 74.75 (-18.35) | 0.058 | 51.3 | 33 | 81.5 | 188 | 904 | 6336 | 76% |
| vk @256 pp traj | 99% | 0.88 | 72.63 (-20.47) | 0.058 | 52.4 | 33 | 83.6 | 194 | 930 | 6305 | 76% |
| vk @256 pp red | 99% | 0.88 | 73.54 (-19.56) | 0.058 | 52.5 | 33 | 83.6 | 193 | 927 | 6334 | 76% |
| vk @256 pp traj red | 99% | 0.87 | 73.09 (-20.02) | 0.059 | 50.7 | 33 | 81.0 | 189 | 910 | 6367 | 76% |
| vk @256 pp fp8 | 99% | 0.88 | 73.09 (-20.02) | 0.059 | 50.4 | 32 | 80.1 | 186 | 921 | 6518 | 76% |
| vk @256 pp e64 | 96% | 0.82 | 79.76 (-13.34) | 0.035 | 42.4 | 30 | 65.8 | 149 | 797 | 7073 | 70% |

## vk_ratio vs value_attention, best row per budget

| budget | best vk (config) | best va (config) | base vk (pp) | base va (pp) | KV saved (pp) |
|---|---|---|---|---|---|
| 1024 | 93.18 (plain) | 93.56 (pp) | 92.57 | 93.56 | 8 to 9% |
| 512 | 92.34 (pp red) | 93.63 (pp traj red) | 91.96 | 93.03 | 26 to 29% |
| 256 | 79.76 (pp e64) | 83.40 (pp e64) | 74.75 | 77.18 | 73 to 76% |

## How this compares with AIME 2024

Same model, same levers, different trace length. On AIME (7,000-token traces) the lossless budget was 4096 tokens and prompt protection recovered 16 points at 2048; on GSM8K (450-token traces) the lossless budget is 512 and prompt protection recovers 4 to 8 points. In both, value_attention beats vk_ratio wherever eviction is active, FP8 with eviction costs a little accuracy, and 64-token blocks trade accuracy for eviction count. The new observation here is the cadence lever: when the budget is below the trace length, evicting every 64 steps instead of 16 is worth 5 to 6 points on both metrics, which did not show on AIME because there the budget was above most of the trace.
