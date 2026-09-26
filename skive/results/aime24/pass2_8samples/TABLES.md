# SKIVE on openai/gpt-oss-20b -- accuracy and latency (attention backend TRITON_ATTN)

## aime24: accuracy and latency (all budgets)

Math reasoning, pass rate. FullKV acc 68.8, judge -. n=30 prompts, max_tokens=16384, reasoning=medium. dAcc / % deltas are vs FullKV (eager). evict rate = cache tokens evicted per generated token; evict blk/req = evicted blocks per request; KV saved = evicted / total tokens.

| Config | cross% | evict rate | evict blk/req | acc (dAcc) | judge | rouge2 | MiniLM | rep4g | ttft (s) | tpot (ms) | e2e (s) | wall (s) | med gen tok | batch tok/s | KV saved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FullKV cg | - | - | - | 68.8 | - | - | - | 0.035 | 1.2 | 87 | 550.3 | 788 | 62452 | 2384 | - |
| FullKV cg_fp8 | - | - | - | 71.7 (+2.9) | - | - | - | 0.040 | 5.2 (+323%) | 63 (-28%) | 406.6 (-26%) | 603 (-23%) | 54956 (-12%) | 3089 (+30%) | - |
| va @2048 base | 89% | 0.78 | 414 | 63.8 (-5.0) | - | - | - | 0.049 | 1.2 (+2%) | 73 (-15%) | 486.0 (-12%) | 656 (-17%) | 63470 (+2%) | 3085 (+29%) | 78% |
| va @2048 base_blk64 | 87% | 0.79 | 108 | 60.0 (-8.8) | - | - | - | 0.064 | 2.5 (+103%) | 58 (-34%) | 418.3 (-24%) | 572 (-27%) | 69754 (+12%) | 3653 (+53%) | 79% |
| va @2048 red | 89% | 0.79 | 437 | 64.6 (-4.2) | - | - | - | 0.057 | 1.3 (+6%) | 71 (-19%) | 499.8 (-9%) | 674 (-14%) | 73142 (+17%) | 3138 (+32%) | 79% |
| va @2048 traj | 90% | 0.79 | 430 | 63.3 (-5.4) | - | - | - | 0.044 | 1.3 (+11%) | 104 (+20%) | 557.0 (+1%) | 739 (-6%) | 72521 (+16%) | 2828 (+19%) | 79% |
| va @2048 traj_blk64 | 88% | 0.80 | 110 | 63.8 (-5.0) | - | - | - | 0.046 | 5.5 (+348%) | 81 (-7%) | 473.4 (-14%) | 644 (-18%) | 78417 (+26%) | 3294 (+38%) | 80% |
| va @2048 traj_red | 90% | 0.80 | 448 | 62.9 (-5.8) | - | - | - | 0.055 | 1.4 (+15%) | 97 (+12%) | 580.3 (+5%) | 754 (-4%) | 76702 (+23%) | 2859 (+20%) | 80% |
| va @4096 base | 64% | 0.56 | 272 | 72.1 (+3.3) | - | - | - | 0.026 | 1.2 (+1%) | 81 (-7%) | 506.9 (-8%) | 721 (-9%) | 58279 (-7%) | 2565 (+8%) | 56% |
| va @4096 base_fp8 | 65% | 0.61 | 333 | 61.2 (-7.5) | - | - | - | 0.056 | 2.4 (+94%) | 78 (-10%) | 522.8 (-5%) | 692 (-12%) | 69702 (+12%) | 3015 (+26%) | 61% |
| va @4096 traj | 63% | 0.58 | 292 | 70.4 (+1.7) | - | - | - | 0.038 | 8.2 (+576%) | 91 (+5%) | 584.2 (+6%) | 821 (+4%) | 62274 (-0%) | 2346 (-2%) | 58% |
| va @4096 traj_fp8 | 64% | 0.60 | 319 | 68.3 (-0.4) | - | - | - | 0.048 | 8.0 (+554%) | 86 (-1%) | 545.2 (-1%) | 739 (-6%) | 66910 (+7%) | 2751 (+15%) | 60% |

Accuracy cost -8.8 at 2048 (79% KV saved) shrinking to +3.3 at 4096. Preemptions: FullKV 0.0, eviction 0.0.

## vk_ratio vs value_attention

Representative budget per dataset (largest budget where both metrics were run).

| Dataset (budget) | acc vk / va | judge vk / va | rouge2 vk / va | batch tok/s vk / va | ttft (s) vk / va | max concurrency vk / va |
|---|---|---|---|---|---|---|

