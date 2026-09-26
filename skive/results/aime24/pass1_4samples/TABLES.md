# SKIVE on openai/gpt-oss-20b -- accuracy and latency (attention backend TRITON_ATTN)

## aime24: accuracy and latency (all budgets)

Math reasoning, pass rate. FullKV acc 73.3, judge -. n=30 prompts, max_tokens=16384, reasoning=medium. dAcc / % deltas are vs FullKV (eager). evict rate = cache tokens evicted per generated token; evict blk/req = evicted blocks per request; KV saved = evicted / total tokens.

| Config | cross% | evict rate | evict blk/req | acc (dAcc) | judge | rouge2 | MiniLM | rep4g | ttft (s) | tpot (ms) | e2e (s) | wall (s) | med gen tok | batch tok/s | KV saved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FullKV | - | - | - | 73.3 | - | - | - | 0.036 | 2.8 | 50 | 303.0 | 529 | 24998 | 1626 | - |
| FullKV cgpw | - | - | - | 74.2 (+0.8) | - | - | - | 0.034 | 4.0 (+42%) | 44 (-13%) | 297.3 (-2%) | 515 (-3%) | 27229 (+9%) | 1756 (+8%) | - |
| vk @2048 | 89% | 0.00 | 0 | 47.5 (-25.8) | - | - | - | 0.076 | 0.7 (-74%) | 62 (+23%) | 431.2 (+42%) | 567 (+7%) | 42762 (+71%) | 2078 (+28%) | 81% |
| va @2048 | 84% | 0.00 | 0 | 49.2 (-24.2) | - | - | - | 0.061 | 0.7 (-74%) | 60 (+19%) | 436.5 (+44%) | 611 (+15%) | 35477 (+42%) | 1707 (+5%) | 79% |
| va @2048 pp | 88% | 0.00 | 0 | 65.0 (-8.3) | - | - | - | 0.043 | 0.9 (-68%) | 64 (+26%) | 434.9 (+44%) | 661 (+25%) | 29197 (+17%) | 1461 (-10%) | 77% |
| va @2048 pp_l32 | 85% | 0.00 | 0 | 66.7 (-6.7) | - | - | - | 0.049 | 0.8 (-72%) | 64 (+28%) | 440.3 (+45%) | 673 (+27%) | 32640 (+31%) | 1476 (-9%) | 78% |
| va @2048 pp_l32_e64 | 87% | 0.00 | 0 | 63.3 (-10.0) | - | - | - | 0.040 | 0.7 (-74%) | 64 (+28%) | 409.6 (+35%) | 632 (+20%) | 26159 (+5%) | 1559 (-4%) | 77% |
| va @2048 pp_l32_e64_cg | 88% | 0.00 | 0 | 64.2 (-9.2) | - | - | - | 0.048 | 4.2 (+50%) | 51 (+1%) | 306.4 (+1%) | 438 (-17%) | 32184 (+29%) | 2338 (+44%) | 78% |
| va @4096 | 60% | 0.00 | 0 | 68.3 (-5.0) | - | - | - | 0.052 | 0.7 (-74%) | 56 (+11%) | 379.2 (+25%) | 611 (+16%) | 28274 (+13%) | 1553 (-4%) | 58% |

Accuracy cost -25.8 at 2048 (81% KV saved) shrinking to -5.0 at 4096. Preemptions: FullKV 0.0, eviction 0.0.

## vk_ratio vs value_attention

Representative budget per dataset (largest budget where both metrics were run).

| Dataset (budget) | acc vk / va | judge vk / va | rouge2 vk / va | batch tok/s vk / va | ttft (s) vk / va | max concurrency vk / va |
|---|---|---|---|---|---|---|
| aime24 @2048 | 47.5 / 49.2 | - / - | - / - | 2078 / 1707 | 0.7 / 0.7 | 32.9 / 30.4 |

