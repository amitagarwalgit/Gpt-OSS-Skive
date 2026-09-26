# AIME 2024 results (gpt-oss-20b, NVIDIA L40S 48 GB, vLLM 0.23.0 + SKIVE)

Each directory holds one JSON per configuration (metrics plus every generated answer), `TABLES.md` in the standard layout, and `aime24_answers.csv` with one row per generated sample (config, budget, problem, sample, question, reference, gold, answer, score, generated tokens).

| directory | what it is |
| --- | --- |
| `aime24/pass1_4samples/` | First pass, 4 samples per problem: FullKV eager and with CUDA graphs, vk_ratio and value_attention at 2048, value_attention at 4096, and the lever sweep that found the prompt-protection effect (`pp`), the local window (`l32`), eviction cadence (`e64`) and piecewise CUDA graphs (`cg`). |
| `aime24/pass2_8samples/` | Second pass, 8 samples per problem (240 concurrent requests): FullKV with graphs and with an FP8 cache; value_attention at 2048 and 4096 with the best settings (`base` = prompt protected, local 32, graphs) and the exploratory variants: trajectory scoring (`traj`), redundancy penalty (`red`), 64-token blocks (`blk64`), FP8 cache with eviction (`fp8`). |
| `aime24/new_run/` | Created by `scripts/run_aime24.sh` when you run it. |

Configuration names: `fullkv` (no eviction), `vk@N` / `va@N` (vk_ratio / value_attention with an N-token budget); the suffix after `+` lists the variant knobs. Every JSON records the exact knob values under `knobs`. `docs/GPTOSS_AIME24_EXPLAINED.md` walks through all of it.
