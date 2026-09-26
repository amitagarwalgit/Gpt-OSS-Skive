"""AIME 2024 on gpt-oss with SKIVE (vk_ratio) vs FullKV -- TriAttention protocol.

Protocol (TriAttention, arXiv 2604.04921, Sec. 5.1): 30 AIME-2024 problems,
8 samples each, temperature 0.6, top_p 0.95, max generation 32k; report the
average pass rate. KV budget 2048 tokens (= 128 blocks at block_size 16).

Usage (after scripts/install.sh; launch from OUTSIDE the repo dir):
    export VLLM_USE_FLASHINFER_SAMPLER=0
    python run_aime24_gptoss.py fullkv
    python run_aime24_gptoss.py skive
Env: SKIVE_MODEL, TP, BUDGET (blocks, default 128), SINK (2), LOCAL (8),
     NSAMP (8), MAXTOK (24576), NPROB (30), OUT (json path),
     SKIVE_ATTN_BACKEND (auto: FLASH_ATTN on Hopper, TRITON_ATTN elsewhere --
     the two backends carrying SKIVE's hooks; RTX PRO 6000 / L40S -> TRITON_ATTN).
"""
import json
import os
import re
import sys
import time

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")  # collective_rpc(callable) via MP engine
os.environ.setdefault("SKIVE_METRIC", "vk_ratio")        # static proxy: no query capture needed
os.environ.setdefault("SKIVE_SPARSE_GATHER", "1")
os.environ.setdefault("SKIVE_EVICT_EVERY", "16")
os.environ.setdefault("SKIVE_LOG_EVICT", "1")

MODE = (sys.argv[1] if len(sys.argv) > 1 else "skive").lower()
MODEL = os.environ.get("SKIVE_MODEL", "openai/gpt-oss-20b")
TP = int(os.environ.get("TP", "1"))
BUDGET = int(os.environ.get("BUDGET", "128"))
SINK = int(os.environ.get("SINK", "2"))
LOCAL = int(os.environ.get("LOCAL", "8"))
NSAMP = int(os.environ.get("NSAMP", "8"))
MAXTOK = int(os.environ.get("MAXTOK", "24576"))
NPROB = int(os.environ.get("NPROB", "30"))
OUT = os.environ.get("OUT", f"aime24_gptoss_{MODE}.json")

PROMPT = ("Solve the following AIME problem. Reason step by step, then give the final "
          "integer answer inside \\boxed{{}}.\n\nProblem: {problem}")


def pick_attention_backend():
    """FLASH_ATTN on Hopper (FA3 sinks), TRITON_ATTN on every other GPU: gpt-oss
    needs sinks, and those are the two sink-capable backends that carry SKIVE's
    query-capture / sparse-gather hooks (see run_gptoss.py)."""
    forced = os.environ.get("SKIVE_ATTN_BACKEND", "auto").upper()
    if forced != "AUTO":
        return forced
    try:
        import torch
        major = torch.cuda.get_device_capability(0)[0]
    except Exception:  # noqa: BLE001
        return None
    return "FLASH_ATTN" if major == 9 else "TRITON_ATTN"


def load_aime24():
    from datasets import load_dataset
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    rows = [(str(r["Problem"]), str(r["Answer"]).strip()) for r in ds]
    return rows[:NPROB]


def extract_answer(text: str):
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    cand = m[-1] if m else None
    if cand is None:
        nums = re.findall(r"-?\d+", text)
        cand = nums[-1] if nums else None
    if cand is None:
        return None
    nums = re.findall(r"-?\d+", cand.replace(",", ""))
    return nums[-1] if nums else None


def main():
    from vllm import LLM, SamplingParams

    backend = pick_attention_backend()
    kw = dict(model=MODEL, dtype="bfloat16", max_model_len=32768,
              gpu_memory_utilization=0.90, tensor_parallel_size=TP,
              trust_remote_code=True, enforce_eager=True, enable_prefix_caching=False)
    if backend:
        kw["attention_backend"] = backend
    if MODE == "skive":
        kw.update(kv_evict_enabled=True, kv_evict_budget=BUDGET,
                  kv_evict_num_sink_blocks=SINK, kv_evict_num_local_blocks=LOCAL)
    print(f"[aime24] mode={MODE} model={MODEL} tp={TP} budget={BUDGET} sink={SINK} "
          f"local={LOCAL} nsamp={NSAMP} maxtok={MAXTOK} metric={os.environ['SKIVE_METRIC']} "
          f"attention_backend={backend}", flush=True)
    llm = LLM(**kw)
    rows = load_aime24()
    convs = [[{"role": "user", "content": PROMPT.format(problem=p)}] for p, _ in rows]
    params = SamplingParams(n=NSAMP, temperature=0.6, top_p=0.95, max_tokens=MAXTOK)

    llm.chat(convs[:1], SamplingParams(temperature=0.0, max_tokens=8))  # warm-up
    t0 = time.perf_counter()
    outs = llm.chat(convs, params)
    dt = time.perf_counter() - t0

    per_problem, out_tok = [], 0
    for (_, gold), o in zip(rows, outs):
        gold_n = re.findall(r"-?\d+", gold)
        gold_n = gold_n[-1] if gold_n else gold
        correct = 0
        for s in o.outputs:
            out_tok += len(s.token_ids)
            if extract_answer(s.text) == gold_n:
                correct += 1
        per_problem.append(correct / len(o.outputs))
    pass_rate = 100.0 * sum(per_problem) / len(per_problem)

    evicted = None
    if MODE == "skive":
        try:
            from vllm.kv_evict.integration import _skive_get_evicted_by_req
            res = llm.collective_rpc(_skive_get_evicted_by_req)
            evicted = sum(sum(d.values()) for d in res if isinstance(d, dict))
        except Exception as e:  # noqa: BLE001
            evicted = f"n/a ({type(e).__name__})"

    print("\n===== AIME 2024 result =====")
    print(f"mode={MODE}  problems={len(rows)}  samples/problem={NSAMP}")
    print(f"pass_rate={pass_rate:.1f}%  out_tok={out_tok}  wall={dt:.1f}s  "
          f"tok/s={out_tok / dt:.1f}  evicted_blocks={evicted}")
    json.dump(dict(mode=MODE, model=MODEL, attention_backend=backend, tp=TP, budget_blocks=BUDGET,
                   budget_tokens=BUDGET * 16, sink=SINK, local=LOCAL, nsamp=NSAMP,
                   maxtok=MAXTOK, problems=len(rows), pass_rate=round(pass_rate, 1),
                   per_problem=per_problem, out_tok=out_tok, wall_s=round(dt, 1),
                   tok_s=round(out_tok / dt, 1), evicted_blocks=evicted,
                   metric=os.environ["SKIVE_METRIC"]),
              open(OUT, "w"), indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
