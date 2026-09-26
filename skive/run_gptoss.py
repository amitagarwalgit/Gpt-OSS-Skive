"""SKIVE on gpt-oss-20b / gpt-oss-120b — mirrors the team's inference recipe.

Usage (after scripts/install.sh; launch from OUTSIDE the repo dir, e.g. /tmp):
    export VLLM_USE_FLASHINFER_SAMPLER=0            # team's setting
    python run_gptoss.py fullkv                      # baseline
    python run_gptoss.py skive                       # SKIVE eviction on
    SKIVE_MODEL=openai/gpt-oss-120b python run_gptoss.py skive     # 96 GB GPU: TP=1 is enough

Env knobs (defaults chosen for gpt-oss, see GPTOSS.md):
    SKIVE_MODEL      model path/id (default openai/gpt-oss-20b)
    TP               tensor_parallel_size (default 1; 120b fits one 96 GB RTX PRO 6000
                     or one 80 GB H100 in MXFP4; use 2 on 40-48 GB cards)
    BUDGET           KV blocks kept per sequence (default 128 -> 2048 tok @ bs16)
    SINK / LOCAL     protected leading / trailing blocks (default 2 / 8; LOCAL>=8
                     keeps the whole 128-token sliding window at block_size 16)
    MAXTOK           max_tokens (default 2048, as in the team's script)
    MAXLEN / GPUMEM  max_model_len (32768) / gpu_memory_utilization (0.90)
    MAXSEQS          max_num_seqs (vLLM default if unset)
    PROMPTS          path to a .txt/.jsonl of prompts (default: built-in set)
    SKIVE_ATTN_BACKEND  auto (default) | FLASH_ATTN | TRITON_ATTN | FLASHINFER
                     auto = FLASH_ATTN on Hopper (FA3 sinks), TRITON_ATTN elsewhere
                     (RTX PRO 6000 / SM120, L40S / L4 / Ada) -- the two backends
                     that carry SKIVE's query-capture + sparse-gather hooks.
"""
import json
import os
import sys
import time

# --- SKIVE / vLLM environment (must be set before importing vllm) ---------------
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")   # SKIVE hooks live in the V1 runner
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # team's gpt-oss setting
# collective_rpc(callable) to the multiprocess EngineCore needs cloudpickle, which
# vLLM's msgpack encoder only allows with this flag (else TypeError: not serializable)
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("SKIVE_METRIC", "value_attention")   # SKIVE value-aware scoring
os.environ.setdefault("SKIVE_HEAD_AGG", "zmax")            # TriAttention GQA aggregation
os.environ.setdefault("SKIVE_SCORE_LAYER_STRIDE", "4")     # score every 4th full layer
os.environ.setdefault("SKIVE_SPARSE_GATHER", "1")          # skip evicted blocks in the read
os.environ.setdefault("SKIVE_EVICT_EVERY", "16")           # deferred eviction cadence
os.environ.setdefault("SKIVE_LOG_EVICT", "1")              # print evict/reclaim counts

MODE = (sys.argv[1] if len(sys.argv) > 1 else "skive").lower()   # fullkv | skive
MODEL = os.environ.get("SKIVE_MODEL", "openai/gpt-oss-20b")
TP = int(os.environ.get("TP", "1"))
BUDGET = int(os.environ.get("BUDGET", "128"))
SINK = int(os.environ.get("SINK", "2"))
LOCAL = int(os.environ.get("LOCAL", "8"))
MAXTOK = int(os.environ.get("MAXTOK", "2048"))

_LONG_DOC = " ".join(
    f"Section {i}: The committee reviewed procurement item {i}, noting a unit cost of "
    f"{100 + 7 * i} dollars, a lead time of {3 + i % 9} weeks and a supplier rating of "
    f"{(i * 37) % 100} out of 100; the recommendation was to {'renew' if i % 3 else 'retire'} "
    f"the contract and to revisit the decision in fiscal year {2027 + i % 4}."
    for i in range(1, 121))   # ~4,000 tokens: exceeds the default 2048-token budget, so eviction fires

DEFAULT_PROMPTS = [
    "Explain KV-cache eviction in two sentences.",
    "A train leaves city A at 60 km/h and another leaves city B, 300 km away, at "
    "90 km/h toward it. When and where do they meet? Reason step by step.",
    "Write a Python function that returns the k most frequent words in a text, "
    "then explain its time complexity.",
    "Summarize the trade-offs between sliding-window attention and full attention "
    "for long-context reasoning.",
    "Read the following report and answer: which sections recommended retiring a "
    "contract, and what was the highest unit cost mentioned?\n\n" + _LONG_DOC,
]


def gpu_info():
    """(name, (major, minor)) of GPU 0, or (None, None) without CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None
        return torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)
    except Exception:  # noqa: BLE001
        return None, None


def pick_attention_backend(cc):
    """Which vLLM attention backend to force.

    gpt-oss needs attention sinks. vLLM serves sinks through FlashAttention-3
    (Hopper, SM90) or through FlashInfer's TRT-LLM kernels (SM100 datacenter
    Blackwell only). On every other GPU -- RTX PRO 6000 Blackwell (SM120),
    L40S / L4 / RTX 4090 (Ada) -- the selector falls through to TRITON_ATTN.
    SKIVE's query-capture and sparse-gather hooks are in FLASH_ATTN and
    TRITON_ATTN, so we force one of those two explicitly: the run is then
    deterministic and the log names the backend that carries the hooks.
    """
    forced = os.environ.get("SKIVE_ATTN_BACKEND", "auto").upper()
    if forced != "AUTO":
        return forced
    if cc is None:
        return None
    return "FLASH_ATTN" if cc[0] == 9 else "TRITON_ATTN"


def load_prompts():
    p = os.environ.get("PROMPTS")
    if not p:
        return DEFAULT_PROMPTS
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if p.endswith(".jsonl"):
                obj = json.loads(line)
                out.append(obj.get("prompt") or obj.get("question") or obj.get("input"))
            else:
                out.append(line)
    return out


def main():
    name, cc = gpu_info()
    backend = pick_attention_backend(cc)
    print(f"[run_gptoss] gpu={name} compute_capability={cc} attention_backend={backend}",
          flush=True)

    from vllm import LLM, SamplingParams

    kw = dict(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=int(os.environ.get("MAXLEN", "32768")),
        gpu_memory_utilization=float(os.environ.get("GPUMEM", "0.90")),
        tensor_parallel_size=TP,
        trust_remote_code=True,
        enforce_eager=True,
        enable_prefix_caching=False,   # SKIVE reclaim assumes no shared-block refcounts
    )
    if backend:
        kw["attention_backend"] = backend
    if os.environ.get("MAXSEQS"):
        kw["max_num_seqs"] = int(os.environ["MAXSEQS"])
    if MODE == "skive":
        kw.update(
            kv_evict_enabled=True,
            kv_evict_budget=BUDGET,
            kv_evict_num_sink_blocks=SINK,
            kv_evict_num_local_blocks=LOCAL,
        )
    print(f"[run_gptoss] mode={MODE} model={MODEL} tp={TP} "
          f"budget={BUDGET} sink={SINK} local={LOCAL} "
          f"metric={os.environ['SKIVE_METRIC']} agg={os.environ['SKIVE_HEAD_AGG']} "
          f"max_model_len={kw['max_model_len']} max_tokens={MAXTOK}",
          flush=True)
    llm = LLM(**kw)

    params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=MAXTOK)
    prompts = load_prompts()
    convs = [[{"role": "user", "content": p}] for p in prompts]

    # warm-up (kernel compile / graph capture excluded from timing)
    llm.chat(convs[:1], SamplingParams(temperature=0.0, max_tokens=8))

    t0 = time.perf_counter()
    outs = llm.chat(convs, params)
    dt = time.perf_counter() - t0

    out_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    in_tok = sum(len(o.prompt_token_ids) for o in outs)
    evicted = None
    if MODE == "skive":
        try:  # measured per-request eviction counts from the worker
            from vllm.kv_evict.integration import _skive_get_evicted_by_req
            res = llm.collective_rpc(_skive_get_evicted_by_req)
            evicted = sum(sum(d.values()) for d in res if isinstance(d, dict))
        except Exception as e:  # noqa: BLE001
            evicted = f"n/a ({type(e).__name__})"

    print("\n===== gpt-oss result =====")
    print(f"mode={MODE}  model={MODEL}  gpu={name}  backend={backend}  "
          f"metric={os.environ['SKIVE_METRIC'] if MODE == 'skive' else '-'}")
    print(f"n_prompts={len(prompts)}  in_tok={in_tok}  out_tok={out_tok}")
    print(f"wall={dt:.1f}s  tok/s={out_tok / dt:.1f}  evicted_blocks={evicted}")
    for p, o in zip(prompts, outs):
        print("\n--- prompt:", p[:80].replace("\n", " "))
        print(o.outputs[0].text[:600])

    tag = os.environ.get("RESULT_TAG", MODE)
    with open(f"gptoss_{tag}_result.json", "w") as f:
        json.dump(dict(mode=MODE, model=MODEL, gpu=name, compute_capability=cc,
                       attention_backend=backend, tp=TP, budget=BUDGET, sink=SINK,
                       local=LOCAL, n=len(prompts), in_tok=in_tok, out_tok=out_tok,
                       wall_s=round(dt, 2), tok_s=round(out_tok / dt, 1),
                       evicted_blocks=evicted,
                       metric=os.environ["SKIVE_METRIC"],
                       head_agg=os.environ["SKIVE_HEAD_AGG"],
                       outputs=[o.outputs[0].text for o in outs]), f, indent=2)
    print(f"saved gptoss_{tag}_result.json")


if __name__ == "__main__":
    main()
