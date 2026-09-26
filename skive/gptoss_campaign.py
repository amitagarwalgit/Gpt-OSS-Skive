"""gpt-oss accuracy + latency campaign for SKIVE -- produces the slide tables.

One process per (dataset, config) so SKIVE_METRIC (read at vLLM import) and
enforce_eager (FullKV vs FullKV cudagraph) are clean. Results are JSON files in
--out; `tables` renders them as the "<dataset>: accuracy and latency (all
budgets)" tables plus the "vk_ratio vs value_attention" table.

  python gptoss_campaign.py run    --dataset gov_report --config vk@1024 [--n 100] [--out DIR]
  python gptoss_campaign.py judge  --dataset gov_report [--out DIR]      # optional LLM-judge pass
  python gptoss_campaign.py tables [--out DIR] [> TABLES.md]

configs : fullkv | fullkv_cg | vk@<budget tokens> | va@<budget tokens>
datasets: gov_report, multi_news  (LongBench summarization, acc = ROUGE-L)
          longalpaca              (LongAlpaca-12k long QA, acc = token-F1; long items only)
          gsm8k                   (exact match)
          aime24                  (mean pass rate over NSAMP samples, TriAttention protocol)

Columns (per config): cross% (requests that evicted >= 1 block), acc (dAcc vs
FullKV), judge (1-5, optional), rouge2, MiniLM (cosine, optional), rep4g
(repeated 4-gram fraction), ttft (s), tpot (ms), e2e (s), wall (s), med gen
tok, batch tok/s. Latencies come from vLLM's per-request RequestStateStats.

Env: SKIVE_MODEL (openai/gpt-oss-20b) MAXLEN (32768) GPUMEM (0.90) TP (1)
     SINK (2) LOCAL (8) REASONING (low) MAXTOK (per-dataset default) NSAMP (8)
     SKIVE_ATTN_BACKEND (auto) SEED (0)
"""
import argparse
import glob
import io
import json
import os
import re
import statistics
import sys
import time
import zipfile

# ---- argv is parsed (in main) BEFORE importing vllm: SKIVE_METRIC must be in the env first
ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["run", "judge", "tables", "csv"])
ap.add_argument("--dataset", default=None)
ap.add_argument("--config", default="fullkv")
ap.add_argument("--n", type=int, default=int(os.environ.get("N", "100")))
ap.add_argument("--out", default=os.environ.get("OUT", "gptoss_results"))
ap.add_argument("--rep-budget", default=os.environ.get("REP_BUDGET", ""),
                help="tables: budget for the vk-vs-va table, e.g. 1024 (default: largest common)")
ap.add_argument("--tag", default=os.environ.get("TAG", ""),
                help="run: suffix for variant runs of the same config (e.g. pp_l32); kept in the row label")
ARGS = None   # set in main(); gptoss_throughput.py imports this module without running main
CFG = None

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# the evicted-block counter is fetched with collective_rpc(callable); the multiprocess
# EngineCore only accepts cloudpickled callables with this flag (else TypeError)
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("SKIVE_HEAD_AGG", "zmax")
os.environ.setdefault("SKIVE_SCORE_LAYER_STRIDE", "4")
os.environ.setdefault("SKIVE_SPARSE_GATHER", "1")
os.environ.setdefault("SKIVE_EVICT_EVERY", "16")

MODEL = os.environ.get("SKIVE_MODEL", "openai/gpt-oss-20b")
MAXLEN = int(os.environ.get("MAXLEN", "32768"))
GPUMEM = float(os.environ.get("GPUMEM", "0.90"))
TP = int(os.environ.get("TP", "1"))
SINK = int(os.environ.get("SINK", "2"))
LOCAL = int(os.environ.get("LOCAL", "8"))
REASONING = os.environ.get("REASONING", "low")
NSAMP = int(os.environ.get("NSAMP", "8"))
SEED = int(os.environ.get("SEED", "0"))
BLOCK = int(os.environ.get("BLOCK_SIZE", "16"))   # vLLM block_size (budgets are given in tokens)

DEFAULT_MAXTOK = {"gov_report": 512, "multi_news": 512, "longalpaca": 512, "hotpotqa": 256,
                  "gsm8k": 1024, "aime24": 8192}
ACC_NAME = {"gov_report": "ROUGE-L", "multi_news": "ROUGE-L", "longalpaca": "token-F1",
            "hotpotqa": "token-F1", "gsm8k": "exact match", "aime24": "pass rate"}
KIND = {"gov_report": "Summarization", "multi_news": "Summarization",
        "longalpaca": "Long-context QA", "hotpotqa": "Multi-hop long-context QA",
        "gsm8k": "Math reasoning", "aime24": "Math reasoning"}


# ---------------------------------------------------------------------------
# config parsing
# ---------------------------------------------------------------------------
def parse_config(c):
    """-> dict(name, evict, metric, budget_tok, budget_blk, cudagraph)."""
    c = c.lower()
    if c in ("fullkv", "fullkv_cg"):
        return dict(name=c, evict=False, metric=None, budget_tok=None, budget_blk=None,
                    cudagraph=(c == "fullkv_cg"))
    m = re.fullmatch(r"(vk|va)@(\d+)", c)
    if not m:
        sys.exit(f"bad --config {c!r}: fullkv | fullkv_cg | vk@<tok> | va@<tok>")
    tok = int(m.group(2))
    if tok % BLOCK:
        sys.exit(f"budget {tok} must be a multiple of block_size {BLOCK}")
    return dict(name=c, evict=True, metric={"vk": "vk_ratio", "va": "value_attention"}[m.group(1)],
                budget_tok=tok, budget_blk=tok // BLOCK, cudagraph=False)


def apply_config_env(cfg):
    """SKIVE_METRIC is read by the attention hooks at vLLM import: set it first."""
    if cfg and cfg["evict"]:
        os.environ["SKIVE_METRIC"] = cfg["metric"]
    else:
        os.environ.setdefault("SKIVE_METRIC", "vk_ratio")


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
LONGBENCH_PROMPT = {
    "gov_report": ("You are given a report by a government agency. Write a one-page summary "
                   "of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of "
                   "the report.\n\nSummary:"),
    "multi_news": ("You are given several news passages. Write a one-page summary of all "
                   "news.\n\nNews:\n{context}\n\nNow, write a one-page summary of all the "
                   "news.\n\nSummary:"),
    "hotpotqa": ("Answer the question based on the given passages. Only give me the answer "
                 "and do not output any other words.\n\nThe following are given passages.\n"
                 "{context}\n\nAnswer the question based on the given passages. Only give me "
                 "the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"),
}


def _longbench_rows(name):
    """LongBench jsonl rows via the dataset's data.zip (no loading script needed)."""
    try:
        from huggingface_hub import hf_hub_download
        zp = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
        with zipfile.ZipFile(zp) as z:
            member = next(m for m in z.namelist() if m.endswith(f"/{name}.jsonl") or m == f"{name}.jsonl")
            with z.open(member) as f:
                return [json.loads(line) for line in io.TextIOWrapper(f, encoding="utf-8") if line.strip()]
    except Exception as e:  # noqa: BLE001
        print(f"[data] data.zip route failed ({type(e).__name__}: {e}); trying load_dataset", flush=True)
        from datasets import load_dataset
        return list(load_dataset("THUDM/LongBench", name, split="test"))


def load_items(name, n, tokenizer, maxtok):
    """-> list of dict(prompt, ref, gold)  (gold only for exact-answer tasks)."""
    items = []
    if name in LONGBENCH_PROMPT:
        for r in _longbench_rows(name):
            items.append(dict(prompt=LONGBENCH_PROMPT[name].format(context=r["context"],
                                                                   input=r.get("input", "")),
                              question=r.get("input") or "(summarize)",
                              ref=r["answers"][0] if r.get("answers") else "",
                              refs=list(r.get("answers") or [])))
    elif name == "longalpaca":
        from datasets import load_dataset
        ds = load_dataset("Yukang/LongAlpaca-12k", split="train")
        rows = [(r["instruction"], r["output"]) for r in ds]
        # long-prompt items only (every request must genuinely need its context)
        min_tok = int(os.environ.get("LONGALPACA_MIN_TOK", "8000"))
        for ins, out in rows:
            if len(tokenizer(ins).input_ids) >= min_tok:
                items.append(dict(prompt=ins, question=ins[-300:], ref=out))
            if len(items) >= n:
                break
    elif name == "gsm8k":
        from datasets import load_dataset
        for r in load_dataset("openai/gsm8k", "main", split=f"test[:{n}]"):
            m = re.search(r"####\s*([-0-9,\.]+)", r["answer"])
            items.append(dict(prompt=r["question"] + "\nPlease reason step by step and put "
                              "your final answer after '####'.", question=r["question"],
                              ref=r["answer"], gold=m.group(1).replace(",", "").rstrip(".") if m else None))
    elif name == "aime24":
        from datasets import load_dataset
        for r in load_dataset("Maxwell-Jia/AIME_2024", split="train"):
            items.append(dict(prompt="Solve the following AIME problem. Reason step by step, then "
                              "give the final integer answer inside \\boxed{}.\n\nProblem: "
                              + str(r["Problem"]), question=str(r["Problem"]), ref=str(r["Answer"]),
                              gold=re.findall(r"-?\d+", str(r["Answer"]))[-1]))
    else:
        sys.exit(f"unknown dataset {name}")
    items = items[:n]
    # middle-truncate prompts that would not fit (LongBench convention)
    limit = MAXLEN - maxtok - 256
    for it in items:
        ids = tokenizer(it["prompt"]).input_ids
        if len(ids) > limit:
            half = limit // 2
            it["prompt"] = tokenizer.decode(ids[:half]) + tokenizer.decode(ids[-half:])
            it["truncated"] = True
    return items


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
_HARMONY_TOK = re.compile(r"<\|[a-z_]+\|>")


def extract_final(text):
    """gpt-oss harmony output -> the final channel only (drop analysis)."""
    for marker in ("<|channel|>final<|message|>", "assistantfinal"):
        if marker in text:
            text = text.split(marker)[-1]
            break
    return _HARMONY_TOK.sub("", text).strip()


def _norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return s.split()


def token_f1(pred, ref):
    p, r = _norm(pred), _norm(ref)
    if not p or not r:
        return float(p == r)
    from collections import Counter
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def rep4g(text):
    """fraction of 4-grams that are repeats (1 - unique/total); 0 for short text."""
    w = text.split()
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 0.0 if len(grams) < 2 else 1.0 - len(set(grams)) / len(grams)


def last_number(text):
    m = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return m[-1].rstrip(".") if m else None


def boxed_number(text):
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    cand = m[-1] if m else text
    n = re.findall(r"-?\d+", cand)
    return n[-1] if n else None


def num_eq(a, b):
    try:
        return a is not None and b is not None and abs(float(a) - float(b)) < 1e-4
    except ValueError:
        return a == b


def score_items(name, items, preds):
    """-> dict(acc, rouge2, minilm, per_item) with acc on the 0-100 scale; preds = list[list[str]]."""
    accs, r2s = [], []
    scorer = None
    if name in ("gov_report", "multi_news"):
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge2", "rougeL"], use_stemmer=True)
    for it, ps in zip(items, preds):
        if name in ("gov_report", "multi_news"):
            s = scorer.score(it["ref"], ps[0])
            accs.append(100 * s["rougeL"].fmeasure); r2s.append(s["rouge2"].fmeasure)
        elif name == "hotpotqa":   # LongBench: best F1 over the reference answers
            refs = it.get("refs") or [it["ref"]]
            accs.append(100 * max(token_f1(ps[0], r) for r in refs))
        elif name == "longalpaca":
            accs.append(100 * token_f1(ps[0], it["ref"]))
            if scorer is None:
                from rouge_score import rouge_scorer
                scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)
            r2s.append(scorer.score(it["ref"], ps[0])["rouge2"].fmeasure)
        elif name == "gsm8k":
            accs.append(100.0 * num_eq(last_number(ps[0]), it["gold"]))
        elif name == "aime24":
            accs.append(100.0 * sum(boxed_number(p) == it["gold"] for p in ps) / len(ps))
    out = dict(acc=statistics.mean(accs), rouge2=(statistics.mean(r2s) if r2s else None), minilm=None,
               per_item=accs)
    if name in ("gov_report", "multi_news", "longalpaca") and os.environ.get("MINILM", "1") == "1":
        try:
            from sentence_transformers import SentenceTransformer, util
            st = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            e1 = st.encode([p[0] for p in preds], convert_to_tensor=True, batch_size=16)
            e2 = st.encode([it["ref"] for it in items], convert_to_tensor=True, batch_size=16)
            out["minilm"] = float(util.cos_sim(e1, e2).diagonal().mean())
        except Exception as e:  # noqa: BLE001
            print(f"[score] MiniLM skipped ({type(e).__name__}: {e})", flush=True)
    return out


# ---------------------------------------------------------------------------
# engine helpers
# ---------------------------------------------------------------------------
def pick_backend():
    forced = os.environ.get("SKIVE_ATTN_BACKEND", "auto").upper()
    if forced != "AUTO":
        return forced
    try:
        import torch
        return "FLASH_ATTN" if torch.cuda.get_device_capability(0)[0] == 9 else "TRITON_ATTN"
    except Exception:  # noqa: BLE001
        return None


def make_llm(cfg):
    from vllm import LLM
    kw = dict(model=MODEL, dtype="bfloat16", max_model_len=MAXLEN, gpu_memory_utilization=GPUMEM,
              tensor_parallel_size=TP, trust_remote_code=True, seed=SEED,
              enforce_eager=not cfg["cudagraph"], enable_prefix_caching=False,
              disable_log_stats=False)
    # CUDAGRAPH=piecewise: graphs for the MoE/MLP pieces, attention stays eager, so
    # SKIVE's block-table edits and sparse-gather compaction remain legal (the Triton
    # hook only disables compaction under FULL decode capture).
    if os.environ.get("CUDAGRAPH", "").lower() == "piecewise":
        from vllm.config import CompilationConfig, CUDAGraphMode
        kw["enforce_eager"] = False
        kw["compilation_config"] = CompilationConfig(cudagraph_mode=CUDAGraphMode.PIECEWISE)
    if os.environ.get("KV_DTYPE"):            # e.g. fp8: half the bytes per cached token
        kw["kv_cache_dtype"] = os.environ["KV_DTYPE"]
    if BLOCK != 16:
        kw["block_size"] = BLOCK
    b = pick_backend()
    if b:
        kw["attention_backend"] = b
    if os.environ.get("MAXSEQS"):
        kw["max_num_seqs"] = int(os.environ["MAXSEQS"])
    if cfg["evict"]:
        kw.update(kv_evict_enabled=True, kv_evict_budget=cfg["budget_blk"],
                  kv_evict_num_sink_blocks=SINK, kv_evict_num_local_blocks=LOCAL)
    llm = LLM(**kw)
    try:
        cc = llm.llm_engine.vllm_config.cache_config
        max_conc = cc.num_gpu_blocks * cc.block_size / MAXLEN
    except Exception:  # noqa: BLE001
        max_conc = None
    return llm, b, max_conc


def chat(llm, convs, sp):
    try:
        return llm.chat(convs, sp, chat_template_kwargs={"reasoning_effort": REASONING})
    except Exception as e:  # noqa: BLE001
        print(f"[gen] reasoning_effort kwarg rejected ({type(e).__name__}); using system prompt",
              flush=True)
        convs = [[{"role": "system", "content": f"Reasoning: {REASONING}"}] + c for c in convs]
        return llm.chat(convs, sp)


def req_latencies(o):
    """(ttft_s, tpot_s, e2e_s, gen_tokens) from vLLM's RequestStateStats."""
    n = sum(len(c.token_ids) for c in o.outputs)
    m = o.metrics
    if m is None or not getattr(m, "first_token_ts", 0):
        return None, None, None, n
    ttft = float(getattr(m, "first_token_latency", 0.0) or 0.0)
    decode = max(0.0, float(m.last_token_ts - m.first_token_ts))
    n_first = len(o.outputs[0].token_ids)
    tpot = decode / (n_first - 1) if n_first > 1 else None
    return ttft, tpot, ttft + decode, n


def evicted_by_req(llm):
    try:
        from vllm.kv_evict.integration import _skive_get_evicted_by_req
        d = {}
        for part in llm.collective_rpc(_skive_get_evicted_by_req):
            if isinstance(part, dict):
                d.update(part)
        return d
    except Exception as e:  # noqa: BLE001
        print(f"[evict] counter unavailable ({type(e).__name__}: {str(e)[:300]})", flush=True)
        return {}


def gauge(llm, name_sub):
    try:
        for m in llm.get_metrics():
            if name_sub in m.name and hasattr(m, "value"):
                return float(m.value)
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def cmd_run():
    name, cfg = ARGS.dataset, CFG
    maxtok = int(os.environ.get("MAXTOK", DEFAULT_MAXTOK[name]))
    os.makedirs(ARGS.out, exist_ok=True)
    run_name = cfg["name"] + (f"+{ARGS.tag}" if ARGS.tag else "")
    out_path = os.path.join(ARGS.out, f"{name}__{run_name}.json")
    knobs = {k: os.environ.get(k) for k in ("SKIVE_PROTECT_PROMPT", "SKIVE_EVICT_EVERY", "SKIVE_EVICT_MARGIN",
                                            "SKIVE_SCORE_LAYER_STRIDE", "CUDAGRAPH", "LOCAL", "SINK",
                                            "SKIVE_QHIST", "SKIVE_QHIST_STRIDE", "SKIVE_REDUNDANCY",
                                            "KV_DTYPE", "BLOCK_SIZE")
             if os.environ.get(k)}
    print(f"[campaign] dataset={name} config={run_name} model={MODEL} n={ARGS.n} "
          f"maxtok={maxtok} reasoning={REASONING} metric={os.environ.get('SKIVE_METRIC')} knobs={knobs}",
          flush=True)

    llm, backend, max_conc = make_llm(cfg)
    tok = llm.get_tokenizer()
    items = load_items(name, ARGS.n, tok, maxtok)
    convs = [[{"role": "user", "content": it["prompt"]}] for it in items]
    from vllm import SamplingParams
    if name == "aime24":
        sp = SamplingParams(n=NSAMP, temperature=0.6, top_p=0.95, max_tokens=maxtok,
                            skip_special_tokens=False, seed=SEED)
    else:
        sp = SamplingParams(temperature=0.0, max_tokens=maxtok, skip_special_tokens=False, seed=SEED)

    chat(llm, convs[:1], SamplingParams(temperature=0.0, max_tokens=8))  # warm-up
    # snapshot the worker's per-request eviction counter AFTER the warm-up: keys
    # that appear later belong to this batch (worker ids differ from o.request_id)
    ev0 = evicted_by_req(llm) if cfg["evict"] else {}
    t0 = time.perf_counter()
    outs = chat(llm, convs, sp)
    wall = time.perf_counter() - t0

    preds = [[extract_final(c.text) for c in o.outputs] for o in outs]
    raw_len = [sum(len(c.token_ids) for c in o.outputs) for o in outs]
    ttfts, tpots, e2es, gens = [], [], [], []
    for o in outs:
        ttft, tpot, e2e, n = req_latencies(o)
        gens.append(n)
        if ttft is not None:
            ttfts.append(ttft); e2es.append(e2e)
        if tpot is not None:
            tpots.append(tpot)
    prompt_tok = sum(len(o.prompt_token_ids) for o in outs)
    gen_tok = sum(raw_len)

    ev1 = evicted_by_req(llm) if cfg["evict"] else {}
    n_req = len(outs) * (NSAMP if name == "aime24" else 1)
    # per-request eviction counts for THIS batch only = counter keys new since the
    # post-warm-up snapshot (exact, independent of request-id naming)
    per_req = [v for k, v in ev1.items() if k not in ev0]
    crossed = min(n_req, sum(1 for v in per_req if v > 0))
    evicted_blocks = sum(per_req)
    preempt = gauge(llm, "num_preemptions")

    sc = score_items(name, items, preds)
    res = dict(
        dataset=name, config=run_name, base_config=cfg["name"], tag=ARGS.tag, knobs=knobs,
        metric=cfg["metric"], budget_tok=cfg["budget_tok"],
        cudagraph=cfg["cudagraph"] or os.environ.get("CUDAGRAPH", "").lower() == "piecewise",
        model=MODEL, backend=backend, n=len(items),
        n_requests=n_req, maxtok=maxtok, reasoning=REASONING,
        acc=sc["acc"], acc_name=ACC_NAME[name], rouge2=sc["rouge2"], minilm=sc["minilm"],
        rep4g=statistics.mean(rep4g(p[0]) for p in preds),
        cross_pct=(100.0 * crossed / n_req) if cfg["evict"] else None,
        evicted_blocks=evicted_blocks,
        # eviction rate = cache tokens evicted per generated token (how aggressively the
        # cache churns while decoding); evict_blk_per_req = mean evicted blocks per request
        evict_rate=(evicted_blocks * BLOCK / max(1, gen_tok)) if cfg["evict"] else 0.0,
        evict_blk_per_req=(evicted_blocks / max(1, n_req)) if cfg["evict"] else 0.0,
        kv_saved_pct=(100.0 * evicted_blocks * BLOCK / max(1, prompt_tok + gen_tok)) if cfg["evict"] else 0.0,
        ttft_s=statistics.mean(ttfts) if ttfts else None,
        ttft_p90_s=(sorted(ttfts)[int(0.9 * (len(ttfts) - 1))] if ttfts else None),
        tpot_ms=1000 * statistics.mean(tpots) if tpots else None,
        e2e_s=statistics.mean(e2es) if e2es else None,
        wall_s=wall, med_gen_tok=statistics.median(gens), mean_gen_tok=statistics.mean(gens),
        prompt_tok=prompt_tok, gen_tok=gen_tok, batch_tok_s=gen_tok / wall,
        prefill_tok_s=prompt_tok / wall, max_concurrency=max_conc, preemptions=preempt,
        truncated=sum(1 for it in items if it.get("truncated")),
        judge=None,
        outputs=[dict(question=it.get("question", "")[:1000], pred=p[0][:4000],
                      preds=[x[:4000] for x in p] if len(p) > 1 else None,
                      ref=it["ref"][:4000], gold=it.get("gold"), score=s, gen_tok=g)
                 for p, it, g, s in zip(preds, items, gens, sc["per_item"])],
    )
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"RESULT {json.dumps({k: v for k, v in res.items() if k != 'outputs'})}", flush=True)
    print(f"saved {out_path}", flush=True)


# ---------------------------------------------------------------------------
# judge (optional): the same model grades every config's outputs 1-5 vs the reference
# ---------------------------------------------------------------------------
JUDGE_PROMPT = ("You are grading a model answer against a reference.\n\nTask type: {kind}\n\n"
                "Reference:\n{ref}\n\nModel answer:\n{pred}\n\nRate the model answer from 1 "
                "(useless or wrong) to 5 (as good as the reference) for correctness, coverage "
                "and coherence. Reply with the single digit only.")


def cmd_judge():
    name = ARGS.dataset
    files = sorted(glob.glob(os.path.join(ARGS.out, f"{name}__*.json")))
    if not files:
        sys.exit(f"no results for {name} in {ARGS.out}")
    llm, _, _ = make_llm(parse_config("fullkv"))
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=512, skip_special_tokens=False)
    for fp in files:
        res = json.load(open(fp))
        convs = [[{"role": "user", "content": JUDGE_PROMPT.format(
            kind=KIND[name], ref=o["ref"][:6000], pred=o["pred"][:6000])}] for o in res["outputs"]]
        outs = chat(llm, convs, sp)
        grades = []
        for o in outs:
            m = re.findall(r"[1-5]", extract_final(o.outputs[0].text))
            if m:
                grades.append(int(m[0]))
        res["judge"] = statistics.mean(grades) if grades else None
        res["judge_n"] = len(grades)
        json.dump(res, open(fp, "w"), indent=1)
        print(f"[judge] {os.path.basename(fp)}: {res['judge']} (n={len(grades)})", flush=True)


# ---------------------------------------------------------------------------
# csv: every generated answer of a dataset, all configs (one row per sample)
# ---------------------------------------------------------------------------
def cmd_csv():
    import csv
    name = ARGS.dataset
    files = sorted(glob.glob(os.path.join(ARGS.out, f"{name}__*.json")))
    if not files:
        sys.exit(f"no results for {name} in {ARGS.out}")
    path = os.path.join(ARGS.out, f"{name}_answers.csv")
    n_rows = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "config", "budget_tokens", "item", "sample", "question",
                    "reference", "gold", "answer", "score", "gen_tokens"])
        for fp in files:
            r = json.load(open(fp))
            for i, o in enumerate(r["outputs"]):
                samples = o.get("preds") or [o["pred"]]
                for k, ans in enumerate(samples):
                    sc = o.get("score")
                    if o.get("preds") and o.get("gold") is not None:   # per-sample correctness
                        sc = 100.0 * (boxed_number(ans) == o["gold"])
                    w.writerow([name, r["config"], r["budget_tok"] or "", i, k, o.get("question", ""),
                                o["ref"], o.get("gold") or "", ans, "" if sc is None else round(sc, 2),
                                o.get("gen_tok", "")])
                    n_rows += 1
    print(f"wrote {path} ({n_rows} rows)")


# ---------------------------------------------------------------------------
# tables (markdown, the slide layout)
# ---------------------------------------------------------------------------
def _f(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def _delta_abs(x, base, nd=1):
    if x is None or base is None:
        return _f(x, nd)
    return f"{x:.{nd}f} ({x - base:+.{nd}f})"


def _delta_pct(x, base, nd=1):
    if x is None or base is None or not base:
        return _f(x, nd)
    return f"{x:.{nd}f} ({100 * (x - base) / base:+.0f}%)"


def _label(r):
    base = r.get("base_config") or r["config"].split("+")[0]
    tag = r.get("tag") or ("+" + r["config"].split("+", 1)[1] if "+" in r["config"] else "")
    tag = (" " + tag.lstrip("+")) if tag else ""
    if base == "fullkv":
        return "FullKV" + tag
    if base == "fullkv_cg":
        return "FullKV cg" + tag
    return f"{base.split('@')[0]} @{r['budget_tok']}{tag}"


def _order(r):
    c = r.get("base_config") or r["config"].split("+")[0]
    return (0, 0, r["config"]) if c == "fullkv" else (1, 0, r["config"]) if c == "fullkv_cg" else \
        (2 if c.startswith("vk") else 3, r["budget_tok"], r["config"])


def cmd_tables():
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ARGS.out, "*__*.json")))]
    if not rows:
        sys.exit(f"no results in {ARGS.out}")
    for r in rows:
        r.pop("outputs", None)
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    model = rows[0]["model"]; backend = rows[0].get("backend")
    print(f"# SKIVE on {model} -- accuracy and latency (attention backend {backend})\n")
    for ds, rs in by_ds.items():
        rs.sort(key=_order)
        base = next((r for r in rs if r["config"] == "fullkv"), rs[0])
        for r in rs:  # cross% / KV columns key off the base config name
            r["_base"] = r.get("base_config") or r["config"].split("+")[0]
        print(f"## {ds}: accuracy and latency (all budgets)\n")
        print(f"{KIND[ds]}, {ACC_NAME[ds]}. FullKV acc {_f(base['acc'])}, judge {_f(base.get('judge'), 2)}. "
              f"n={base['n']} prompts, max_tokens={base['maxtok']}, reasoning={base['reasoning']}. "
              f"dAcc / % deltas are vs FullKV (eager). evict rate = cache tokens evicted per generated "
              f"token; evict blk/req = evicted blocks per request; KV saved = evicted / total tokens.\n")
        print("| Config | cross% | evict rate | evict blk/req | acc (dAcc) | judge | rouge2 | MiniLM | rep4g "
              "| ttft (s) | tpot (ms) | e2e (s) | wall (s) | med gen tok | batch tok/s | KV saved |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rs:
            print("| " + " | ".join([
                _label(r),
                "-" if r["cross_pct"] is None else f"{min(100.0, r['cross_pct']):.0f}%",
                "-" if r["cross_pct"] is None else f"{r.get('evict_rate', 0.0):.2f}",
                "-" if r["cross_pct"] is None else f"{r.get('evict_blk_per_req', 0.0):.0f}",
                _delta_abs(r["acc"], base["acc"]) if r is not base else _f(r["acc"]),
                _f(r.get("judge"), 2), _f(r["rouge2"], 3), _f(r["minilm"], 2), _f(r["rep4g"], 3),
                _delta_pct(r["ttft_s"], base["ttft_s"]) if r is not base else _f(r["ttft_s"]),
                _delta_pct(r["tpot_ms"], base["tpot_ms"], 0) if r is not base else _f(r["tpot_ms"], 0),
                _delta_pct(r["e2e_s"], base["e2e_s"]) if r is not base else _f(r["e2e_s"]),
                _delta_pct(r["wall_s"], base["wall_s"], 0) if r is not base else _f(r["wall_s"], 0),
                _delta_pct(r["med_gen_tok"], base["med_gen_tok"], 0) if r is not base else _f(r["med_gen_tok"], 0),
                _delta_pct(r["batch_tok_s"], base["batch_tok_s"], 0) if r is not base else _f(r["batch_tok_s"], 0),
                f"{r['kv_saved_pct']:.0f}%" if r["_base"] not in ("fullkv", "fullkv_cg") else "-",
            ]) + " |")
        print()
        worst = min((r for r in rs if r["_base"].startswith(("vk", "va"))), key=lambda r: r["acc"], default=None)
        best = max((r for r in rs if r["_base"].startswith(("vk", "va"))), key=lambda r: r["acc"], default=None)
        if worst and best:
            print(f"Accuracy cost {worst['acc'] - base['acc']:+.1f} at {worst['budget_tok']} "
                  f"({worst['kv_saved_pct']:.0f}% KV saved) shrinking to {best['acc'] - base['acc']:+.1f} "
                  f"at {best['budget_tok']}. Preemptions: FullKV {base.get('preemptions')}, "
                  f"eviction {best.get('preemptions')}.\n")

    # vk_ratio vs value_attention at a representative budget per dataset
    print("## vk_ratio vs value_attention\n")
    print("Representative budget per dataset (largest budget where both metrics were run"
          + (f", or {ARGS.rep_budget}" if ARGS.rep_budget else "") + ").\n")
    print("| Dataset (budget) | acc vk / va | judge vk / va | rouge2 vk / va | batch tok/s vk / va "
          "| ttft (s) vk / va | max concurrency vk / va |")
    print("|---|---|---|---|---|---|---|")
    for ds, rs in by_ds.items():
        vk = {r["budget_tok"]: r for r in rs if r["config"].startswith("vk") and "+" not in r["config"]}
        va = {r["budget_tok"]: r for r in rs if r["config"].startswith("va") and "+" not in r["config"]}
        common = sorted(set(vk) & set(va))
        if not common:
            continue
        b = int(ARGS.rep_budget) if ARGS.rep_budget and int(ARGS.rep_budget) in common else common[-1]
        k, v = vk[b], va[b]
        print(f"| {ds} @{b} | {_f(k['acc'])} / {_f(v['acc'])} | {_f(k.get('judge'), 2)} / {_f(v.get('judge'), 2)} "
              f"| {_f(k['rouge2'], 3)} / {_f(v['rouge2'], 3)} | {_f(k['batch_tok_s'], 0)} / {_f(v['batch_tok_s'], 0)} "
              f"| {_f(k['ttft_s'])} / {_f(v['ttft_s'])} | {_f(k['max_concurrency'])} / {_f(v['max_concurrency'])} |")
    print()


def main():
    global ARGS, CFG
    ARGS = ap.parse_args()
    if ARGS.cmd == "run":
        if not ARGS.dataset:
            sys.exit("run needs --dataset")
        CFG = parse_config(ARGS.config)
        apply_config_env(CFG)
    elif ARGS.cmd in ("judge", "csv"):
        if not ARGS.dataset:
            sys.exit(f"{ARGS.cmd} needs --dataset")
        apply_config_env(None)
    {"run": cmd_run, "judge": cmd_judge, "tables": cmd_tables, "csv": cmd_csv}[ARGS.cmd]()


if __name__ == "__main__":
    main()
