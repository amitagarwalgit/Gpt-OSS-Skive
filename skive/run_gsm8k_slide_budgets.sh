#!/usr/bin/env bash
# ==========================================================================
# GSM8K at the nine "slide" budgets (1,024 .. 16,384 tokens), both metrics,
# prompt protected: the like-for-like table against the NarrativeQA / HotpotQA
# slides. Run after skive/run_gsm8k_grid.sh (needs the FullKV rows and 1,024);
# ~50 min on an L40S. Then render the slide:
#   python skive/make_gsm8k_slide.py skive/results/gsm8k/grid skive/docs/pdf/SKIVE_gptoss_gsm8k_slide
# Env: NPROB (1319) REASONING (medium) MAXTOK (2048) OUT (skive/results/gsm8k/grid)
# ==========================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/skive/_env.sh"
OUT="${OUT:-$ROOT/skive/results/gsm8k/grid}"; mkdir -p "$OUT/logs"; cd /tmp
BIN="$ROOT/skive/gptoss_campaign.py"
NPROB="${NPROB:-1319}"
COMMON="REASONING=${REASONING:-medium} MAXTOK=${MAXTOK:-2048} MINILM=0 CUDAGRAPH=piecewise"
run() { local cfg=$1 tag=$2; shift 2; local name="${cfg}${tag:++$tag}"
  [ -f "$OUT/gsm8k__${name}.json" ] && { echo "[skip] $name (done)"; return; }
  echo "--- gsm8k $name  ($(date +%H:%M)) ---"
  env $COMMON "$@" python "$BIN" run --dataset gsm8k --config "$cfg" --n "$NPROB" --tag "$tag" --out "$OUT" \
    > "$OUT/logs/gsm8k__${name}.log" 2>&1 || echo "!! failed $name: see $OUT/logs/gsm8k__${name}.log"
  grep -E "^RESULT|Traceback" "$OUT/logs/gsm8k__${name}.log" | cut -c1-160 | tail -1; }
run fullkv eager CUDAGRAPH=
run fullkv cg
for b in 1024 2048 3072 4096 6144 8192 11264 13664 16384; do
  for m in va vk; do run $m@$b pp LOCAL=8 SKIVE_PROTECT_PROMPT=1; done
done
python "$BIN" csv --dataset gsm8k --out "$OUT"
python "$BIN" compare --dataset gsm8k --out "$OUT" --tag pp | tee "$OUT/COMPARE_pp.md"
