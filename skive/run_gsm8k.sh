#!/usr/bin/env bash
# ==========================================================================
# GSM8K, full test set (1,319 problems), on gpt-oss-20b: FullKV vs SKIVE.
#   bash skive/run_gsm8k.sh                 # ~1 h on an L40S
#   NPROB=200 bash skive/run_gsm8k.sh       # quick check
# Configs: FullKV (piecewise CUDA graphs), FullKV FP8, SKIVE value_attention at
#          1024 / 512 / 256-token budgets (prompt protected, graphs), vk_ratio at 512.
# Env: NPROB (1319) REASONING (medium) MAXTOK (2048) OUT (skive/results/gsm8k/new_run)
# ==========================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/skive/_env.sh"
OUT="${OUT:-$ROOT/skive/results/gsm8k/new_run}"; mkdir -p "$OUT/logs"; cd /tmp
BIN="$ROOT/skive/gptoss_campaign.py"
NPROB="${NPROB:-1319}"
COMMON="REASONING=${REASONING:-medium} MAXTOK=${MAXTOK:-2048} MINILM=0"
BEST="SKIVE_PROTECT_PROMPT=1 LOCAL=8 CUDAGRAPH=piecewise"
run() { local cfg=$1 tag=$2; shift 2; local name="${cfg}${tag:++$tag}"
  [ -f "$OUT/gsm8k__${name}.json" ] && { echo "[skip] $name (done)"; return; }
  echo "--- gsm8k $name  ($(date +%H:%M)) ---"
  env $COMMON "$@" python "$BIN" run --dataset gsm8k --config "$cfg" --n "$NPROB" --tag "$tag" --out "$OUT" \
    > "$OUT/logs/gsm8k__${name}.log" 2>&1 || echo "!! failed $name: see $OUT/logs/gsm8k__${name}.log"
  grep -E "^RESULT|Traceback" "$OUT/logs/gsm8k__${name}.log" | cut -c1-200 | tail -1; }
run fullkv  cg        CUDAGRAPH=piecewise
run fullkv  cg_fp8    CUDAGRAPH=piecewise KV_DTYPE=fp8
run va@1024 best      $BEST
run va@512  best      $BEST
run va@256  best_l4   SKIVE_PROTECT_PROMPT=1 LOCAL=4 CUDAGRAPH=piecewise
run vk@512  best      $BEST
python "$BIN" csv --dataset gsm8k --out "$OUT"
python "$BIN" tables --out "$OUT" > "$OUT/TABLES.md" 2> "$OUT/logs/tables.err"
echo; echo "== results =="; cat "$OUT/TABLES.md"; echo "answers: $OUT/gsm8k_answers.csv"
