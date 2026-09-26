#!/usr/bin/env bash
# ==========================================================================
# AIME 2024 on gpt-oss-20b: FullKV vs SKIVE KV-cache eviction.
#
#   bash skive/run_aime24.sh                   # full: 30 problems x 8 samples, 4 configs (~1.5 h on an L40S)
#   NPROB=5 NSAMP=2 bash skive/run_aime24.sh   # quick check (~10 min)
#   EXTRA=1 bash skive/run_aime24.sh           # also the exploratory rows (trajectory, redundancy, block 64, FP8)
#
# Configs run by default:
#   fullkv+cg        FullKV, piecewise CUDA graphs (the reference)
#   fullkv+cg_fp8    FullKV with an FP8 KV cache
#   va@2048+best     SKIVE value_attention, 2048-token budget, prompt protected, local window 32, graphs
#   va@4096+best     same at a 4096-token budget (accuracy at FullKV level, 56% of KV reclaimed)
#
# Env: NPROB (30) NSAMP (8) MAXTOK (16384) REASONING (medium) OUT (skive/results/aime24/new_run)
# Output: $OUT/TABLES.md (results table), $OUT/aime24_answers.csv (every generated answer),
#         one JSON per config, logs in $OUT/logs. Finished configs are skipped on re-run.
# ==========================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/skive/_env.sh"
OUT="${OUT:-$ROOT/skive/results/aime24/new_run}"; mkdir -p "$OUT/logs"; cd /tmp
BIN="$ROOT/skive/gptoss_campaign.py"
NPROB="${NPROB:-30}"
AIME="NSAMP=${NSAMP:-8} MAXTOK=${MAXTOK:-16384} REASONING=${REASONING:-medium}"
BASE="SKIVE_PROTECT_PROMPT=1 LOCAL=32 CUDAGRAPH=piecewise"
run() { # run <config> <tag> [env...]
  local cfg=$1 tag=$2; shift 2; local name="${cfg}${tag:++$tag}"
  [ -f "$OUT/aime24__${name}.json" ] && { echo "[skip] $name (done)"; return; }
  echo "--- aime24 $name  ($(date +%H:%M)) ---"
  env $AIME "$@" python "$BIN" run --dataset aime24 --config "$cfg" --n "$NPROB" --tag "$tag" --out "$OUT" \
    > "$OUT/logs/aime24__${name}.log" 2>&1 || echo "!! failed $name: see $OUT/logs/aime24__${name}.log"
  grep -E "^RESULT|Traceback" "$OUT/logs/aime24__${name}.log" | cut -c1-220 | tail -1
}
run fullkv  cg          CUDAGRAPH=piecewise
run fullkv  cg_fp8      CUDAGRAPH=piecewise KV_DTYPE=fp8
run va@2048 best        $BASE
run va@4096 best        $BASE
if [ "${EXTRA:-0}" = "1" ]; then
  run va@2048 traj      $BASE SKIVE_QHIST=16
  run va@2048 red       $BASE SKIVE_REDUNDANCY=0.3
  run va@2048 blk64     SKIVE_PROTECT_PROMPT=1 LOCAL=8 CUDAGRAPH=piecewise BLOCK_SIZE=64
  run va@4096 fp8       $BASE KV_DTYPE=fp8
fi
python "$BIN" csv --dataset aime24 --out "$OUT"
python "$BIN" tables --out "$OUT" > "$OUT/TABLES.md" 2> "$OUT/logs/tables.err"
echo; echo "== results =="; cat "$OUT/TABLES.md"; echo "answers: $OUT/aime24_answers.csv"
