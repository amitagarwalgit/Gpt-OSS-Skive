#!/usr/bin/env bash
# ==========================================================================
# GSM8K full lever grid on gpt-oss-20b, all 1,319 test problems:
#   {vk_ratio, value_attention} x {256, 512, 1024 tokens} x
#   {plain, prompt protected (pp), pp+trajectory, pp+redundancy, pp+both, pp+FP8 cache,
#    pp+evict every 64 steps, pp+64-token blocks}  plus FullKV eager / graphs / FP8.
#   bash skive/run_gsm8k_grid.sh            # ~2.5 h on an L40S; finished configs are skipped on re-run
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
run fullkv eager  CUDAGRAPH=
run fullkv cg
run fullkv cg_fp8 KV_DTYPE=fp8
for m in va vk; do
  for b in 256 512 1024; do
    L=8; [ "$b" = 256 ] && L=4          # the local window must fit inside a 256-token budget
    run $m@$b plain        LOCAL=$L
    run $m@$b pp           LOCAL=$L SKIVE_PROTECT_PROMPT=1
    run $m@$b pp_traj      LOCAL=$L SKIVE_PROTECT_PROMPT=1 SKIVE_QHIST=16
    run $m@$b pp_red       LOCAL=$L SKIVE_PROTECT_PROMPT=1 SKIVE_REDUNDANCY=0.3
    run $m@$b pp_traj_red  LOCAL=$L SKIVE_PROTECT_PROMPT=1 SKIVE_QHIST=16 SKIVE_REDUNDANCY=0.3
    run $m@$b pp_fp8       LOCAL=$L SKIVE_PROTECT_PROMPT=1 KV_DTYPE=fp8
    run $m@$b pp_e64       LOCAL=$L SKIVE_PROTECT_PROMPT=1 SKIVE_EVICT_EVERY=64 SKIVE_EVICT_MARGIN=16
    [ "$b" != 256 ] && run $m@$b pp_blk64 LOCAL=2 SKIVE_PROTECT_PROMPT=1 BLOCK_SIZE=64
  done
done
python "$BIN" csv --dataset gsm8k --out "$OUT"
python "$BIN" tables --out "$OUT" > "$OUT/TABLES.md" 2> "$OUT/logs/tables.err"
echo; echo "== results =="; cat "$OUT/TABLES.md"; echo "answers: $OUT/gsm8k_answers.csv"
