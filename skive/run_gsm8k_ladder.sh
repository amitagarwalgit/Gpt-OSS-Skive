#!/usr/bin/env bash
# ==========================================================================
# GSM8K budget ladder on gpt-oss-20b (all 1,319 test problems): fills in the
# 384 / 640 / 768-token budgets between the grid's 256 / 512 / 1024, for both
# metrics, with the three levers that matter (pp, pp+e64, pp+traj+red), then
# prints the budget x {vk, va} comparison tables (skive/gptoss_campaign.py compare).
#   bash skive/run_gsm8k_grid.sh && bash skive/run_gsm8k_ladder.sh   # ~1 h for the ladder on an L40S
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
run fullkv cg
for b in 384 640 768; do
  L=8; [ "$b" = 384 ] && L=4          # the local window must fit inside the budget
  for m in va vk; do
    run $m@$b pp          LOCAL=$L SKIVE_PROTECT_PROMPT=1
    run $m@$b pp_e64      LOCAL=$L SKIVE_PROTECT_PROMPT=1 SKIVE_EVICT_EVERY=64 SKIVE_EVICT_MARGIN=16
    run $m@$b pp_traj_red LOCAL=$L SKIVE_PROTECT_PROMPT=1 SKIVE_QHIST=16 SKIVE_REDUNDANCY=0.3
  done
done
python "$BIN" csv --dataset gsm8k --out "$OUT"
{ for t in plain pp pp_e64 pp_traj_red pp_red pp_traj pp_fp8 pp_blk64; do python "$BIN" compare --dataset gsm8k --out "$OUT" --tag $t; done
  python "$BIN" tables --out "$OUT"; } > "$OUT/TABLES.md" 2> "$OUT/logs/tables.err"
echo; echo "== results =="; cat "$OUT/TABLES.md"; echo "answers: $OUT/gsm8k_answers.csv"
