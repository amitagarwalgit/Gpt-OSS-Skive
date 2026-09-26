#!/usr/bin/env bash
# gpt-oss-20b smoke test: FullKV, then SKIVE with vk_ratio and value_attention (about 3 minutes).
#   bash skive/run_smoke.sh
# Env: SKIVE_MODEL (openai/gpt-oss-20b) MAXLEN (8192) MAXTOK (256) METRICS ("vk_ratio value_attention")
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/skive/_env.sh"
LOGS="$ROOT/skive/results/smoke"; mkdir -p "$LOGS"
export SKIVE_MODEL="${SKIVE_MODEL:-openai/gpt-oss-20b}" MAXLEN="${MAXLEN:-8192}" MAXTOK="${MAXTOK:-256}" SKIVE_DBG2=1 SKIVE_LOG_EVICT=1
METRICS="${METRICS:-vk_ratio value_attention}"
PAT='Using .*(attention backend|AttentionBackendEnum)|Mxfp4 MoE backend|run_gptoss\]|SKIVE DBG\] step=(1|2|3) |SKIVE 4c\] evicted|^mode=|^wall=|Traceback|Error|OutOfMemory'
cd /tmp
echo "--- FullKV ---"
python "$ROOT/skive/run_gptoss.py" fullkv > "$LOGS/fullkv.log" 2>&1 || true
grep -E "$PAT" "$LOGS/fullkv.log" | tail -6
for m in $METRICS; do
  echo "--- SKIVE metric=$m ---"
  SKIVE_METRIC=$m RESULT_TAG="skive_$m" python "$ROOT/skive/run_gptoss.py" skive > "$LOGS/skive_$m.log" 2>&1 || true
  grep -E "$PAT" "$LOGS/skive_$m.log" | tail -8
done
echo "logs: $LOGS/"
