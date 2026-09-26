#!/usr/bin/env bash
# ==========================================================================
# SKIVE fast install (no CUDA build): a venv with the official vLLM 0.23.0
# wheel, then this tree's 7 patched vLLM files and vllm/kv_evict copied over
# the installed package. The compiled kernels come from the wheel; SKIVE is
# pure Python, so the result is exactly this fork, ready in about ten minutes.
#
#   bash skive/install.sh                 # creates ./.venv
#   PYTHON=python3.11 VENV=/path bash skive/install.sh
#   SKIP_TESTS=1 bash skive/install.sh
#
# Full source build instead (needs the CUDA toolkit, about an hour):
#   python -m venv .venv && source .venv/bin/activate && pip install -e .   (see README.vllm.md)
# ==========================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
PYTHON="${PYTHON:-python3}"
VLLM_VERSION="${VLLM_VERSION:-0.23.0}"

echo "== 1/4 venv at $VENV =="
[ -x "$VENV/bin/python" ] || "$PYTHON" -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install -q -U pip
"$PY" --version

echo "== 2/4 vLLM $VLLM_VERSION wheel + evaluation dependencies (several minutes) =="
"$PY" -m pip install -q "vllm==$VLLM_VERSION" -r "$ROOT/skive/requirements.txt"

echo "== 3/4 overlay this tree's SKIVE files onto the installed wheel =="
# Resolve the installed package from /tmp: from inside this repo, `import vllm`
# would pick up the source tree (which has no compiled kernels).
VLLM_DIR="$(cd /tmp && "$PY" -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
INSTALLED="$(cd /tmp && "$PY" -c 'import vllm; print(vllm.__version__)')"
[ "$INSTALLED" = "$VLLM_VERSION" ] || { echo "!! installed vllm $INSTALLED != $VLLM_VERSION"; exit 1; }
for f in config/cache.py engine/arg_utils.py v1/worker/gpu_model_runner.py \
         v1/attention/backends/flash_attn.py v1/attention/backends/triton_attn.py \
         v1/engine/core.py v1/worker/gpu/model_runner.py; do
  test -f "$VLLM_DIR/$f" || { echo "!! installed vLLM lacks $f"; exit 1; }
  cp "$ROOT/vllm/$f" "$VLLM_DIR/$f"
done
rm -rf "$VLLM_DIR/kv_evict"; mkdir -p "$VLLM_DIR/kv_evict"
cp "$ROOT"/vllm/kv_evict/*.py "$VLLM_DIR/kv_evict/"
( cd /tmp && "$PY" - <<'PY'
import os, vllm.kv_evict.integration as m
root = os.path.dirname(os.path.dirname(m.__file__))
for f in ("v1/attention/backends/flash_attn.py", "v1/attention/backends/triton_attn.py"):
    src = open(os.path.join(root, f)).read()
    assert "_skive_can_compact" in src and "layer._skive_q" in src, f"{f}: SKIVE hooks missing"
assert "kv_evict_enabled" in open(os.path.join(root, "config/cache.py")).read()
print("SKIVE overlay verified ->", root)
PY
)

if [ "${SKIP_TESTS:-0}" != "1" ]; then
  echo "== 4/4 SKIVE unit tests =="
  ( cd "$ROOT" && "$PY" -m pytest skive/tests -q -p no:cacheprovider 2>&1 | tail -3 )
else
  echo "== 4/4 unit tests skipped =="
fi
echo
echo "Done. Next:  source $VENV/bin/activate && bash skive/run_smoke.sh   (or bash skive/run_aime24.sh)"
