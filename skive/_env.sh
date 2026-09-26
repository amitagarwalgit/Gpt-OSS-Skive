# Shared environment for the SKIVE run scripts (sourced, not executed).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -x "$ROOT/.venv/bin/python" ]; then
  export PATH="$ROOT/.venv/bin:$PATH"
fi
# Checked from /tmp on purpose: inside the repo, `import vllm` resolves to the
# source tree, which has no compiled kernels. The run scripts also cd to /tmp.
( cd /tmp && python -c "import vllm.kv_evict.integration" ) \
  || { echo "!! vLLM with the SKIVE overlay is not importable; run skive/install.sh"; exit 1; }
export VLLM_USE_V2_MODEL_RUNNER=0            # SKIVE's hooks live in the V1 runner
export VLLM_USE_FLASHINFER_SAMPLER=0         # the gpt-oss inference recipe
export VLLM_ALLOW_INSECURE_SERIALIZATION=1   # eviction counters cross the multiprocess engine
export TOKENIZERS_PARALLELISM=false
