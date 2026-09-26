"""Make the vendored SKIVE package importable as `kv_evict` for the unit tests.

In this fork the package lives at vllm/kv_evict (so vLLM imports it as
vllm.kv_evict); the tests import it directly, without vLLM, so the vllm/
directory itself goes on sys.path. Nothing here touches the installed vLLM.
"""
import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for p in (os.path.join(_ROOT, "vllm"), os.path.join(_ROOT, "skive")):
    if p not in sys.path:
        sys.path.insert(0, p)
