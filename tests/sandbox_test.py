"""Sandbox-integrity test for the worker_context / thread-local Protector.

Sends two traces in parallel against a single NDIF deployment:

  A. Benign trace — captures a save as usual.
  B. Malicious trace — attempts ``import os`` inside the intervention
     body. The compiled intervention function, when executed inside the
     mediator worker thread, must hit the TLS-keyed whitelist and
     raise ImportError (or equivalent).

Pass conditions:
  * Benign trace returns saves with the expected shape.
  * Malicious trace raises; the ray container's worker log shows no
    spurious ImportError on internal ray/pydantic/gc imports (sandbox
    only kicks in on the worker threads, not elsewhere).
"""
from __future__ import annotations

import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import nnsight

MODEL_KEY = "Qwen/Qwen2.5-7B-Instruct"


def run_benign(idx: int):
    model = nnsight.LanguageModel(MODEL_KEY)
    with model.trace("The capital of France is", remote=True):
        h0 = model.model.layers[0].output.save()
    val = h0[0] if isinstance(h0, (tuple, list)) else h0
    return {"idx": idx, "kind": "benign", "shape": tuple(val.shape)}


def run_malicious(idx: int):
    model = nnsight.LanguageModel(MODEL_KEY)
    with model.trace("Test prompt", remote=True):
        # Captured into the intervention function's source; runs inside
        # the mediator worker thread on the server, where the Protector
        # scope is active.
        import os as _leaked_os  # noqa: F401
        _leaked_os.system("echo pwn")  # should never execute
        h0 = model.model.layers[0].output.save()
    val = h0[0] if isinstance(h0, (tuple, list)) else h0
    return {"idx": idx, "kind": "malicious_escaped", "shape": tuple(val.shape)}


def main():
    nnsight.CONFIG.set_default_api_key("api key")
    nnsight.CONFIG.API.HOST = "http://localhost:5001"
    nnsight.CONFIG.API.COMPRESS = False

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fb = pool.submit(run_benign, 0)
        fm = pool.submit(run_malicious, 1)
        outcomes = {}
        for fut, name in ((fb, "benign"), (fm, "malicious")):
            try:
                outcomes[name] = ("ok", fut.result())
            except Exception as e:
                outcomes[name] = ("err", repr(e), traceback.format_exc())
    wall = time.time() - t_start

    print(f"\n=== sandbox test ({wall:.1f}s wall) ===", flush=True)

    status, payload, *rest = outcomes["benign"] + (None,)
    if status == "ok":
        print(f"  BENIGN    : OK  shape={payload['shape']}")
    else:
        print(f"  BENIGN    : FAIL  {payload}")
        print("    (benign trace must succeed — sandbox broke something)")

    status, payload, *rest = outcomes["malicious"] + (None,)
    if status == "err":
        tb = rest[0] if rest else ""
        whitelist_hit = (
            "not whitelisted" in str(payload).lower()
            or "importerror" in str(payload).lower()
        )
        print(f"  MALICIOUS : BLOCKED ({'whitelist' if whitelist_hit else 'other error'})")
        print(f"    error: {payload}")
        if not whitelist_hit:
            print("    (expected ImportError / 'not whitelisted'; check tb)")
            print(tb)
    else:
        print(f"  MALICIOUS : ESCAPED the sandbox!  payload={payload}")
        print("    SECURITY REGRESSION — malicious trace returned normally")
        return 1

    return 0 if outcomes["benign"][0] == "ok" and outcomes["malicious"][0] == "err" else 1


if __name__ == "__main__":
    sys.exit(main())
