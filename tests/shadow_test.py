"""Per-name builtin shadow test.

NDIF's Protector shadows risky builtins (open, eval, exec, compile)
in the user intervention function's globals dict. Python name lookup
is locals → globals → builtins, so a global-level shadow intercepts
before the frame's bound builtins — the only per-frame technique that
works on already-compiled functions.

This test fires three traces concurrently:

  * BENIGN    — normal save, must return the expected shape.
  * open()    — intervention calls ``open("/etc/hostname")``. Must be
                blocked by the shadow with PermissionError.
  * eval()    — intervention calls ``eval("1+1")``. Must be blocked.

Pass conditions:
  * benign trace returns a (1, 1, 3584) hidden-state shape.
  * both malicious traces error with the sandbox marker in the
    traceback; the specific wording is "is not permitted in the
    sandboxed scope" from ``_make_blocker`` in
    ``protected_environment.py``.
"""
from __future__ import annotations

import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import nnsight

MODEL_KEY = "Qwen/Qwen2.5-7B-Instruct"
_SHADOW_MARKER = "not permitted in the sandboxed scope"


def run_benign(idx: int):
    model = nnsight.LanguageModel(MODEL_KEY)
    with model.trace("The capital of France is", remote=True):
        h0 = model.model.layers[0].output.save()
    val = h0[0] if isinstance(h0, (tuple, list)) else h0
    return {"idx": idx, "kind": "benign", "shape": tuple(val.shape)}


def run_open(idx: int):
    model = nnsight.LanguageModel(MODEL_KEY)
    with model.trace("Test prompt", remote=True):
        # Free-name ``open`` inside an intervention body. The compiled
        # function's globals carry the Protector's shadow; lookup beats
        # the real builtin.
        _data = open("/etc/hostname").read()  # should never execute
        h0 = model.model.layers[0].output.save()
    val = h0[0] if isinstance(h0, (tuple, list)) else h0
    return {"idx": idx, "kind": "open_escaped", "shape": tuple(val.shape)}


def run_eval(idx: int):
    model = nnsight.LanguageModel(MODEL_KEY)
    with model.trace("Test prompt", remote=True):
        _val = eval("1+1")  # should never execute
        h0 = model.model.layers[0].output.save()
    val = h0[0] if isinstance(h0, (tuple, list)) else h0
    return {"idx": idx, "kind": "eval_escaped", "shape": tuple(val.shape)}


def main():
    nnsight.CONFIG.set_default_api_key("api key")
    nnsight.CONFIG.API.HOST = "http://localhost:5001"
    nnsight.CONFIG.API.COMPRESS = False

    jobs = [
        ("benign", run_benign),
        ("open", run_open),
        ("eval", run_eval),
    ]

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {name: pool.submit(fn, i) for i, (name, fn) in enumerate(jobs)}
        outcomes: dict[str, tuple] = {}
        for name, fut in futures.items():
            try:
                outcomes[name] = ("ok", fut.result())
            except Exception as e:
                outcomes[name] = ("err", repr(e), traceback.format_exc())
    wall = time.time() - t_start

    print(f"\n=== shadow test ({wall:.1f}s wall) ===", flush=True)

    verdicts: list[tuple[str, bool]] = []

    def _assess_benign():
        status, *rest = outcomes["benign"]
        if status == "ok":
            payload = rest[0]
            if payload["shape"] == (1, 1, 3584):
                print(f"  BENIGN : OK  shape={payload['shape']}")
                return True
            print(f"  BENIGN : WRONG shape {payload['shape']}")
            return False
        print(f"  BENIGN : FAILED {rest[0]}")
        return False

    def _assess_blocked(name: str):
        status, *rest = outcomes[name]
        if status != "err":
            print(f"  {name.upper():7}: LEAKED — returned {rest[0]!r}")
            return False
        tb = rest[1] if len(rest) > 1 else ""
        if _SHADOW_MARKER in tb or _SHADOW_MARKER in str(rest[0]):
            print(f"  {name.upper():7}: BLOCKED by shadow")
            return True
        print(f"  {name.upper():7}: BLOCKED but wrong error — {rest[0]}")
        return False

    verdicts.append(("benign", _assess_benign()))
    verdicts.append(("open", _assess_blocked("open")))
    verdicts.append(("eval", _assess_blocked("eval")))

    all_pass = all(ok for _, ok in verdicts)
    print(f"\n=== SHADOW VERDICT: {'PASS' if all_pass else 'FAIL'} ===", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
