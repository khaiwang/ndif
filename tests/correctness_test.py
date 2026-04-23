"""Correctness A/B: batched vs sequential intervention results.

Fires the same set of intervention requests against two independent NDIF
stacks — one with ``NDIF_MAX_CONCURRENCY`` large (batching: multiple
concurrent ``__call__`` fuse into one VanillaBatchServer forward) and
one with ``NDIF_MAX_CONCURRENCY=1`` (single-in-flight, each request is
its own forward). For each (prompt, saved tensor) pair, compare the
batched result against the sequential result element-wise.

On the batched side the requests are submitted concurrently (so the
server actually has the chance to fuse them). On the sequential side
concurrency doesn't matter — the actor serializes anyway — so we submit
the same way for symmetry.

Passes if every tensor pair matches within tight fp tolerance.

Env vars:
  BATCH_HOST       — default http://localhost:5002
  SEQ_HOST         — default http://localhost:5003
  MODEL_KEY        — default Qwen/Qwen2.5-7B-Instruct
  N_REQUESTS       — number of concurrent requests per side (default 8)
  RTOL, ATOL       — tolerance for torch.allclose (default 1e-5, 1e-6)
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
import pickle
import base64
from pathlib import Path

import nnsight
import torch

BATCH_HOST = os.environ.get("BATCH_HOST", "http://localhost:5002")
SEQ_HOST = os.environ.get("SEQ_HOST", "http://localhost:5003")
MODEL_KEY = os.environ.get("MODEL_KEY", "Qwen/Qwen2.5-7B-Instruct")
N_REQUESTS = int(os.environ.get("N_REQUESTS", "8"))
RTOL = float(os.environ.get("RTOL", "1e-5"))
ATOL = float(os.environ.get("ATOL", "1e-6"))

# Deliberately varied prompts — different lengths, different token
# content — so any cross-request contamination shows up as mismatch.
PROMPTS = [
    "The capital of France is",
    "In computer science, a binary tree is",
    "Photosynthesis is the process by which",
    "The Pythagorean theorem states",
    "Shakespeare's most famous play is",
    "The speed of light in a vacuum is",
    "Machine learning is a subset of",
    "The largest planet in our solar system is",
]


def _configure_client(host: str):
    nnsight.CONFIG.set_default_api_key("api key")
    nnsight.CONFIG.API.HOST = host
    nnsight.CONFIG.API.COMPRESS = False


def _trace_one(args):
    """Run one intervention trace; return (idx, saves_dict_or_error).

    ``saves_dict`` is a dict keyed by layer-path with pickled bytes of
    the CPU tensor, so it survives multiprocessing pickling cleanly.
    """
    host, idx, prompt = args
    _configure_client(host)
    try:
        model = nnsight.LanguageModel(MODEL_KEY)
        with model.trace(prompt, remote=True):
            # Two saves per request so we exercise both early and late
            # layers. Bare output of a layer is typically a tuple; take
            # [0] for the hidden state.
            layer0 = model.model.layers[0].output.save()
            layer_last = model.model.layers[-1].output.save()
        l0 = layer0[0] if isinstance(layer0, (tuple, list)) else layer0
        ll = layer_last[0] if isinstance(layer_last, (tuple, list)) else layer_last
        saves = {
            "layer0": l0.detach().cpu(),
            "layer_last": ll.detach().cpu(),
        }
        return (idx, {"saves": _pickle(saves), "error": None})
    except Exception as e:
        return (idx, {"saves": None, "error": repr(e)[:400]})


def _pickle(obj) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def _unpickle(blob: bytes):
    return pickle.loads(blob)


def run_side(host: str, label: str) -> dict:
    """Run N_REQUESTS concurrently against one stack. Returns {idx: saves}."""
    ctx = mp.get_context("spawn")
    args = [(host, i, PROMPTS[i % len(PROMPTS)]) for i in range(N_REQUESTS)]

    t0 = time.time()
    with ctx.Pool(N_REQUESTS) as pool:
        results = pool.map(_trace_one, args)
    wall = time.time() - t0

    by_idx = {}
    errors = []
    for idx, payload in results:
        if payload["error"] is not None:
            errors.append((idx, payload["error"]))
        else:
            by_idx[idx] = _unpickle(payload["saves"])

    print(f"[{label}] {host}: {N_REQUESTS} reqs in {wall:.2f}s "
          f"({len(by_idx)} ok, {len(errors)} err)", flush=True)
    for idx, err in errors:
        print(f"    req {idx} error: {err}", flush=True)
    return by_idx, errors, wall


def compare(batched: dict, sequential: dict) -> int:
    """Element-wise compare matching (idx, save_name) pairs. Return mismatch count."""
    common = sorted(set(batched) & set(sequential))
    mismatches = 0
    print(f"\nComparing {len(common)} request pairs "
          f"(rtol={RTOL}, atol={ATOL})...", flush=True)
    for idx in common:
        for name in ("layer0", "layer_last"):
            t_b = batched[idx][name]
            t_s = sequential[idx][name]
            if t_b.shape != t_s.shape:
                print(f"  ✗ req {idx} {name}: shape mismatch "
                      f"batched={tuple(t_b.shape)} seq={tuple(t_s.shape)}")
                mismatches += 1
                continue
            if torch.allclose(t_b, t_s, rtol=RTOL, atol=ATOL):
                continue
            mismatches += 1
            diff = (t_b.float() - t_s.float()).abs()
            print(f"  ✗ req {idx} {name}: shape={tuple(t_b.shape)} "
                  f"max|Δ|={diff.max().item():.3e} "
                  f"mean|Δ|={diff.mean().item():.3e} "
                  f"batched_norm={t_b.float().norm().item():.3e} "
                  f"seq_norm={t_s.float().norm().item():.3e}")
    print(f"\nmismatches: {mismatches} / {2 * len(common)} tensor pairs", flush=True)
    return mismatches


def main():
    print(f"MODEL={MODEL_KEY}  N_REQUESTS={N_REQUESTS}", flush=True)
    print(f"BATCH_HOST={BATCH_HOST}", flush=True)
    print(f"SEQ_HOST={SEQ_HOST}", flush=True)
    print("-" * 60, flush=True)

    batched, b_err, b_wall = run_side(BATCH_HOST, "BATCH")
    sequential, s_err, s_wall = run_side(SEQ_HOST, "SEQ  ")

    if b_err or s_err:
        print(f"\nNOTE: {len(b_err)} batch errors, {len(s_err)} seq errors", flush=True)

    missing_batch = set(range(N_REQUESTS)) - set(batched)
    missing_seq = set(range(N_REQUESTS)) - set(sequential)
    if missing_batch or missing_seq:
        print(f"missing batch={sorted(missing_batch)} "
              f"seq={sorted(missing_seq)}", flush=True)

    mismatches = compare(batched, sequential)

    print("=" * 60, flush=True)
    if mismatches == 0 and not b_err and not s_err:
        print("PASS: batched and sequential results match.", flush=True)
        sys.exit(0)
    else:
        print(f"FAIL: mismatches={mismatches} batch_errors={len(b_err)} "
              f"seq_errors={len(s_err)}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
