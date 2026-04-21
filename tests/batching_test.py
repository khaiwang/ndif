"""Concurrent trace test against Llama-3.1-8B-Instruct.

Fires N traces in parallel threads against a single NDIF deployment so
the VanillaBatchServer inside the Ray actor has a chance to mix them
into one forward pass. Pass/fail is whether all returned saves match
the expected shape; the `[VanillaBatchServer] _step batch_size=...`
prints in the ray container logs are the actual batching evidence.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import nnsight

MODEL_KEY = "Qwen/Qwen2.5-7B-Instruct"
N_CONCURRENT = int(os.environ.get("N_CONCURRENT", "4"))

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


def run_one(idx: int, prompt: str):
    model = nnsight.LanguageModel(MODEL_KEY)
    t0 = time.time()
    with model.trace(prompt, remote=True):
        h0 = model.model.layers[0].output.save()
        last = model.model.layers[-1].output.save()
    dt = time.time() - t0
    h0_val = h0[0] if isinstance(h0, (tuple, list)) else h0
    last_val = last[0] if isinstance(last, (tuple, list)) else last
    return {
        "idx": idx,
        "prompt": prompt[:30],
        "dt": dt,
        "h0_shape": tuple(h0_val.shape),
        "last_shape": tuple(last_val.shape),
    }


def main():
    nnsight.CONFIG.set_default_api_key("api key")
    nnsight.CONFIG.API.HOST = "http://localhost:5001"
    nnsight.CONFIG.API.COMPRESS = False

    prompts = PROMPTS[:N_CONCURRENT]
    print(f"Firing {N_CONCURRENT} concurrent traces against {MODEL_KEY}", flush=True)

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=N_CONCURRENT) as pool:
        futures = {pool.submit(run_one, i, p): i for i, p in enumerate(prompts)}
        results = []
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"request failed: {e!r}", flush=True)
                raise
    wall = time.time() - t_start

    results.sort(key=lambda r: r["idx"])
    for r in results:
        print(
            f"  [{r['idx']}] prompt={r['prompt']!r:32}  "
            f"h0={r['h0_shape']}  last={r['last_shape']}  dt={r['dt']:.2f}s",
            flush=True,
        )
    print(f"\ntotal wall time: {wall:.2f}s over {N_CONCURRENT} concurrent requests", flush=True)
    print("Now grep 'VanillaBatchServer' in docker logs dev-ray-1 to confirm batching.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
