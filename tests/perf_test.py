"""End-to-end throughput harness.

Workload: ``N_CLIENTS`` independent user processes, each sending
``M_REQUESTS`` traces back-to-back against the NDIF server. Each client
is a fresh spawn()'d Python process that builds its own
``LanguageModel`` once and loops. Server mode (batched vs single-in-
flight) is controlled by the actor's ``NDIF_MAX_CONCURRENCY`` at the
server side — this test harness itself is agnostic to it.

Env vars (client):
  MODEL_KEY      — HF repo id (default Qwen/Qwen3-32B)
  N_CLIENTS      — concurrent user processes (default 4)
  M_REQUESTS     — requests per client (default 10)
  LABEL          — tag saved with results (default inferred from server mode)
  RESULTS_DIR    — where to write JSON (default tests/results/)

Results are saved as
  ``tests/results/perf_<YYYYMMDD>_<HHMMSS>_<label>_N<N>_M<M>.json``
"""
from __future__ import annotations

import datetime as _dt
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path

import nnsight

MODEL_KEY = os.environ.get("MODEL_KEY", "Qwen/Qwen3-32B")
N_CLIENTS = int(os.environ.get("N_CLIENTS", "4"))
M_REQUESTS = int(os.environ.get("M_REQUESTS", "10"))
LABEL = os.environ.get("LABEL", "run")
RESULTS_DIR = Path(os.environ.get(
    "RESULTS_DIR",
    Path(__file__).parent / "results",
))

PROMPTS = [
    "The capital of France is",
    "In computer science, a binary tree is",
    "Photosynthesis is the process by which",
    "The Pythagorean theorem states",
    "Shakespeare's most famous play is",
    "The speed of light in a vacuum is",
    "Machine learning is a subset of",
    "The largest planet in our solar system is",
    "The French Revolution began in",
    "Quantum entanglement refers to",
    "The theory of relativity was proposed by",
    "Deep learning differs from shallow ML because",
    "The mitochondrion is known as",
    "Water boils at a temperature of",
    "The prime factorization of 360 is",
    "Neural networks learn representations by",
]


def _configure_client():
    nnsight.CONFIG.set_default_api_key("api key")
    nnsight.CONFIG.API.HOST = os.environ.get("NDIF_HOST", "http://localhost:5001")
    nnsight.CONFIG.API.COMPRESS = False


def _trace_once(model, prompt):
    t0 = time.time()
    with model.trace(prompt, remote=True):
        h0 = model.model.layers[0].output.save()
        last = model.model.layers[-1].output.save()
    dt = time.time() - t0
    h0_val = h0[0] if isinstance(h0, (tuple, list)) else h0
    last_val = last[0] if isinstance(last, (tuple, list)) else last
    return dt, tuple(h0_val.shape), tuple(last_val.shape)


def _client_worker(args):
    """Runs in a child process: build one LanguageModel, send M traces.

    Wraps trace exceptions in RuntimeError(str) so they survive pickling
    back to the parent (nnsight.NNsightException isn't picklable across
    spawn)."""
    client_idx, m_requests, prompts_for_client = args

    _configure_client()
    t_init_start = time.time()
    model = nnsight.LanguageModel(MODEL_KEY)
    t_init = time.time() - t_init_start

    per_req = []
    errors = []
    t_first = time.time()
    for req_idx, prompt in enumerate(prompts_for_client):
        try:
            dt, h0_shape, last_shape = _trace_once(model, prompt)
        except Exception as e:
            errors.append({"req_idx": req_idx, "error": repr(e)[:400]})
            per_req.append({"req_idx": req_idx, "dt_s": -1.0,
                             "h0_shape": None, "last_shape": None})
            continue
        per_req.append({
            "req_idx": req_idx,
            "dt_s": dt,
            "h0_shape": list(h0_shape),
            "last_shape": list(last_shape),
        })
    t_last = time.time()

    return {
        "client_idx": client_idx,
        "init_s": t_init,
        "client_span_s": t_last - t_first,
        "per_req": per_req,
        "errors": errors,
    }


def run_matrix():
    ctx = mp.get_context("spawn")

    # Round-robin prompts across clients' request slots so prompts vary
    # across the run while staying deterministic.
    def prompts_for_client(ci):
        return [PROMPTS[(ci * M_REQUESTS + i) % len(PROMPTS)] for i in range(M_REQUESTS)]

    args = [(ci, M_REQUESTS, prompts_for_client(ci)) for ci in range(N_CLIENTS)]

    t_start = time.time()
    with ctx.Pool(N_CLIENTS) as pool:
        client_results = pool.map(_client_worker, args)
    wall = time.time() - t_start
    return wall, client_results


def main():
    print(
        f"[{LABEL}] N_CLIENTS={N_CLIENTS} × M_REQUESTS={M_REQUESTS} "
        f"= {N_CLIENTS * M_REQUESTS} total requests against {MODEL_KEY}",
        flush=True,
    )

    wall, client_results = run_matrix()

    total_requests = N_CLIENTS * M_REQUESTS
    all_dts = [r["dt_s"] for cr in client_results
               for r in cr["per_req"] if r["dt_s"] > 0]
    init_ss = [cr["init_s"] for cr in client_results]
    all_errors = [e for cr in client_results for e in cr.get("errors", [])]

    print(f"\n=== PERF [{LABEL}] N_CLIENTS={N_CLIENTS} M_REQUESTS={M_REQUESTS} "
          f"{MODEL_KEY} ===")
    print(f"  wall            : {wall:.2f}s")
    print(f"  total requests  : {total_requests} "
          f"(completed={len(all_dts)}, errored={len(all_errors)})")
    print(f"  throughput      : {len(all_dts) / wall:.2f} req/s "
          f"(completed-only)")
    if all_dts:
        print(f"  per-req dt (s)  : min={min(all_dts):.2f} "
              f"median={statistics.median(all_dts):.2f} "
              f"mean={statistics.mean(all_dts):.2f} "
              f"max={max(all_dts):.2f} "
              f"stdev={statistics.stdev(all_dts):.2f}")
    print(f"  per-client init : min={min(init_ss):.2f} max={max(init_ss):.2f} "
          f"mean={statistics.mean(init_ss):.2f}")
    if all_errors:
        print(f"  first error     : {all_errors[0]['error']}")
    for cr in client_results:
        spans = [r["dt_s"] for r in cr["per_req"]]
        print(f"    client {cr['client_idx']} init={cr['init_s']:.2f}s "
              f"span={cr['client_span_s']:.2f}s "
              f"per-req mean={statistics.mean(spans):.2f}s")
    print("=" * 68, flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / (
        f"perf_{timestamp}_{LABEL}_N{N_CLIENTS}_M{M_REQUESTS}.json"
    )
    payload = {
        "label": LABEL,
        "model_key": MODEL_KEY,
        "n_clients": N_CLIENTS,
        "m_requests": M_REQUESTS,
        "total_requests": total_requests,
        "completed_requests": len(all_dts),
        "errored_requests": len(all_errors),
        "errors": all_errors,
        "wall_s": wall,
        "throughput_req_s": len(all_dts) / wall,
        "per_req_dt_s": {
            "min": min(all_dts),
            "median": statistics.median(all_dts),
            "mean": statistics.mean(all_dts),
            "max": max(all_dts),
            "stdev": statistics.stdev(all_dts),
        },
        "per_client_init_s": {
            "min": min(init_ss),
            "max": max(init_ss),
            "mean": statistics.mean(init_ss),
        },
        "clients": client_results,
        "timestamp": timestamp,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"saved: {out_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
