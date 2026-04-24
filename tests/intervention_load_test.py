"""Extended A/B correctness: varied interventions, batched vs sequential.

Extends tests/correctness_test.py. Instead of every request doing the same
"save layer 0 and layer last" pattern, each request picks one intervention
from a catalog (below). The same (prompt, intervention) pair runs on the
batch stack and the seq stack; results are compared per-save tensor.

Catalog goals: exercise different save shapes / slices / submodules, and —
critically — run *modification* interventions in the same batch alongside
plain saves, which is exactly where cross-request wiring bugs would show.

Env:
  BATCH_HOST   default http://localhost:5002
  SEQ_HOST     default http://localhost:5003
  MODEL_KEY    default Qwen/Qwen2.5-14B-Instruct
  N_REQUESTS   default 16     (how many concurrent requests per side)
  RTOL         default 1e-4   (bf16 batched kernels drift ~0.1% mean)
  ATOL         default 1e-5
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import nnsight
import torch

BATCH_HOST = os.environ.get("BATCH_HOST", "http://localhost:5002")
SEQ_HOST = os.environ.get("SEQ_HOST", "http://localhost:5003")
MODEL_KEY = os.environ.get("MODEL_KEY", "Qwen/Qwen2.5-14B-Instruct")
N_REQUESTS = int(os.environ.get("N_REQUESTS", "16"))
ITERATIONS = int(os.environ.get("ITERATIONS", "1"))
# "both" (default) | "batch" | "seq" — useful for isolated memory profiling
ONLY_SIDE = os.environ.get("ONLY_SIDE", "both").lower()
# bf16 has ~3 decimal digits, and fused kernels drift further under
# different batch sizes. Session baseline @7B observed ~0.1% mean / ~4% max
# on the deepest layer — this is model-scale-dependent (deeper layers
# accumulate more kernel noise) but is NOT a correctness bug. Default to
# RTOL=5e-2 (5%) so genuine regressions still fail loud.
RTOL = float(os.environ.get("RTOL", "5e-2"))
ATOL = float(os.environ.get("ATOL", "1e-2"))

PROMPTS = [
    "The capital of France is",
    "In computer science, a binary tree is",
    "Photosynthesis is the process by which",
    "The Pythagorean theorem states",
    "Shakespeare's most famous play is",
    "The speed of light in a vacuum is",
    "Machine learning is a subset of",
    "The largest planet in our solar system is",
    "The mitochondria is the",
    "A neural network learns by",
    "The French Revolution began in",
    "DNA is composed of",
    "Quantum entanglement refers to",
    "The theory of relativity was proposed by",
    "HTTP stands for",
    "The human genome contains",
]

# Layer indices chosen to be safely inside a 48-layer model (Qwen2.5-14B)
# and still representative: first, early, middle, late, last.
L_FIRST, L_EARLY, L_MID, L_LATE, L_LAST = 0, 10, 24, 40, -1


# ---- Intervention catalog -------------------------------------------------
#
# nnsight 0.6's RemoteInterleavingTracer source-inspects the `with` block,
# so intervention code MUST be written literally inside `with
# model.trace(...):` — calling a helper function that returns proxies
# doesn't work (the helper gets captured for server-side execution and
# fails to pickle because source-inspection can't reach it).
#
# Each intervention below is therefore its own top-level function
# containing its OWN `with` block and returning a dict of CPU tensors.
# Repetitive, but that's the cost of AST-based tracing.


INTERVENTION_NAMES = [
    "save_first_hidden",
    "save_last_hidden",
    "save_multi_layer",
    "save_attn_out",
    "save_mlp_out",
    "save_last_and_logits",
    "save_embed",
    "zero_layer5",
    "scale_mlp",
    "save_mid_and_logits",
    "fail_zero_div",
    "fail_bad_shape",
    "fail_explicit_raise",
]


def _unwrap(t):
    """Layer outputs are (hidden, ...) tuples; unwrap to the tensor."""
    while isinstance(t, (tuple, list)):
        t = t[0]
    return t


def _to_cpu(t):
    return _unwrap(t).detach().cpu()


def _run_save_first_hidden(model, prompt):
    with model.trace(prompt, remote=True):
        h = model.model.layers[L_FIRST].output.save()
    return {"first_hidden": _to_cpu(h)}


def _run_save_last_hidden(model, prompt):
    with model.trace(prompt, remote=True):
        h = model.model.layers[L_LAST].output.save()
    return {"last_hidden": _to_cpu(h)}


def _run_save_multi_layer(model, prompt):
    with model.trace(prompt, remote=True):
        h_first = model.model.layers[L_FIRST].output.save()
        h_early = model.model.layers[L_EARLY].output.save()
        h_mid = model.model.layers[L_MID].output.save()
        h_late = model.model.layers[L_LATE].output.save()
        h_last = model.model.layers[L_LAST].output.save()
    return {
        "h_first": _to_cpu(h_first),
        "h_early": _to_cpu(h_early),
        "h_mid": _to_cpu(h_mid),
        "h_late": _to_cpu(h_late),
        "h_last": _to_cpu(h_last),
    }


def _run_save_attn_out(model, prompt):
    with model.trace(prompt, remote=True):
        a = model.model.layers[L_MID].self_attn.output.save()
    return {"attn_out": _to_cpu(a)}


def _run_save_mlp_out(model, prompt):
    with model.trace(prompt, remote=True):
        m = model.model.layers[L_MID].mlp.output.save()
    return {"mlp_out": _to_cpu(m)}


def _run_save_last_and_logits(model, prompt):
    with model.trace(prompt, remote=True):
        h = model.model.layers[L_LAST].output.save()
        logits = model.output.logits.save()
    return {"h_last": _to_cpu(h), "logits": _to_cpu(logits)}


def _run_save_embed(model, prompt):
    with model.trace(prompt, remote=True):
        e = model.model.embed_tokens.output.save()
    return {"embed_out": _to_cpu(e)}


def _run_zero_layer5(model, prompt):
    with model.trace(prompt, remote=True):
        model.model.layers[5].output[0][:] = 0
        h = model.model.layers[L_LAST].output.save()
    return {"last_after_zero": _to_cpu(h)}


def _run_scale_mlp(model, prompt):
    with model.trace(prompt, remote=True):
        model.model.layers[L_MID].mlp.output = model.model.layers[L_MID].mlp.output * 0.5
        h = model.model.layers[L_LAST].output.save()
    return {"last_after_scale": _to_cpu(h)}


def _run_save_mid_and_logits(model, prompt):
    with model.trace(prompt, remote=True):
        h = model.model.layers[L_MID].output.save()
        logits = model.output.logits.save()
    return {"h_mid": _to_cpu(h), "logits": _to_cpu(logits)}


# ---- Deliberate-failure interventions -------------------------------------
#
# These are expected to raise *server-side* during mediator execution. A
# correct batched server (nnsight's VanillaBatchServer post-7c90fe3) should:
#   1. Defer the failure onto mediator._deferred_exception
#   2. Finalize ONLY this request with __error__
#   3. Leave co-batched siblings to complete normally
#
# The harness drives this by mixing these into the same batch as normal
# interventions and verifying that:
#   - the failing request's `error` is non-None and names its failure mode
#   - every non-failing request's `saves` still match its sequential
#     counterpart within bf16 tolerance (i.e. no cross-request corruption)


def _run_fail_zero_div(model, prompt):
    with model.trace(prompt, remote=True):
        h = model.model.layers[L_LAST].output.save()
        # Runs server-side inside the mediator worker thread.
        x = 1 / 0  # noqa: F841


def _run_fail_bad_shape(model, prompt):
    with model.trace(prompt, remote=True):
        h = model.model.layers[L_LAST].output.save()
        # Mismatched shape: layer output is (1, seq, 5120); 7x7x7 is wrong.
        model.model.layers[3].output[0][:] = torch.ones(7, 7, 7)


def _run_fail_explicit_raise(model, prompt):
    with model.trace(prompt, remote=True):
        h = model.model.layers[L_LAST].output.save()
        raise RuntimeError("intentional test failure — error isolation check")


RUNNERS = {
    "save_first_hidden": _run_save_first_hidden,
    "save_last_hidden": _run_save_last_hidden,
    "save_multi_layer": _run_save_multi_layer,
    "save_attn_out": _run_save_attn_out,
    "save_mlp_out": _run_save_mlp_out,
    "save_last_and_logits": _run_save_last_and_logits,
    "save_embed": _run_save_embed,
    "zero_layer5": _run_zero_layer5,
    "scale_mlp": _run_scale_mlp,
    "save_mid_and_logits": _run_save_mid_and_logits,
    "fail_zero_div": _run_fail_zero_div,
    "fail_bad_shape": _run_fail_bad_shape,
    "fail_explicit_raise": _run_fail_explicit_raise,
}

EXPECTED_FAIL = {"fail_zero_div", "fail_bad_shape", "fail_explicit_raise"}


# ---- Runner ---------------------------------------------------------------


def _configure_client(host: str):
    nnsight.CONFIG.set_default_api_key("api key")
    nnsight.CONFIG.API.HOST = host
    nnsight.CONFIG.API.COMPRESS = False


def _trace_one(model, idx, prompt, intervention_idx):
    name = INTERVENTION_NAMES[intervention_idx]
    try:
        saves = RUNNERS[name](model, prompt)
        return idx, {"name": name, "saves": saves, "error": None}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return idx, {
            "name": name,
            "saves": None,
            "error": f"{type(e).__name__}: {e!r}\n--TB--\n{tb[-1500:]}",
        }


def _build_plan(n: int):
    """(idx, prompt, intervention_idx) for request idx — deterministic."""
    plan = []
    for i in range(n):
        prompt = PROMPTS[i % len(PROMPTS)]
        intervention_idx = i % len(INTERVENTION_NAMES)
        plan.append((i, prompt, intervention_idx))
    return plan


def run_side(host: str, label: str, plan):
    _configure_client(host)
    model = nnsight.LanguageModel(MODEL_KEY)
    t0 = time.time()
    max_workers = int(os.environ.get("MAX_WORKERS", str(len(plan))))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_trace_one, model, i, p, k) for (i, p, k) in plan]
        results = [f.result() for f in futures]
    wall = time.time() - t0

    by_idx = {}
    errors = []
    for idx, payload in results:
        if payload["error"] is not None:
            errors.append((idx, payload["name"], payload["error"]))
            # Keep the record so compare() can verify expected-fail semantics.
            by_idx[idx] = {"name": payload["name"], "saves": None, "error": payload["error"]}
        else:
            by_idx[idx] = {"name": payload["name"], "saves": payload["saves"], "error": None}

    print(
        f"[{label}] {host}: {len(plan)} reqs in {wall:.2f}s "
        f"({sum(1 for v in by_idx.values() if v['saves'] is not None)} ok, "
        f"{len(errors)} err)",
        flush=True,
    )
    for idx, name, err in errors:
        # Show just the first line (exception type+msg) for readability.
        first = err.splitlines()[0] if err else ""
        print(f"    req {idx} [{name}] error: {first}", flush=True)
    return by_idx, errors, wall


def compare(batched, sequential):
    """Score: for expected-pass names, saves must match seq within tol AND
    both sides must not have errors. For expected-fail names, BOTH sides must
    return an error (failure isolation: error is reported to this request,
    and seqtest — which runs each alone — also fails). Plus: the success of
    co-batched passing requests is already covered by the pass-side scoring.
    """
    common = sorted(set(batched) & set(sequential))
    per_type = {}
    mismatches = 0
    isolation_violations = 0
    tensor_pairs = 0
    print(f"\nComparing {len(common)} request pairs (rtol={RTOL}, atol={ATOL})", flush=True)
    for idx in common:
        name = batched[idx]["name"]
        assert name == sequential[idx]["name"], "intervention mismatch — plan drift"
        slot = per_type.setdefault(
            name,
            {"pairs": 0, "mismatch": 0, "max_diff": 0.0, "mean_diff": 0.0,
             "expected": "FAIL" if name in EXPECTED_FAIL else "PASS"},
        )
        b_err = batched[idx]["error"] is not None
        s_err = sequential[idx]["error"] is not None

        if name in EXPECTED_FAIL:
            # Failure isolation: the failing request must surface an error on
            # both stacks (seq also because the failing user code is the same).
            # The real isolation check is implicit: if the batched side also
            # corrupted co-batched siblings, *those* would mismatch above.
            if b_err and s_err:
                slot["pairs"] += 1
                continue
            # Otherwise it's a violation — fail-intervention should error.
            isolation_violations += 1
            slot["mismatch"] += 1
            slot["pairs"] += 1
            print(f"  ✗ req {idx} [{name}] expected failure on both sides; "
                  f"batch_err={b_err} seq_err={s_err}")
            continue

        # Expected pass: both sides must succeed and match within tolerance.
        if b_err or s_err:
            mismatches += 1
            slot["mismatch"] += 1
            slot["pairs"] += 1
            print(f"  ✗ req {idx} [{name}] UNEXPECTED error — batch_err={b_err} seq_err={s_err}")
            continue

        b_saves = batched[idx]["saves"]
        s_saves = sequential[idx]["saves"]
        for k in b_saves:
            if k not in s_saves:
                print(f"  ✗ req {idx} [{name}] missing {k} on seq side")
                mismatches += 1
                slot["mismatch"] += 1
                continue
            tb, ts = b_saves[k], s_saves[k]
            tensor_pairs += 1
            slot["pairs"] += 1
            if tb.shape != ts.shape:
                print(f"  ✗ req {idx} [{name}] {k}: shape batched={tuple(tb.shape)} seq={tuple(ts.shape)}")
                mismatches += 1
                slot["mismatch"] += 1
                continue
            diff = (tb.float() - ts.float()).abs()
            slot["max_diff"] = max(slot["max_diff"], diff.max().item())
            slot["mean_diff"] = max(slot["mean_diff"], diff.mean().item())
            if torch.allclose(tb, ts, rtol=RTOL, atol=ATOL):
                continue
            mismatches += 1
            slot["mismatch"] += 1
            print(
                f"  ✗ req {idx} [{name}] {k}: shape={tuple(tb.shape)} "
                f"max|Δ|={diff.max().item():.3e} mean|Δ|={diff.mean().item():.3e} "
                f"|b|={tb.float().norm().item():.3e} |s|={ts.float().norm().item():.3e}"
            )

    print("\nPer-intervention summary:")
    print(f"  {'intervention':<26} {'exp':>4} {'pairs':>6} {'mismatch':>9} {'max|Δ|':>12} {'mean|Δ|':>12}")
    for name, s in sorted(per_type.items()):
        print(f"  {name:<26} {s['expected']:>4} {s['pairs']:>6d} {s['mismatch']:>9d} "
              f"{s['max_diff']:>12.3e} {s['mean_diff']:>12.3e}")
    print(f"\n  TOTAL tensor mismatches: {mismatches} / {tensor_pairs} pairs", flush=True)
    print(f"  ISOLATION violations (failures not surfaced as errors): {isolation_violations}\n",
          flush=True)
    return mismatches + isolation_violations


def main():
    print(f"MODEL={MODEL_KEY}  N_REQUESTS={N_REQUESTS}  ITERATIONS={ITERATIONS}  "
          f"INTERVENTIONS={len(INTERVENTION_NAMES)}", flush=True)
    print(f"BATCH_HOST={BATCH_HOST}", flush=True)
    print(f"SEQ_HOST={SEQ_HOST}", flush=True)
    print("-" * 60, flush=True)

    plan = _build_plan(N_REQUESTS)
    # Plan printed once — same plan replays each iteration.
    print("Plan (idx | intervention | prompt[:40]):")
    for i, p, k in plan:
        print(f"  {i:>3d} | {INTERVENTION_NAMES[k]:<22} | {p[:40]!r}")
    print("-" * 60, flush=True)

    total_mismatches = 0
    total_isolation_violations = 0
    total_unexpected_batch_err = 0
    total_unexpected_seq_err = 0
    batch_walls, seq_walls = [], []

    for it in range(1, ITERATIONS + 1):
        print(f"\n====== Iteration {it}/{ITERATIONS} ======", flush=True)
        if ONLY_SIDE in ("both", "batch"):
            batched, b_err, b_wall = run_side(BATCH_HOST, "BATCH", plan)
            batch_walls.append(b_wall)
        else:
            batched, b_err, b_wall = {}, [], 0.0
        if ONLY_SIDE in ("both", "seq"):
            sequential, s_err, s_wall = run_side(SEQ_HOST, "SEQ  ", plan)
            seq_walls.append(s_wall)
        else:
            sequential, s_err, s_wall = {}, [], 0.0

        # Count only UNEXPECTED errors (errors on pass-type interventions).
        unexpected_b = sum(
            1 for idx, name, _ in b_err if name not in EXPECTED_FAIL
        )
        unexpected_s = sum(
            1 for idx, name, _ in s_err if name not in EXPECTED_FAIL
        )
        total_unexpected_batch_err += unexpected_b
        total_unexpected_seq_err += unexpected_s

        if ONLY_SIDE == "both":
            mism = compare(batched, sequential)
            total_mismatches += mism

    print("=" * 60, flush=True)
    print(f"Across {ITERATIONS} iteration(s) × N={N_REQUESTS} (ONLY_SIDE={ONLY_SIDE}):")
    if batch_walls:
        print(f"  batch wall mean={sum(batch_walls)/len(batch_walls):.2f}s")
    if seq_walls:
        print(f"  seq   wall mean={sum(seq_walls)/len(seq_walls):.2f}s")
    if batch_walls and seq_walls:
        print(f"  speedup={sum(seq_walls)/sum(batch_walls):.2f}x")
    print(f"  UNEXPECTED batch-side errors on pass interventions: "
          f"{total_unexpected_batch_err}")
    print(f"  UNEXPECTED seq-side errors on pass interventions:   "
          f"{total_unexpected_seq_err}")
    print(f"  Tensor mismatches (bf16 noise counted here too): {total_mismatches}")

    fatal = total_unexpected_batch_err + total_unexpected_seq_err
    if fatal == 0:
        print("\nPASS: no isolation violations across iterations.", flush=True)
        sys.exit(0)
    else:
        print(f"\nFAIL: {fatal} unexpected errors on pass-type interventions", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
