# TODO: attribute writes on protected modules under batching

Status: **open design issue, not scheduled**. Current posture: writes refused.

## Background

`ProtectedObject.__setattr__` in `src/services/ray/src/ray/nn/security/protected_objects.py`
wraps every loaded `torch.nn.Module` before user intervention code sees it.
Upstream dev (post-March security overhaul) added a **mutate-then-rollback**
pattern:

```python
def __setattr__(self, name, value):
    ...
    SET_ATTRS[id(self)][name] = getattr(PROTECTIONS[id(self)], name)
    PROTECTIONS[id(self)].__dict__[name] = value     # real module mutated
```

paired with `clear_set_attrs()` called at end of request to revert.

That design is safe when requests are serialized on the actor, which is
upstream's current operating mode. **It is unsafe under the batched
execution path introduced in this branch.** Our branch refuses the
write:

```python
def __setattr__(self, name, value):
    if not protected(self):
        object.__setattr__(self, name, value)
    else:
        raise AttributeError(
            f"Attribute '{name}' cannot be set after initialization"
        )
```

This is the minimal safe default while batching is in scope. The rest of
this doc describes why the upstream design breaks under batching and what
a proper fix looks like when we're ready to build it.

The `SET_ATTRS` dict and `clear_set_attrs()` helper are **kept in tree**
but dormant, gated behind a `_exclusive_mode` `threading.Event`. Default
state: event cleared, writes refused. When the quiesce-then-exclusive
coordinator described below ships, enabling writes is just a matter of
setting the event after the batch drain completes. The rollback indexing
bug that made upstream's `clear_set_attrs` unreliable (keyed by raw
module id instead of wrapper id) is fixed in this branch.

## Why mutate-then-rollback breaks batching

`VanillaBatchServer` fuses N concurrent requests into a single forward
pass at `batch_size=N`. All N users' mediators run on that shared
forward. If user A does `module.weight = attack_tensor`:

1. The real shared module's `__dict__` is mutated synchronously.
2. The fused forward (already running or about to run) reads that mutated
   weight.
3. **All N−1 co-batched users get saves computed from the attacked
   weights** — a cross-request integrity violation.
4. `clear_set_attrs()` runs only at end of request, which is after the
   shared forward already consumed the bad state.

There is no point in the batched pipeline where the rollback can land
that preserves isolation for concurrent users while keeping the shared
forward. "Serialize the write-request" collapses the batch to bs=1,
defeating the purpose. "Run user A in an overlay dict" makes the forward
not see the write, making the semantic empty.

`SET_ATTRS` is also not attribute-selective: writes to
`module.training`, `module._forward_hooks`, or arbitrary scalar config
attributes are all mutated on the real module and visible to concurrent
users until rollback.

## Current policy (enforced)

- `ProtectedObject.__setattr__` raises `AttributeError` on any write to
  a protected module after `__init__` **unless `_exclusive_mode` is set**.
- User intervention code that tries `module.some_attr = value` will fail
  loudly at the point of attempted write.
- `SET_ATTRS` stays empty by default (nothing is ever mutated), so
  `clear_set_attrs()` is a no-op on the refuse path.

This is strictly safer than upstream's mutate-then-rollback when
concurrent users share an actor. Cost is a slightly smaller
intervention API surface: users cannot patch module attrs for the
duration of a trace via direct assignment.

## Design sketch for a future "quiesce then exclusive" mode

If the feature becomes needed, the batching-safe way to allow writes is
an **exclusive-mode escape hatch** that detects write-intent, drains the
current batch, runs the write-request alone, then resumes batching.

### Detection

`__setattr__` becomes mode-aware:

```python
def __setattr__(self, name, value):
    if not protected(self):
        object.__setattr__(self, name, value)
        return
    if not _exclusive_mode.is_set():
        raise RequiresExclusiveError(self, name, value)
    # exclusive mode: now safe to mutate + track for rollback
    SET_ATTRS[id(self)][name] = getattr(PROTECTIONS[id(self)], name)
    PROTECTIONS[id(self)].__dict__[name] = value
```

`_exclusive_mode` is an actor-level `threading.Event` set only while
the actor runs one request with nothing else in `_active`.

### Propagation

`VanillaBatchServer._generation_loop` currently has a catch-all that
errors the entire batch on any exception. It needs per-mediator error
scoping so that one mediator raising `RequiresExclusiveError` finalizes
only that request's future with a sentinel, leaving the rest of the
batch's saves intact. This is the same nnsight-side fix also needed to
prevent one user's bug from killing co-batched peers generally.

### Actor-side coordinator

```python
class BaseModelDeployment:
    def __init__(...):
        self._batch_gate = asyncio.Event()    # cleared during exclusive run
        self._batch_gate.set()
        self._in_flight: set[str] = set()

    async def execute_batched(self, request, request_model):
        await self._batch_gate.wait()
        self._in_flight.add(request.id)
        try:
            saves_list = await ... normal batched path ...
        finally:
            self._in_flight.discard(request.id)

        if any(s.get("__needs_exclusive__") for s in saves_list):
            return await self._run_exclusive(request, request_model)
        return merge(saves_list)

    async def _run_exclusive(self, request, request_model):
        self._batch_gate.clear()                        # block new submits
        while self._in_flight:                          # drain
            await asyncio.sleep(0.01)
        try:
            _exclusive_mode.set()
            saves_list = await ... solo batched path ...
        finally:
            clear_set_attrs()
            _exclusive_mode.clear()
            self._batch_gate.set()
        return merge(saves_list)
```

### Correctness properties

- No cross-request contamination. The real module is never mutated
  while any other request is in `_active`.
- Aborted batched reads produced by the user before the failed
  `__setattr__` are computed on the un-mutated forward — co-batched
  users' saves remain valid.
- Rollback runs in `finally` — restoration guaranteed even on exception.
- Fairness: a write-request serializes itself but doesn't starve
  subsequent reads — when `_batch_gate` is set again, batching resumes.

### Costs

- Wasted work on the first (batched) run of a write-request up to the
  failing `__setattr__`. Acceptable if the write path is rare; bad if
  a large fraction of requests write.
- Priority inversion: write-request waits for the current batch's
  slowest peer to finish, plus any requests submitted ahead of it.
  Bounded by `mediator_timeout` so cannot stall forever.
- Latency: write-requests pay one full solo forward plus the drain
  wait. Expected 2–5× slower than batched reads.

### Dependency ordering

This feature depends on the **per-mediator error scoping fix in
`VanillaBatchServer`** (nnsight side) that's also required for the
co-batched error scoping bug noted in the session summary. Scheduling:

1. First: nnsight fix so one mediator's exception doesn't tank peers.
2. Then: `_exclusive_mode` + drain + retry on the actor side.
3. Gate behind `NDIF_ALLOW_EXCLUSIVE_WRITES=1` for rollout.

## When we should revisit

- User complaints about interventions that legitimately need to patch
  module attrs (e.g. swapping in a LoRA adapter mid-trace, patching
  `training=True`).
- Analytics showing a meaningful fraction of traces hit `__setattr__`.
- nnsight's `VanillaBatchServer` per-mediator error scoping lands.

Until then: refuse writes, keep the isolation guarantee trivial.

## Related code

- `src/services/ray/src/ray/nn/security/protected_objects.py` —
  `ProtectedObject.__setattr__` enforces the policy.
- `src/services/ray/src/ray/deployments/modeling/base.py` —
  `execute_batched` is where the drain/exclusive coordinator would
  attach if this ever ships.
- nnsight `src/nnsight/modeling/hf_serve/vanilla_server.py` —
  `_generation_loop`'s catch-all error path needs per-mediator scoping
  before this is implementable.
