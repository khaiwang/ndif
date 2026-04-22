# TODO: rebuild upstream's security defenses on a TLS-safe foundation

Status: **open follow-up PR**. Current branch ships TLS-safe import +
per-name builtin shadow only; the rest of upstream dev's six-layer
sandbox is intentionally inert in this branch (files
``protector.py`` / ``importer.py`` / ``guards.py`` /
``whitelist.py`` are present but not imported by ``__init__.py``).

## What this branch ships

- TLS-keyed ``__import__`` dispatcher (process-global hook installed
  once at module load; behavior gated on ``_tls.active``).
- Per-name globals shadow of ``{eval, exec, open, compile}`` in user
  intervention frames via ``_ProtectorScope.__enter__``.
- Module allowlist with deserialization variant.
- ``NEVER_ALLOWED`` hard-block set checked before the allowlist.

That's it. All concurrency-safe under the batched ``__call__`` path —
multiple mediator threads can enter their own Protector scopes
without stomping on each other.

## What upstream dev has that this branch deliberately omits

Upstream's ``Protector`` (in ``protector.py``) layers six defenses on
top of the import allowlist. All but layer 6 use
``nnsight.util.Patcher`` to mutate process-global state on
``__enter__`` and restore on ``__exit__``. Under our batched
execution path multiple Protector scopes can be active concurrently
on different mediator threads, and the save/restore dance against
shared state then races.

| # | Layer | Patches | Concurrency-safe today? | Can be made TLS-safe? |
|---|---|---|---|---|
| 1 | ``__import__`` | ``__builtins__["__import__"]`` | We already have this | ✅ same pattern |
| 2 | ``SandboxFinder`` | inserts/removes from ``sys.meta_path`` | No — list-mutation race | ✅ install once + TLS-gate ``find_spec`` |
| 3a | ``cloudpickle.subimport`` | replaces module function | No — save/restore stack race | ✅ install once + TLS-gate body |
| 3b | ``CustomCloudUnpickler.find_class`` | replaces class attribute | No — same | ✅ install once + TLS-gate body |
| 4 | builtins stripping | ``del builtins[k]`` then restore | No — see ``TODO_batched_attr_writes.md`` style race | ❌ structurally process-global |
| 5 | ``restricted_compile`` / ``restricted_exec`` | replaces ``__builtins__["compile"]`` / ``["exec"]`` | No — same as 3a | ✅ install once + TLS-gate body |
| 6 | ``sys.addaudithook`` | hook is permanent; ``sandbox_active.enabled`` TLS flag toggles its enforcement | Already TLS-designed upstream | ✅ already there, just plumb into our Protector |

Layer 4 is the only one that truly cannot be ported. The other four
(2, 3a, 3b, 5) all fit the same "install once + TLS-gate" template
that our existing import dispatcher uses.

## The race — concrete example

For any layer that does Patcher-based save/restore against shared
process state, two concurrent ``Protector`` scopes interleave like:

```
t=0.00s  Thread A  Patcher.__enter__
                   saved_A[key] = original_value
                   target[key] = replacement

t=0.10s  Thread B  Patcher.__enter__
                   saved_B[key] = replacement   # already replaced by A
                   target[key] = replacement    # no-op

t=0.50s  Thread A  Patcher.__exit__
                   target[key] = saved_A[key]   # restored to original
                   But Thread B is still sandboxed; its view assumes
                   the replacement is still installed.  ← LEAK

t=1.00s  Thread B  Patcher.__exit__
                   target[key] = saved_B[key]   # = replacement
                   Replacement now permanently installed even though
                   no Protector is active.  ← LEAK
```

For layer 4 (builtins stripping) the leaked state is "key missing
from builtins", which means subsequent unsandboxed code dies on
``NameError: name 'open' is not defined``. For layers 2/3a/3b/5 the
leaked state is "secure version still installed", which fails safe
(no security regression) but holds stale closure references and is
ugly.

## The TLS-safe pattern (model)

Our ``__import__`` dispatcher is the template:

```python
# Module load — install once, never re-patch.
_real_import = builtins.__import__

def _dispatching_import(name, *args, **kwargs):
    active = getattr(_tls, "active", None)
    if active is None:
        return _real_import(name, *args, **kwargs)   # not sandboxed → pass through
    return active._importer(name, *args, **kwargs)   # gated check

builtins.__import__ = _dispatching_import   # one-time, permanent

# Protector scope — just toggles TLS, no patching.
class Protector:
    def __enter__(self):
        self._prev = getattr(_tls, "active", None)
        _tls.active = self
    def __exit__(self, *exc):
        _tls.active = self._prev
```

Every concurrent thread gets its own ``_tls.active`` view. No shared
mutation. No save/restore dance. Multiple Protectors active
simultaneously is fine.

The same pattern adapts to layers 2, 3a, 3b, 5:

- Layer 2 (SandboxFinder): insert ONE finder at module load. Its
  ``find_spec`` checks ``_tls.active`` — return ``None`` (let normal
  finders run) when no sandbox active; return blocking spec when
  active and module not allowed.
- Layer 3a (cloudpickle.subimport): wrap once at module load. Body
  checks TLS; if no sandbox, call original; if sandboxed, check
  whitelist.
- Layer 3b (find_class on CustomCloudUnpickler): same.
- Layer 5 (restricted_compile/exec): replace ``__builtins__["compile"]``
  and ``["exec"]`` ONCE at module load with TLS-gated wrappers.
  Wrapper checks TLS; if no sandbox, call real compile/exec; if
  sandboxed, run the restricted version.
- Layer 6 (audit hook): already designed this way upstream; plumb
  ``sandbox_active.enabled = True`` in our Protector ``__enter__``
  and ``= False`` in ``__exit__``.

Total work: ~1 day of careful porting + tests.

## Why we punted to a follow-up PR

This branch's job is to ship cross-request batching plus the minimum
sandbox needed to not regress security obviously. Forcing the security
hardening through the same PR mixes two largely orthogonal changes
and triples the merge surface area.

When the follow-up PR lands, every layer above gets an equivalent
TLS-gated implementation; ``protector.py`` / ``importer.py`` /
``guards.py`` get rewritten on the install-once pattern; our
``protected_environment.py`` gets deleted; ``__init__.py`` re-exports
from the new modules; ``base.py`` calls remain unchanged.

## Layer 4 — the irreducible drop

Builtins stripping (``del builtins[k]``) cannot be made TLS-safe
because the ``builtins`` module dict is one shared object. There is
no mechanism in CPython for "this key is present on thread A but
absent on thread B" without a C extension.

The follow-up PR should drop layer 4. Coverage is partially preserved
by our per-name globals shadow (``eval``/``exec``/``open``/``compile``
intercepted in the user frame's ``__globals__`` before falling
through to builtins). The remaining gap is "whitelisted library code
internally calls ``open()`` as a free name that resolves through
process builtins" — a real but narrow attack surface that the
security team should triage on its own merits.

## Related code

- ``src/services/ray/src/ray/nn/security/protected_environment.py`` —
  this branch's TLS Protector (active).
- ``src/services/ray/src/ray/nn/security/protector.py`` —
  upstream's six-layer Protector (inert in this branch; rewritten by
  the follow-up PR).
- ``src/services/ray/src/ray/nn/security/importer.py`` —
  upstream's Importer with SandboxFinder (inert; rewritten).
- ``src/services/ray/src/ray/nn/security/guards.py`` — upstream's
  ``restricted_compile`` / ``restricted_exec`` / audit hook
  (inert; selectively used in the follow-up).
- ``src/services/ray/src/ray/nn/security/whitelist.py`` +
  ``whitelist.yaml`` — upstream's policy data; this branch already
  ships them as inert reference files.
