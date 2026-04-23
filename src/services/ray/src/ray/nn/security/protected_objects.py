"""Protected object wrappers for model and tokenizer instances.

When a model is loaded by the ModelActor, its torch.nn.Module is wrapped with
``protect(module)`` before being handed to user code.  This prevents users
from:

    - Moving the model between devices — ``.to()``, ``.cuda()``, ``.cpu()``,
      and dtype-changing methods are blocked because they would break the
      shared GPU memory layout.
    - Silently mutating tensor attributes in-place — reads of Tensor, list,
      and dict attributes return deep copies so the original stays intact.
    - Modifying module attributes — writes are refused outright on this
      branch (see __setattr__ for why; the upstream rollback design is
      incompatible with our batched execution path).

Implementation note: ``protect(obj)`` creates a dynamic subclass of both
ProtectedObject and ``obj.__class__``.  This way ``isinstance(wrapped, Module)``
still returns True, keeping downstream code (nnsight, accelerate, etc.) happy.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from copy import deepcopy
from typing import Any

import torch

# Maps ``id(wrapper)`` → original unwrapped object.
PROTECTIONS: dict[int, Any] = {}

# Tracks attribute writes so they can be rolled back when exclusive mode is
# active. Maps ``id(wrapper)`` → { attr_name: original_value }. Dormant by
# default (writes are refused unless ``_exclusive_mode`` is set). See
# ``docs/TODO_batched_attr_writes.md`` for the quiesce-then-exclusive design
# this supports.
SET_ATTRS: defaultdict[int, dict[str, Any]] = defaultdict(dict)

# When set, ``__setattr__`` allows writes and records them for rollback via
# ``clear_set_attrs()``. The actor-side coordinator in
# ``BaseModelDeployment.execute_batched`` is responsible for only entering
# exclusive mode after draining the in-flight batch — see the TODO doc.
_exclusive_mode = threading.Event()

# Methods that move the model between devices or change its dtype.
# Blocked because the model is shared across requests and pinned to
# specific GPUs by the cluster scheduler.
_BLOCKED_METHODS = frozenset(
    {
        # Device movement
        "to",
        "cuda",
        "cpu",
        "xpu",
        "ipu",
        "to_empty",
        # Dtype changes (these return a new module / modify in-place)
        "half",
        "float",
        "double",
        "bfloat16",
        # Gradient state mutation
        "requires_grad_",
    }
)

# Attributes that should pass through without the deepcopy treatment.
# These are forward-hook dicts that nnsight's intervention machinery
# legitimately reads; deepcopying them breaks intervention registration.
_ALLOWED_ATTRIBUTES = frozenset(
    {
        "_forward_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_pre_hooks",
        "_forward_post_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_forward_post_hooks_with_kwargs",
    }
)


def protected(obj: Any) -> bool:
    """True if *obj* is a ProtectedObject that has finished __init__."""
    return id(obj) in PROTECTIONS


class ProtectedObject:

    def __init__(self, obj: Any):
        PROTECTIONS[id(self)] = obj

    def __getattribute__(self, name: str):
        if name in _BLOCKED_METHODS:
            raise ValueError(f"Method `{name}` cannot be called on a protected object")

        obj = PROTECTIONS[id(self)]
        value = getattr(obj, name)

        # Return deep copies of mutable types so users can't silently mutate
        # the model's internal state (e.g. bias vectors, config dicts).
        if name not in _ALLOWED_ATTRIBUTES and isinstance(
            value, (torch.Tensor, list, dict)
        ):
            value = deepcopy(value)
            print(
                f" WARNING: Accessing attribute `{name}` of protected object"
                f" `{PROTECTIONS[id(self)]}` will return a deepcopy of the attribute."
            )

        return value

    def __getattr__(self, name: str):
        raise AttributeError(f"Attribute `{name}` cannot be accessed")

    def __setattr__(self, name: str, value: Any):
        # Writes to a protected module are refused by default, and tracked
        # for rollback only when the actor has entered exclusive mode
        # (single in-flight request, batch drained).
        #
        # Upstream dev unconditionally uses the mutate-then-rollback path
        # (SET_ATTRS + clear_set_attrs()). That is safe when requests are
        # serialized on the actor but corrupts co-batched peers under our
        # VanillaBatchServer path: the real shared module is mutated
        # synchronously, the fused forward already reads the bad state,
        # and rollback only lands at end-of-request — too late.
        #
        # The gate here keeps the machinery wired up so that when the
        # quiesce-then-exclusive coordinator ships (see
        # docs/TODO_batched_attr_writes.md), enabling writes is just a
        # matter of setting ``_exclusive_mode`` after the drain completes.
        if not protected(self):
            # Still inside __init__ — allow normal attribute setting.
            object.__setattr__(self, name, value)
        elif _exclusive_mode.is_set():
            # Exclusive mode: actor has drained other in-flight requests
            # and is running this one alone. Safe to mutate the real
            # module; record for rollback.
            SET_ATTRS[id(self)][name] = getattr(PROTECTIONS[id(self)], name)
            PROTECTIONS[id(self)].__dict__[name] = value
        else:
            raise AttributeError(
                f"Attribute '{name}' cannot be set after initialization"
            )


def protect(obj: Any):
    """Wrap *obj* in a ProtectedObject that also inherits from obj's class."""

    class _ProtectedObject(ProtectedObject, obj.__class__):
        pass

    return _ProtectedObject(obj)


def clear_set_attrs():
    """Revert attribute writes recorded during exclusive mode.

    Only meaningful after writes were permitted via ``_exclusive_mode``;
    in the default (refuse-write) posture this is a no-op because
    ``SET_ATTRS`` stays empty. The actor-side coordinator is responsible
    for calling this in a ``finally`` block around any exclusive-mode
    run so a dying request can't leak mutations into the next. See
    ``docs/TODO_batched_attr_writes.md``.
    """
    for wrapper_id, writes in list(SET_ATTRS.items()):
        if not writes:
            continue
        real = PROTECTIONS.get(wrapper_id)
        if real is None:
            writes.clear()
            continue
        for name, original in writes.items():
            real.__dict__[name] = original
        writes.clear()
