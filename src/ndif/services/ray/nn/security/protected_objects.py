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
The synthesized class is memoized per base type so warmup pays
O(unique_module_types) class creations rather than O(modules).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

# Maps ``id(wrapper)`` → original unwrapped object.
PROTECTIONS: dict[int, Any] = {}

# Per-type cache of dynamically synthesized ``_ProtectedObject`` subclasses.
# Without this, every call to ``protect()`` creates a fresh class, which on
# a transformer with ~10 unique module types and ~500 module instances
# costs O(instances) class syntheses at warmup. With it, O(types).
_PROTECTED_CLASS_CACHE: dict[type, type] = {}

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
            raise ValueError(
                f"Method `{name}` cannot be called on a protected object"
            )

        obj = PROTECTIONS[id(self)]

        # Dunder access is Python-internal (pickle, copy, isinstance, repr).
        # User-level protection only needs the non-dunder surface; routing
        # dunders through the deepcopy branch would deepcopy
        # ``module.__dict__`` (containing ``_parameters`` with weight tensors)
        # every time pickle introspects the object — catastrophic at 32B+
        # model sizes.
        if name.startswith("__") and name.endswith("__"):
            return getattr(obj, name)

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
        # Writes to a protected module are refused outright after __init__.
        #
        # Upstream dev uses a mutate-then-rollback pattern (SET_ATTRS +
        # clear_set_attrs()) that is safe only when requests are
        # serialized on the actor. Under our batched execution path
        # (VanillaBatchServer fusing N concurrent users into one forward)
        # mutating the real shared module would corrupt co-batched peers
        # before rollback can land.
        #
        # See docs/TODO_batched_attr_writes.md for the detailed analysis
        # and the design sketch for a future "quiesce then exclusive"
        # escape hatch that would allow writes without sacrificing
        # batching isolation.
        if not protected(self):
            # Still inside __init__ — allow normal attribute setting.
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                f"Attribute '{name}' cannot be set after initialization"
            )


def protect(obj: Any):
    """Wrap *obj* in a ProtectedObject that also inherits from obj's class.

    The synthesized subclass is memoized on ``type(obj)`` so a warmup
    sweep over every persistent module pays O(unique_types) class
    syntheses instead of O(instances).
    """
    base = type(obj)
    cls = _PROTECTED_CLASS_CACHE.get(base)
    if cls is None:
        cls = type("_ProtectedObject", (ProtectedObject, base), {})
        _PROTECTED_CLASS_CACHE[base] = cls
    return cls(obj)
