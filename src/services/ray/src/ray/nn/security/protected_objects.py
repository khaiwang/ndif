from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

import torch

PROTECTIONS = {}

_PROTECTED_CLASS_CACHE: dict[type, type] = {}

# A/B switch for measuring the pre-optimization protect() cost.
# When set, both optimizations are disabled:
#   1. _PROTECTED_CLASS_CACHE is bypassed — fresh ``_ProtectedObject``
#      class synthesized per call.
#   2. ``__getattribute__`` does not bypass dunders — every ``__dict__``
#      access on a wrapped module deepcopies the real module's state,
#      which pickle/copy machinery triggers during unpickle.
_SLOW_PROTECT = os.getenv("NDIF_SLOW_PROTECT") == "1"


def protected(obj: Any):
    return id(obj) in PROTECTIONS


class ProtectedObject:
    def __init__(self, obj: Any):
        PROTECTIONS[id(self)] = obj

    def __getattribute__(self, name: str):
        if name in ["to"]:
            raise ValueError(f"Attribute `{name}` cannot be accessed")

        obj = PROTECTIONS[id(self)]

        if not _SLOW_PROTECT and name.startswith("__") and name.endswith("__"):
            # Dunder access is Python-internal (pickle, copy, isinstance,
            # repr). User-level protection only needs the non-dunder surface.
            return getattr(obj, name)

        value = getattr(obj, name)

        if not isinstance(value, (torch.Tensor, list, dict)):
            return value

        value = deepcopy(value)

        print(
            f" WARNING: Accessing attribute `{name}` of protected object `{PROTECTIONS[id(self)]}` will return a deepcopy of the attribute."
        )

        return value

    def __setattr__(self, name: str, value: Any):
        if not protected(self):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                f"Attribute '{name}' cannot be set after initialization"
            )


def protect(obj: Any):
    base = type(obj)
    if _SLOW_PROTECT:
        return type("_ProtectedObject", (ProtectedObject, base), {})(obj)
    cls = _PROTECTED_CLASS_CACHE.get(base)
    if cls is None:
        cls = type("_ProtectedObject", (ProtectedObject, base), {})
        _PROTECTED_CLASS_CACHE[base] = cls
    return cls(obj)


def protect_persistent_objects(persistent_objects: dict) -> dict:
    """Return a new dict where every ``"Module:*"`` value is wrapped via
    :func:`protect`. ``"Interleaver"``, ``"Tokenizer"``, ``"Processor"``
    pass through unchanged. Called once at deployment init so per-request
    unpickling is a pure dict lookup.
    """
    return {
        key: protect(value) if isinstance(key, str) and key.startswith("Module:") else value
        for key, value in persistent_objects.items()
    }
