"""NDIF sandbox security.

Public API:
    Protector                        – context manager that activates the sandbox
    WHITELISTED_MODULES              – modules allowed during execution
    WHITELISTED_MODULES_DESERIALIZATION – modules allowed during deserialization

This branch ships the TLS-safe Protector implemented in
``protected_environment.py`` (compatible with concurrent mediator
workers under the batched execution path). Upstream dev's
``protector.py`` / ``importer.py`` / ``guards.py`` are present in this
directory but inert — a follow-up PR will rebuild them on the
install-once + TLS-gate pattern; until then they are not imported.
See ``docs/TODO_batched_security_layers.md``.
"""

from .protected_environment import (
    Protector,
    WHITELISTED_MODULES,
    WHITELISTED_MODULES_DESERIALIZATION,
)
