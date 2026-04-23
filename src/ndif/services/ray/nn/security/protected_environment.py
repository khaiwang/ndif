from __future__ import annotations

import inspect
import threading
from functools import wraps
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from pydantic import BaseModel

from nnsight.modeling.mixins.remoteable import StreamTracer

# Built-in functions and types that are allowed to be used
WHITELISTED_BUILTINS = {
    # Built-in exceptions
    "BaseExceptionGroup",
    "ArithmeticError",
    "AssertionError",
    "AttributeError",
    "BaseException",
    "BlockingIOError",
    "BrokenPipeError",
    "BufferError",
    "BytesWarning",
    "ChildProcessError",
    "ConnectionAbortedError",
    "ConnectionError",
    "ConnectionRefusedError",
    "ConnectionResetError",
    "DeprecationWarning",
    "EOFError",
    "Ellipsis",
    "EncodingWarning",
    "EnvironmentError",
    "Exception",
    "False",
    "FileExistsError",
    "FileNotFoundError",
    "FloatingPointError",
    "FutureWarning",
    "GeneratorExit",
    "IOError",
    "ImportError",
    "ImportWarning",
    "IndentationError",
    "IndexError",
    "InterruptedError",
    "IsADirectoryError",
    "KeyError",
    "KeyboardInterrupt",
    "LookupError",
    "MemoryError",
    "ModuleNotFoundError",
    "NameError",
    "None",
    "NotADirectoryError",
    "NotImplemented",
    "NotImplementedError",
    "OSError",
    "OverflowError",
    "PendingDeprecationWarning",
    "PermissionError",
    "ProcessLookupError",
    "RecursionError",
    "ReferenceError",
    "ResourceWarning",
    "RuntimeError",
    "RuntimeWarning",
    "StopAsyncIteration",
    "StopIteration",
    "SyntaxError",
    "SyntaxWarning",
    "SystemError",
    "SystemExit",
    "TabError",
    "TimeoutError",
    "True",
    "TypeError",
    "UnboundLocalError",
    "UnicodeDecodeError",
    "UnicodeEncodeError",
    "UnicodeError",
    "UnicodeTranslateError",
    "UnicodeWarning",
    "UserWarning",
    "ValueError",
    "Warning",
    "ZeroDivisionError",
    # Built-in special attributes
    "__doc__",
    "__import__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
    "__build_class__",
    # Built-in functions
    "abs",
    "aiter",
    "all",
    "anext",
    "any",
    "ascii",
    "bool",
    "bytearray",
    "bytes",
    "callable",
    "chr",
    "classmethod",
    "complex",
    "copyright",
    "credits",
    "delattr",
    "dict",
    "dir",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "getattr",
    "hasattr",
    "hash",
    "hex",
    "id",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "oct",
    "ord",
    "pow",
    "print",
    "property",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "setattr",
    "slice",
    "sorted",
    "staticmethod",
    "str",
    "sum",
    "super",
    "tuple",
    "type",
    "vars",
    "zip",
    "memoryview",
}

SAFE_BUILTINS = {
    key: value for key, value in __builtins__.items() if key in WHITELISTED_BUILTINS
}


class SafeBuiltins(ModuleType):
    """A wrapper around the built-in module that enforces whitelist rules."""

    def __init__(self):
        super().__init__("safe_builtins")

    def __getattribute__(self, name: str):
        return SAFE_BUILTINS[name]

    def __getitem__(self, name: str):
        return SAFE_BUILTINS[name]


PROTECTED_BUILTINS = SafeBuiltins()


class WhitelistedModule(BaseModel):
    """Configuration for a module that is allowed to be imported."""

    name: str
    strict: bool = True

    def check(self, name: str) -> bool:
        return (
            self.strict
            and self.name == name
            or not self.strict
            and name.startswith(self.name)
        )


# Modules that are allowed to be imported. ``import builtins`` is handled
# separately by ``Importer.__call__`` returning ``PROTECTED_BUILTINS``.
WHITELISTED_MODULES = [
    WhitelistedModule(name="torch", strict=False),
    WhitelistedModule(name="collections", strict=False),
    WhitelistedModule(name="nnsight.intervention.envoy", strict=False),
    WhitelistedModule(name="time", strict=False),
    WhitelistedModule(name="numpy", strict=False),
    WhitelistedModule(name="sympy", strict=False),
    WhitelistedModule(name="nnterp", strict=False),
    WhitelistedModule(name="math", strict=False),
    WhitelistedModule(name="einops", strict=False),
    WhitelistedModule(name="urllib3", strict=False),
    WhitelistedModule(name="typing", strict=False),
    WhitelistedModule(name="_operator", strict=True),
    WhitelistedModule(name="operator", strict=True),
    WhitelistedModule(name="pandas", strict = False),
    WhitelistedModule(name="enum", strict = False),
]

# Modules allowed during deserialization
WHITELISTED_MODULES_DESERIALIZATION = [
    WhitelistedModule(name="pickle", strict=False),
    WhitelistedModule(name="cloudpickle", strict=False),
    WhitelistedModule(name="copyreg", strict=False),
    WhitelistedModule(name="nnsight.schema.request", strict=True),
    WhitelistedModule(name="nnsight.modeling.mixins.remoteable", strict=True),
    WhitelistedModule(name="nnsight.intervention.tracing.base", strict=True),
    WhitelistedModule(name="nnsight.intervention.interleaver", strict=True),
    WhitelistedModule(name="nnsight.intervention.batching", strict=True),
    WhitelistedModule(name="nnsight.intervention.serialization", strict=True),
    WhitelistedModule(name="nnsight.modeling", strict=False),
    WhitelistedModule(name="transformers", strict=False),
    *WHITELISTED_MODULES,
]


class ProtectedModule(ModuleType):
    """A wrapper around a module that enforces whitelist rules."""

    def __init__(self, whitelist_entry: WhitelistedModule):
        super().__init__(whitelist_entry.name)
        self.whitelist_entry = whitelist_entry

    def __getattribute__(self, name: str):
        attr = super().__getattribute__(name)

        if not isinstance(attr, ModuleType):
            return attr

        if self.whitelist_entry.strict:
            if self.__name__ != attr.__name__:
                raise AttributeError(
                    f"Module attribute {attr.__name__} is not whitelisted"
                )
        elif not attr.__name__.startswith(self.__name__ + "."):
            raise AttributeError(f"Module attribute {attr.__name__} is not whitelisted")

        protected = ProtectedModule(self.whitelist_entry)
        protected.__dict__.update(attr.__dict__)
        return protected


# ======================================================================
# Thread-local sandbox dispatch
# ======================================================================
# A process-global ``__import__`` dispatcher is installed at module import.
# Threads with no active Protector pass through to the original ``__import__``.
# Threads inside a Protector scope set ``_tls.active`` and route imports
# through the whitelist.

_tls = threading.local()


def _builtins_dict() -> dict:
    """Return the underlying builtins dict in a way that works whether
    ``__builtins__`` is bound to the module (modules) or the dict
    (``__main__`` / many internal frames)."""
    b = __builtins__
    return b if isinstance(b, dict) else b.__dict__


_BUILTINS = _builtins_dict()
_ORIGINAL_IMPORT = _BUILTINS["__import__"]


def _dispatching_import(
    name: str,
    globals: Optional[Dict[str, Any]] = None,
    locals: Optional[Dict[str, Any]] = None,
    fromlist: Optional[List[str]] = None,
    level: int = 0,
):
    """Process-global ``__import__`` replacement. Thread-local fast path:
    unsandboxed threads incur one attribute lookup + one function call
    before reaching the original import."""
    active = getattr(_tls, "active", None)
    if active is None:
        return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
    return active._importer(name, globals, locals, fromlist, level)


_DISPATCH_INSTALLED = False


def _install_dispatch_once():
    """Idempotent install. Safe to call multiple times — the dispatcher is
    re-assigned but the original is captured only once.

    Deferred to first ``Protector(...)`` construction (rather than running
    at module import) because merely importing this module elsewhere in
    the process tree (e.g. into the Ray client-server worker that
    unpickles incoming RPC args) would otherwise hijack ``__import__``
    process-wide. The worker only needs to deserialize plain Python
    objects; routing its imports through our dispatcher serves no
    purpose and leaks our hook to call sites that aren't actor code.
    """
    global _DISPATCH_INSTALLED
    if _DISPATCH_INSTALLED:
        return
    _BUILTINS["__import__"] = _dispatching_import
    # PROTECTED_BUILTINS exposes only whitelisted names; keep its
    # ``__import__`` consistent so user code that looks up the builtin
    # explicitly still lands in the dispatcher.
    SAFE_BUILTINS["__import__"] = _dispatching_import
    _DISPATCH_INSTALLED = True


# ``StreamTracer.execute`` runs pickle/cloudpickle internals that do their
# own imports. Bypass the sandbox for the duration of that call so the
# stream path isn't whitelist-bound. Installed once; the wrapper is cheap
# when no Protector is active.
_orig_stream_execute = StreamTracer.execute


def _unsandboxed_stream_execute(self, *args, **kwargs):
    saved = getattr(_tls, "active", None)
    _tls.active = None
    try:
        return _orig_stream_execute(self, *args, **kwargs)
    finally:
        _tls.active = saved


StreamTracer.execute = _unsandboxed_stream_execute


# Modules that no allowlist may grant access to, even transitively. Each
# exposes a direct escape that ``ProtectedModule`` cannot contain:
#   * sys, importlib  — module-table introspection
#   * threading       — child threads start with no TLS
#   * subprocess, os  — FS / process control
#   * ctypes          — FFI / arbitrary memory
NEVER_ALLOWED = frozenset(
    {"sys", "importlib", "threading", "subprocess", "os", "ctypes"}
)


class Importer:
    """Whitelist-gated module resolver. Invoked only for threads with an
    active Protector (via the ``_dispatching_import`` fast path)."""

    def __init__(self, whitelisted_modules: List[WhitelistedModule]):
        self.whitelisted_modules = whitelisted_modules

    def _real_import(self, *args, **kwargs):
        """Call the real ``__import__`` with the sandbox temporarily
        disabled on this thread. The real importer itself does nested
        imports (submodules, from-imports) and must not loop back through
        the whitelist."""
        saved = getattr(_tls, "active", None)
        _tls.active = None
        try:
            return _ORIGINAL_IMPORT(*args, **kwargs)
        finally:
            _tls.active = saved

    def __call__(
        self,
        name: str,
        globals: Dict[str, Any] = None,
        locals: Dict[str, Any] = None,
        fromlist: List[str] = None,
        level: int = 0,
    ):
        if name in ("builtins", "__builtins__"):
            return PROTECTED_BUILTINS

        # Hard block. Evaluated before the allowlist so it wins
        # unconditionally.
        if level == 0 and name.split(".", 1)[0] in NEVER_ALLOWED:
            raise ImportError(f"Module {name} is not permitted")

        if level > 0:
            # Relative import — resolve first, then whitelist the result.
            result = self._real_import(name, globals, locals, fromlist, level)
            if result.__name__.split(".", 1)[0] in NEVER_ALLOWED:
                raise ImportError(f"Module {result.__name__} is not permitted")
            for module in self.whitelisted_modules:
                if module.check(result.__name__):
                    protected = ProtectedModule(module)
                    protected.__dict__.update(result.__dict__)
                    return protected
            raise ImportError(f"Module {result.__name__} is not whitelisted")

        for module in self.whitelisted_modules:
            if module.check(name):
                result = self._real_import(name, globals, locals, fromlist, level)
                protected = ProtectedModule(module)
                protected.__dict__.update(result.__dict__)
                return protected

        raise ImportError(f"Module {name} is not whitelisted")


# Builtins that user intervention code must never call. Shadowed in the
# user frame's globals: Python name resolution is locals → globals →
# builtins, so a global-level binding intercepts before the frame's
# bound builtins fire.
_SHADOWED_BUILTINS = ("eval", "exec", "open", "compile")
_NO_PRIOR = object()


def _make_blocker(name: str):
    def _blocked(*args, **kwargs):
        raise PermissionError(
            f"Builtin `{name}` is not permitted in the sandboxed scope"
        )

    _blocked.__name__ = f"_blocked_{name}"
    return _blocked


class _ProtectorScope:
    """Per-invocation scope for the factory form ``protector(target_globals)``.

    On enter:
      * ``_tls.active = protector`` — the global ``__import__`` dispatcher
        routes this thread's imports through the allowlist.
      * Each name in ``_SHADOWED_BUILTINS`` is bound to a blocker in
        ``target_globals``.

    On exit both are undone in LIFO. The same ``Protector`` instance can
    be entered concurrently on multiple mediator threads: all per-call
    state lives on the scope object.
    """

    __slots__ = ("_protector", "_target_globals", "_prev_active", "_prev_shadows")

    def __init__(
        self, protector: "Protector", target_globals: Optional[Dict[str, Any]]
    ):
        self._protector = protector
        self._target_globals = target_globals
        self._prev_active = None
        self._prev_shadows: Optional[Dict[str, Any]] = None

    def __enter__(self):
        self._prev_active = getattr(_tls, "active", None)
        _tls.active = self._protector

        g = self._target_globals
        if g is not None:
            prev: Dict[str, Any] = {}
            for name in _SHADOWED_BUILTINS:
                prev[name] = g.get(name, _NO_PRIOR)
                g[name] = _make_blocker(name)
            self._prev_shadows = prev
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        g = self._target_globals
        if g is not None and self._prev_shadows is not None:
            for name, prev in self._prev_shadows.items():
                if prev is _NO_PRIOR:
                    g.pop(name, None)
                else:
                    g[name] = prev
        _tls.active = self._prev_active
        return False


class Protector:
    """Thread-local sandbox policy.

    Two usage patterns:

    1. Context manager — ``with Protector(modules):`` scopes TLS to the
       calling thread (so this thread's imports route through the
       allowlist via the global ``__import__`` dispatcher). Used on code
       paths where *we own* the code (deserialization, orchestration)
       and only need import restriction.

    2. Factory — ``scope = protector(target_globals); with scope: ...``
       activates TLS *and* shadows risky builtins
       (``open``/``eval``/``exec``/``compile``) in ``target_globals``
       via per-name dict writes. Intended for wrapping *user* code
       frames (nnsight's ``worker_context`` hook).

    Deliberately omitted features
    -----------------------------
    Upstream dev's ``Protector`` (in ``protector.py`` / ``importer.py``)
    additionally:
      * strips non-whitelisted names from the real ``builtins`` dict
        for the duration of the scope (its ``builtins=True`` mode);
      * patches ``cloudpickle.subimport`` / ``CustomCloudUnpickler.find_class``
        to close pickle's import-bypass holes;
      * inserts a ``SandboxFinder`` into ``sys.meta_path`` to block
        C-level imports;
      * replaces real ``compile`` / ``exec`` with restricted versions.

    Each of those is implemented via ``nnsight.util.Patcher``, which
    saves the previous value on ``__enter__`` and restores on
    ``__exit__``. Under our batched execution path multiple
    ``Protector`` scopes can be active concurrently on different
    mediator threads; the save/restore dance against shared process
    state then races (see ``docs/TODO_batched_security_layers.md``
    for a worked example).

    The builtins-stripping case is structurally process-global and
    cannot be made TLS-safe. The other layers can be ported to an
    install-once + per-call TLS-gate pattern but that work is
    intentionally deferred to a follow-up PR so this branch can ship
    batching without security regressions or hidden races.
    """

    def __init__(self, whitelisted_modules: List[WhitelistedModule]):
        # Install the global ``__import__`` dispatcher on first Protector
        # construction. Module import alone does not install — see
        # ``_install_dispatch_once`` for why.
        _install_dispatch_once()
        self.whitelisted_modules = whitelisted_modules
        self._importer = Importer(whitelisted_modules)

    # -- Context-manager form: TLS only --------------------------------
    # Previous-active is tracked on a per-thread stack so the same
    # Protector instance can be entered concurrently on multiple threads.
    def __enter__(self):
        stack = getattr(_tls, "_stack", None)
        if stack is None:
            stack = []
            _tls._stack = stack
        stack.append(getattr(_tls, "active", None))
        _tls.active = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _tls.active = _tls._stack.pop()
        return False

    # -- Factory form: worker_context(target_globals) → scope ----------
    def __call__(self, target_globals: Optional[Dict[str, Any]]) -> _ProtectorScope:
        return _ProtectorScope(self, target_globals)


# Bind the real ``compile`` onto ``ast`` so source-based deserialization
# can reach it as ``ast.compile``.
import ast

setattr(ast, "compile", compile)
