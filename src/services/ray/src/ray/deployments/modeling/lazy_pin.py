"""Pipelined lazy-pin transfers for GPU↔CPU model weight movement.

Pins destination memory *in-place* via ``cudaHostRegister`` at transfer time,
then DMA-copies directly between GPU VRAM and the final CPU tensor — no
intermediate relay buffer, no persistent pinned pool.

A 3-stage pipeline overlaps the work across chunks of the model::

    Chunk 0:  [touch pages]  [cudaHostRegister]  [DMA copy]
    Chunk 1:                 [touch pages]        [cudaHostRegister]  [DMA copy]
    Chunk 2:                                      [touch pages]       [register]  [DMA]

``torch.empty()`` only reserves virtual address space (microseconds even for
15 GB).  Physical pages are allocated by the OS on first write — that's what
the *touch* stage does.  Registration then page-locks those pages for DMA.
All three stages run on separate threads so their costs overlap.

The touch stage uses ``libc.memset`` (not PyTorch ops) to release the GIL,
enabling true concurrent execution across the pipeline stages.

Steady-state throughput ≈ ``max(touch, register, DMA)`` per chunk.
"""

import ctypes
import logging
import os
import queue
import threading
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger("ndif")

_DEFAULT_CHUNK_MB = int(os.environ.get("NDIF_LAZYPIN_CHUNK_MB", "256"))

# ---------------------------------------------------------------------------
# Native library bindings
# ---------------------------------------------------------------------------

_cudart: ctypes.CDLL | None = None
_native_init_done = False
_CUDA_HOST_REGISTER_DEFAULT = 0


def _init_native() -> None:
    """Load libcudart once, setting up argtypes/restypes."""
    global _cudart, _native_init_done
    if _native_init_done:
        return
    _native_init_done = True

    # libcudart — for cudaHostRegister / cudaHostUnregister
    search_paths: list[str] = []
    try:
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        for f in sorted(os.listdir(torch_lib)):
            if f.startswith("libcudart.so"):
                search_paths.append(os.path.join(torch_lib, f))
    except Exception:
        pass
    search_paths.extend(["libcudart.so", "libcudart.so.12", "libcudart.so.11.0"])

    for path in search_paths:
        try:
            lib = ctypes.CDLL(path)
            lib.cudaHostRegister.restype = ctypes.c_int
            lib.cudaHostRegister.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint,
            ]
            lib.cudaHostUnregister.restype = ctypes.c_int
            lib.cudaHostUnregister.argtypes = [ctypes.c_void_p]
            _cudart = lib
            return
        except (OSError, AttributeError):
            continue


def _write_touch_pages(tensor: torch.Tensor) -> None:
    """Write-touch one byte per page to force the OS to allocate physical pages.

    Uses stride-4096 assignment which partially releases the GIL, allowing
    other pipeline stages to make progress concurrently.
    """
    tensor[::4096] = 0


def _read_touch_pages(tensor: torch.Tensor) -> None:
    """Read-touch pages to ensure they're resident (e.g. not swapped out).

    For cpu_to_gpu, source pages already hold model weights — we must NOT
    overwrite them.
    """
    tensor[::4096].sum()


def _register(ptr: int, nbytes: int) -> bool:
    """Pin *nbytes* starting at *ptr* via cudaHostRegister (releases GIL)."""
    _init_native()
    if _cudart is None:
        return False
    return _cudart.cudaHostRegister(
        ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes),
        ctypes.c_uint(_CUDA_HOST_REGISTER_DEFAULT),
    ) == 0


def _unregister(ptr: int) -> None:
    """Unpin memory previously registered with cudaHostRegister."""
    _init_native()
    if _cudart is not None:
        _cudart.cudaHostUnregister(ctypes.c_void_p(ptr))


# ---------------------------------------------------------------------------
# Tensor iteration helper
# ---------------------------------------------------------------------------

def _unique_named_tensors(module: nn.Module):
    """Yield ``(name, tensor)`` for every unique param/buffer by data_ptr."""
    seen: set[int] = set()
    for name, param in module.named_parameters():
        ptr = param.data.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            yield name, param
    for name, buf in module.named_buffers():
        ptr = buf.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            yield name, buf


# ---------------------------------------------------------------------------
# Chunk descriptor for the pipeline
# ---------------------------------------------------------------------------

class _Chunk:
    __slots__ = ("cpu", "gpu", "nbytes")

    def __init__(self, cpu: torch.Tensor, gpu: torch.Tensor) -> None:
        self.cpu = cpu
        self.gpu = gpu
        self.nbytes = cpu.numel()


_SENTINEL = None


# ---------------------------------------------------------------------------
# 3-stage pipeline core
# ---------------------------------------------------------------------------

def _run_pipeline(
    chunks: list[_Chunk],
    *,
    direction: str,
    device: torch.device,
) -> None:
    """Execute the touch → register → DMA pipeline.

    All three stages release the GIL, enabling true concurrent execution:
    - Touch: ``libc.memset`` (ctypes CDLL call)
    - Register: ``cudaHostRegister`` (ctypes CDLL call)
    - DMA: ``torch.Tensor.copy_`` with ``non_blocking=True``

    Args:
        chunks: ordered list of chunks to process.
        direction: ``"g2c"`` (GPU→CPU) or ``"c2g"`` (CPU→GPU).
        device: CUDA device for the stream.
    """
    touched_q: queue.Queue[_Chunk | None] = queue.Queue()
    registered_q: queue.Queue[_Chunk | None] = queue.Queue()
    pinned_ptrs: list[int] = []
    ptrs_lock = threading.Lock()

    # Stage 1 — pre-touch / fault in pages
    # g2c: write-touch via memset (releases GIL, allows true overlap)
    # c2g: read-touch via strided access (pages already hold data)
    def toucher() -> None:
        for c in chunks:
            if direction == "g2c":
                _write_touch_pages(c.cpu)
            else:
                _read_touch_pages(c.cpu)
            touched_q.put(c)
        touched_q.put(_SENTINEL)

    # Stage 2 — cudaHostRegister (releases GIL via ctypes)
    def registerer() -> None:
        while True:
            c = touched_q.get()
            if c is _SENTINEL:
                registered_q.put(_SENTINEL)
                return
            ptr = c.cpu.data_ptr()
            if _register(ptr, c.nbytes):
                with ptrs_lock:
                    pinned_ptrs.append(ptr)
            else:
                logger.warning("cudaHostRegister failed at %#x (%d B)", ptr, c.nbytes)
            registered_q.put(c)

    t1 = threading.Thread(target=toucher, daemon=True)
    t2 = threading.Thread(target=registerer, daemon=True)
    t1.start()
    t2.start()

    # Stage 3 — DMA (main thread, uses CUDA stream)
    stream = torch.cuda.Stream(device=device)
    while True:
        c = registered_q.get()
        if c is _SENTINEL:
            break
        with torch.cuda.stream(stream):
            if direction == "g2c":
                c.cpu.copy_(c.gpu, non_blocking=True)
            else:
                c.gpu.copy_(c.cpu, non_blocking=True)

    stream.synchronize()
    t1.join()
    t2.join()

    # Unpin all registered pages
    for ptr in pinned_ptrs:
        _unregister(ptr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def gpu_to_cpu(
    module: nn.Module,
    chunk_size: int | None = None,
) -> Dict[str, torch.Tensor]:
    """Transfer all GPU tensors in *module* to CPU via pipelined lazy pinning.

    Allocates unpinned CPU destination tensors, then runs a 3-stage pipeline
    (touch → register → DMA) across chunks to overlap the costs.

    Returns ``{name: cpu_tensor}`` — caller is responsible for reassigning
    ``param.data`` on the module.
    """
    if chunk_size is None:
        chunk_size = _DEFAULT_CHUNK_MB * 1024 * 1024

    cpu_tensors: Dict[str, torch.Tensor] = {}
    chunks: list[_Chunk] = []
    device: torch.device | None = None

    for name, tensor in _unique_named_tensors(module):
        gpu_flat = tensor.data.contiguous().view(torch.uint8).reshape(-1)
        cpu_dest = torch.empty(tensor.data.shape, dtype=tensor.data.dtype, device="cpu")
        cpu_tensors[name] = cpu_dest
        cpu_flat = cpu_dest.view(torch.uint8).reshape(-1)

        if device is None:
            device = gpu_flat.device

        total = gpu_flat.numel()
        offset = 0
        while offset < total:
            length = min(chunk_size, total - offset)
            chunks.append(_Chunk(
                cpu_flat.narrow(0, offset, length),
                gpu_flat.narrow(0, offset, length),
            ))
            offset += length

    if chunks and device is not None:
        _run_pipeline(chunks, direction="g2c", device=device)

    return cpu_tensors


def cpu_to_gpu(
    module: nn.Module,
    device: torch.device,
    chunk_size: int | None = None,
) -> Dict[str, torch.Tensor]:
    """Transfer all CPU tensors in *module* to *device* via pipelined lazy pinning.

    Returns ``{name: gpu_tensor}`` — caller is responsible for reassigning
    ``param.data`` on the module.
    """
    if chunk_size is None:
        chunk_size = _DEFAULT_CHUNK_MB * 1024 * 1024

    gpu_tensors: Dict[str, torch.Tensor] = {}
    chunks: list[_Chunk] = []

    for name, tensor in _unique_named_tensors(module):
        cpu_flat = tensor.data.contiguous().view(torch.uint8).reshape(-1)
        gpu_dest = torch.empty(tensor.data.shape, dtype=tensor.data.dtype, device=device)
        gpu_tensors[name] = gpu_dest
        gpu_flat = gpu_dest.view(torch.uint8).reshape(-1)

        total = cpu_flat.numel()
        offset = 0
        while offset < total:
            length = min(chunk_size, total - offset)
            chunks.append(_Chunk(
                cpu_flat.narrow(0, offset, length),
                gpu_flat.narrow(0, offset, length),
            ))
            offset += length

    if chunks:
        _run_pipeline(chunks, direction="c2g", device=device)

    return gpu_tensors
