"""Pre-allocated pinned (page-locked) memory pool for fast GPU↔CPU transfers.

The pool collects all parameters and buffers from a model, deduplicates by
``data_ptr()`` (handling tied parameters), packs them sequentially into ≤2 GiB
``torch.uint8`` chunks with 4 KiB alignment, and exposes typed views that can
be used as drop-in replacements for the original tensors.

When available, chunks are backed by 2 MiB huge pages to reduce TLB pressure
during DMA registration and transfers — especially beneficial on machines with
TBs of memory.  The allocation strategy is tiered:

1. **Explicit huge pages** (``mmap`` + ``MAP_HUGETLB``) — guaranteed 2 MiB
   pages; requires pre-reserved huge pages via ``vm.nr_hugepages``.
2. **Transparent huge pages (THP)** — ``mmap`` + ``madvise(MADV_HUGEPAGE)``;
   the kernel *may* promote to 2 MiB pages opportunistically.
3. **Fallback** — ``torch.empty(pin_memory=True)`` with standard 4 KiB pages.

Allocation runs in a background thread so the model can keep serving while
pinned memory is being registered.
"""

import ctypes
import ctypes.util
import gc
import logging
import os
import threading
import time
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("ndif")

# ≤2 GiB per chunk to avoid cudaHostAlloc limitations with large single allocs.
_MAX_CHUNK_BYTES = 2 * (1024 ** 3)
# 4 KiB alignment for optimal DMA page alignment within chunks.
_ALIGNMENT = 4096
# 2 MiB huge page size — used for mmap size rounding.
_HUGEPAGE_SIZE = 2 * 1024 * 1024

# ---------------------------------------------------------------------------
# Linux mmap / madvise constants
# ---------------------------------------------------------------------------
_PROT_RW = 0x1 | 0x2           # PROT_READ | PROT_WRITE
_MAP_PRIVATE_ANON = 0x02 | 0x20  # MAP_PRIVATE | MAP_ANONYMOUS
_MAP_HUGETLB = 0x40000
_MAP_HUGE_2MB = 21 << 26         # MAP_HUGE_SHIFT = 26
_MADV_HUGEPAGE = 14
_CUDA_HOST_REGISTER_PORTABLE = 1

# ---------------------------------------------------------------------------
# Lazy-loaded native libraries
# ---------------------------------------------------------------------------
_libc_lib: Optional[ctypes.CDLL] = None
_cudart_lib: Optional[ctypes.CDLL] = None
_native_init_done = False


def _init_native_libs() -> None:
    """Load libc and libcudart once, setting up argtypes/restypes."""
    global _libc_lib, _cudart_lib, _native_init_done
    if _native_init_done:
        return
    _native_init_done = True

    # --- libc ---
    try:
        lib_name = ctypes.util.find_library("c")
        if lib_name:
            lib = ctypes.CDLL(lib_name, use_errno=True)
            lib.mmap.restype = ctypes.c_void_p
            lib.mmap.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_long,
            ]
            lib.munmap.restype = ctypes.c_int
            lib.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            lib.madvise.restype = ctypes.c_int
            lib.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            _libc_lib = lib
    except Exception:
        pass

    # --- libcudart ---
    # Try torch's bundled cudart first, then system paths.
    search_paths: list[str] = []
    try:
        torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
        for f in sorted(os.listdir(torch_lib_dir)):
            if f.startswith("libcudart.so"):
                search_paths.append(os.path.join(torch_lib_dir, f))
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
            _cudart_lib = lib
            break
        except (OSError, AttributeError):
            continue


# ---------------------------------------------------------------------------
# Chunk allocation helpers
# ---------------------------------------------------------------------------

class _ChunkAlloc:
    """Tracks how a single chunk was allocated so we can clean up correctly."""
    __slots__ = ("ptr", "mmap_size", "kind")

    def __init__(self, ptr: int, mmap_size: int, kind: str) -> None:
        self.ptr = ptr             # mmap base address (0 for torch-managed)
        self.mmap_size = mmap_size  # mmap region size (may be rounded up for hugepages)
        self.kind = kind           # "hugepage" | "thp" | "torch"


def _align_hugepage(size: int) -> int:
    """Round *size* up to the next 2 MiB boundary."""
    return (size + _HUGEPAGE_SIZE - 1) & ~(_HUGEPAGE_SIZE - 1)


def _mmap_failed(ptr) -> bool:
    """Check whether an mmap return value indicates failure (MAP_FAILED = -1)."""
    if ptr is None:
        return True
    # On 64-bit Linux with c_void_p restype, MAP_FAILED is 2**64 - 1.
    return ptr == (1 << 64) - 1 or ptr == (1 << 32) - 1


def _alloc_pinned_chunk(size: int) -> tuple[torch.Tensor, _ChunkAlloc]:
    """Allocate a pinned chunk, trying huge pages before falling back to torch.

    Returns ``(tensor, alloc_metadata)``.
    """
    _init_native_libs()
    libc = _libc_lib
    cudart = _cudart_lib

    if libc is not None and cudart is not None:
        # ---- Tier 1: explicit 2 MiB huge pages ----
        hp_size = _align_hugepage(size)
        ptr = libc.mmap(
            None, hp_size, _PROT_RW,
            _MAP_PRIVATE_ANON | _MAP_HUGETLB | _MAP_HUGE_2MB,
            -1, 0,
        )
        if not _mmap_failed(ptr):
            ret = cudart.cudaHostRegister(ptr, hp_size, _CUDA_HOST_REGISTER_PORTABLE)
            if ret == 0:
                # Expose exactly 'size' bytes as a tensor (hp_size may be larger).
                arr = (ctypes.c_uint8 * size).from_address(ptr)
                tensor = torch.frombuffer(arr, dtype=torch.uint8)
                return tensor, _ChunkAlloc(ptr, hp_size, "hugepage")
            # cudaHostRegister failed — release the mmap and try next tier.
            libc.munmap(ctypes.c_void_p(ptr), hp_size)

        # ---- Tier 2: THP (transparent huge pages) via madvise hint ----
        mmap_size = _align_hugepage(size)  # align so kernel can promote whole range
        ptr = libc.mmap(
            None, mmap_size, _PROT_RW,
            _MAP_PRIVATE_ANON,
            -1, 0,
        )
        if not _mmap_failed(ptr):
            libc.madvise(ptr, mmap_size, _MADV_HUGEPAGE)
            ret = cudart.cudaHostRegister(ptr, mmap_size, _CUDA_HOST_REGISTER_PORTABLE)
            if ret == 0:
                arr = (ctypes.c_uint8 * size).from_address(ptr)
                tensor = torch.frombuffer(arr, dtype=torch.uint8)
                return tensor, _ChunkAlloc(ptr, mmap_size, "thp")
            libc.munmap(ctypes.c_void_p(ptr), mmap_size)

    # ---- Tier 3: standard torch pinned allocation (4 KiB pages) ----
    tensor = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    return tensor, _ChunkAlloc(0, 0, "torch")


def _release_chunk_alloc(alloc: _ChunkAlloc) -> None:
    """Release native resources for a single chunk allocation."""
    if alloc.kind == "torch":
        return  # torch owns the memory
    _init_native_libs()
    cudart = _cudart_lib
    libc = _libc_lib
    if cudart is not None and alloc.ptr:
        cudart.cudaHostUnregister(ctypes.c_void_p(alloc.ptr))
    if libc is not None and alloc.ptr:
        libc.munmap(ctypes.c_void_p(alloc.ptr), alloc.mmap_size)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _align(offset: int) -> int:
    """Round *offset* up to the next ``_ALIGNMENT`` boundary."""
    return (offset + _ALIGNMENT - 1) & ~(_ALIGNMENT - 1)


def unique_named_tensors(module: nn.Module):
    """Yield (name, tensor) for all unique params and buffers, deduplicated by data_ptr."""
    seen_ptrs: set[int] = set()
    for name, param in module.named_parameters():
        ptr = param.data.data_ptr()
        if ptr not in seen_ptrs:
            seen_ptrs.add(ptr)
            yield name, param
    for name, buf in module.named_buffers():
        ptr = buf.data_ptr()
        if ptr not in seen_ptrs:
            seen_ptrs.add(ptr)
            yield name, buf


class PinnedBufferPool:
    """A contiguous pinned-memory pool sized to exactly one model's state."""

    def __init__(self) -> None:
        self._chunks: list[torch.Tensor] = []
        self._chunk_allocs: list[_ChunkAlloc] = []
        self._views: Dict[str, torch.Tensor] = {}
        self._plan: list[tuple[int, int, int, torch.dtype, torch.Size, str]] = []
        self._ready = threading.Event()
        self._failed = False
        self._alloc_time: float = 0.0
        self._total_bytes: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True once pre-allocation finished successfully."""
        return self._ready.is_set() and not self._failed

    def wait_ready(self, timeout: float = 120.0) -> bool:
        """Block until the pool is ready or *timeout* seconds elapse.

        Returns ``True`` if the pool is usable, ``False`` otherwise.
        """
        self._ready.wait(timeout=timeout)
        return self.is_ready

    def get(self, name: str) -> Optional[torch.Tensor]:
        """Return the pinned view for *name*, or ``None``."""
        return self._views.get(name)

    def matches(self, module: nn.Module) -> bool:
        """Return True if this pool's layout matches *module*'s parameters/buffers.

        Checks that every unique tensor in the module has a corresponding view
        in the pool with the same shape and dtype.
        """
        expected = [(name, t.data.shape, t.data.dtype) for name, t in unique_named_tensors(module)]

        if len(expected) != len(self._views):
            return False

        for name, shape, dtype in expected:
            view = self._views.get(name)
            if view is None or view.shape != shape or view.dtype != dtype:
                return False

        return True

    def transfer_to_device(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Transfer all chunks to *device* and return GPU-side typed views.

        Instead of copying 200+ individual pinned views one-by-one, this copies
        only ~8 contiguous uint8 chunks and then creates typed views on the GPU
        side, drastically reducing per-transfer Python/CUDA API overhead.
        """
        gpu_chunks = []
        for chunk in self._chunks:
            gpu_chunks.append(chunk.to(device, non_blocking=True))

        gpu_views: Dict[str, torch.Tensor] = {}
        for chunk_idx, offset, nbytes, dtype, shape, name in self._plan:
            raw = gpu_chunks[chunk_idx][offset : offset + nbytes]
            gpu_views[name] = raw.view(dtype).reshape(shape)

        return gpu_views

    def release(self) -> None:
        """Explicitly free all pinned chunks and views."""
        self._views.clear()
        self._plan.clear()
        # Release torch tensor references before unmapping the backing memory.
        self._chunks.clear()
        for alloc in self._chunk_allocs:
            _release_chunk_alloc(alloc)
        self._chunk_allocs.clear()
        gc.collect()

    # ------------------------------------------------------------------
    # Allocation (synchronous + async wrapper)
    # ------------------------------------------------------------------

    def preallocate(self, module: nn.Module) -> None:
        """Allocate pinned buffers matching every unique tensor in *module*."""
        start = time.time()

        # 1. Collect unique tensors, deduplicate by data_ptr (tied params).
        entries = [(name, t.data if isinstance(t, nn.Parameter) else t)
                   for name, t in unique_named_tensors(module)]

        if not entries:
            self._ready.set()
            return

        # 2. Plan packing into ≤2 GiB chunks.
        #    Each entry: (chunk_idx, offset_in_chunk, nbytes, dtype, shape, name)
        plan: list[tuple[int, int, int, torch.dtype, torch.Size, str]] = []
        chunk_sizes: list[int] = []
        cur_chunk = 0
        cur_offset = 0

        for name, tensor in entries:
            nbytes = tensor.nelement() * tensor.element_size()
            aligned_offset = _align(cur_offset)

            if aligned_offset + nbytes > _MAX_CHUNK_BYTES and cur_offset > 0:
                # Finish current chunk at its used size.
                chunk_sizes.append(cur_offset)
                cur_chunk += 1
                cur_offset = 0
                aligned_offset = 0

            if nbytes > _MAX_CHUNK_BYTES:
                # Tensor larger than max chunk — give it its own chunk.
                if cur_offset > 0:
                    chunk_sizes.append(cur_offset)
                    cur_chunk += 1
                plan.append((cur_chunk, 0, nbytes, tensor.dtype, tensor.shape, name))
                chunk_sizes.append(nbytes)
                cur_chunk += 1
                cur_offset = 0
                continue

            plan.append(
                (cur_chunk, aligned_offset, nbytes, tensor.dtype, tensor.shape, name)
            )
            cur_offset = aligned_offset + nbytes

        if cur_offset > 0:
            chunk_sizes.append(cur_offset)

        self._total_bytes = sum(chunk_sizes)

        # 3. Allocate pinned chunks (hugepage → THP → torch fallback).
        chunks: list[torch.Tensor] = []
        allocs: list[_ChunkAlloc] = []
        tier_counts: Dict[str, int] = {"hugepage": 0, "thp": 0, "torch": 0}
        try:
            for size in chunk_sizes:
                tensor, alloc = _alloc_pinned_chunk(size)
                chunks.append(tensor)
                allocs.append(alloc)
                tier_counts[alloc.kind] += 1
        except Exception as e:
            logger.warning(
                f"PinnedBufferPool: chunk allocation failed after "
                f"{len(chunks)}/{len(chunk_sizes)} chunks: {e}"
            )
            # Clean up any chunks we already allocated.
            chunks.clear()
            for a in allocs:
                _release_chunk_alloc(a)
            allocs.clear()
            gc.collect()
            self._failed = True
            self._ready.set()
            return

        self._chunks = chunks
        self._chunk_allocs = allocs
        self._plan = plan

        # 4. Create typed views into the chunks.
        for chunk_idx, offset, nbytes, dtype, shape, name in plan:
            raw = self._chunks[chunk_idx][offset : offset + nbytes]
            self._views[name] = raw.view(dtype).reshape(shape)

        self._alloc_time = time.time() - start
        self._ready.set()

        tier_summary = ", ".join(f"{k}={v}" for k, v in tier_counts.items() if v)
        logger.info(
            f"PinnedBufferPool allocated {self._total_bytes / (1024**3):.2f} GiB "
            f"in {len(self._chunks)} chunk(s) for {len(self._views)} tensors "
            f"({self._alloc_time:.2f}s) [{tier_summary}]"
        )

    def preallocate_async(self, module: nn.Module) -> None:
        """Run :meth:`preallocate` in a background daemon thread."""
        t = threading.Thread(
            target=self._safe_preallocate, args=(module,), daemon=True
        )
        t.start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _safe_preallocate(self, module: nn.Module) -> None:
        try:
            self.preallocate(module)
        except Exception as e:
            logger.warning(f"PinnedBufferPool: async preallocation failed: {e}")
            self._failed = True
            self._ready.set()
