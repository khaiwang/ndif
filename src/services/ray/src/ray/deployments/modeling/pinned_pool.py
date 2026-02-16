"""Pre-allocated pinned (page-locked) memory pool for fast GPU↔CPU transfers.

The pool collects all parameters and buffers from a model, deduplicates by
``data_ptr()`` (handling tied parameters), packs them sequentially into ≤2 GiB
``torch.uint8`` chunks with 4 KiB alignment, and exposes typed views that can
be used as drop-in replacements for the original tensors.

Allocation runs in a background thread so the model can keep serving while
pinned memory is being registered.
"""

import gc
import logging
import threading
import time
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("ndif")

# ≤2 GiB per chunk to avoid cudaHostAlloc limitations with large single allocs.
_MAX_CHUNK_BYTES = 2 * (1024 ** 3)
# 4 KiB alignment for optimal DMA page alignment.
_ALIGNMENT = 4096


def _align(offset: int) -> int:
    """Round *offset* up to the next ``_ALIGNMENT`` boundary."""
    return (offset + _ALIGNMENT - 1) & ~(_ALIGNMENT - 1)


class PinnedBufferPool:
    """A contiguous pinned-memory pool sized to exactly one model's state."""

    def __init__(self) -> None:
        self._chunks: list[torch.Tensor] = []
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
        seen_ptrs: set[int] = set()
        expected: list[tuple[str, torch.Size, torch.dtype]] = []

        for name, param in module.named_parameters():
            ptr = param.data.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            expected.append((name, param.data.shape, param.data.dtype))

        for name, buf in module.named_buffers():
            ptr = buf.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            expected.append((name, buf.shape, buf.dtype))

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
        self._chunks.clear()
        gc.collect()

    # ------------------------------------------------------------------
    # Allocation (synchronous + async wrapper)
    # ------------------------------------------------------------------

    def preallocate(self, module: nn.Module) -> None:
        """Allocate pinned buffers matching every unique tensor in *module*."""
        start = time.time()

        # 1. Collect unique tensors, deduplicate by data_ptr (tied params).
        seen_ptrs: set[int] = set()
        entries: list[tuple[str, torch.Tensor]] = []

        for name, param in module.named_parameters():
            ptr = param.data.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            entries.append((name, param.data))

        for name, buf in module.named_buffers():
            ptr = buf.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            entries.append((name, buf))

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

        # 3. Allocate pinned chunks with per-chunk failure cleanup.
        chunks: list[torch.Tensor] = []
        try:
            for size in chunk_sizes:
                chunks.append(torch.empty(size, dtype=torch.uint8, pin_memory=True))
        except Exception as e:
            logger.warning(
                f"PinnedBufferPool: chunk allocation failed after "
                f"{len(chunks)}/{len(chunk_sizes)} chunks: {e}"
            )
            chunks.clear()
            gc.collect()
            self._failed = True
            self._ready.set()
            return

        self._chunks = chunks
        self._plan = plan

        # 4. Create typed views into the chunks.
        for chunk_idx, offset, nbytes, dtype, shape, name in plan:
            raw = self._chunks[chunk_idx][offset : offset + nbytes]
            self._views[name] = raw.view(dtype).reshape(shape)

        self._alloc_time = time.time() - start
        self._ready.set()

        logger.info(
            f"PinnedBufferPool allocated {self._total_bytes / (1024**3):.2f} GiB "
            f"in {len(self._chunks)} chunk(s) for {len(self._views)} tensors "
            f"({self._alloc_time:.2f}s)"
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
