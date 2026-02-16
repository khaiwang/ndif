# Model Loading Performance: Analysis & Optimization Plan

## Current Loading Path (Sequential)

The model loading path from deployment request to inference-ready is fully sequential today. Here is every step, its cost, and what bounds it.

### End-to-End Flow

```
Client Request
  → Redis queue pop                          [I/O, <1ms]
  → Processor creation                       [CPU, <1ms]
  → Controller RPC (dedicated check)         [I/O, 10-100ms]
  → Model evaluation (meta device)           [CPU+I/O, 1-30s; cached: <1ms]
  → GPU scheduling / node evaluation         [CPU, <1ms]
  → Resource reservation                     [CPU, <1ms]
  → Build delta (diff current vs desired)    [CPU, <1ms]
  → Evictions: HOT→WARM  (non-blocking)      [GPU→CPU, 0.7-10s per model (pinned)]
  → Ray actor creation  (overlapped w/ ↑)    [I/O, 1-5s]
  → Provider connections (overlapped w/ ↑)   [I/O, <1s]
  → CUDA context init   (overlapped w/ ↑)    [GPU, <1s]
  → Model load to CPU   (overlapped w/ ↑)    [I/O+CPU, 30s-10min+]
  → GPU dispatch         (after evictions)   [GPU, 0.5-10s]
  → Security sandbox setup                   [CPU, <1s]
  → Polling for actor readiness              [I/O, 0-few seconds]
  → Worker loops start                       [CPU, <1ms]
  → READY
```

**Total cold-deploy time:** ~11s (7B, cached on disk) to ~15min+ (70B, first download)
**Total warm cycle (evict+reload):** ~1.5s (7B) with pinned memory (was ~7s)

### Detailed Step Breakdown

#### Step 1: Request Arrival & Processor Creation
- **File:** `src/services/api/src/queue/dispatcher.py:173-201`
- **What:** Dispatcher pops `BackendRequestModel` from Redis via `brpop("queue", timeout=1)`. Creates a Processor for the model if none exists, starts `processor_worker()` as an asyncio task.
- **Bound:** I/O (Redis). Sub-millisecond on localhost.
- **Dependencies:** None (entry point).

#### Step 2: Processor Provisioning (UNINITIALIZED → PROVISIONING)
- **File:** `src/services/api/src/queue/processor.py:259-395`
- **What:** Queries ControllerActor for dedicated status, filters non-hotswap requests, then calls `Controller.deploy()` via Ray RPC.
- **Bound:** I/O (Ray RPCs). 10-100ms.
- **Dependencies:** Step 1.

#### Step 3a: Model Evaluation / Sizing
- **File:** `src/services/ray/src/ray/deployments/controller/cluster/evaluator.py:47-87`
- **What:** For each model key, loads the model on meta device (zero memory) to count parameters and compute size. Adds 15% padding. Result is cached for future calls.
- **Bound:** CPU+I/O on first call (meta-device instantiation downloads config from HuggingFace if not cached). 1-30s uncached, <1ms cached.
- **Dependencies:** Step 2.
- **Note:** When deploying multiple models, evaluations run sequentially in a loop (`cluster.py:193`).

#### Step 3b: GPU Scheduling / Node Evaluation
- **File:** `src/services/ray/src/ray/deployments/controller/cluster/node.py:418-505`
- **What:** For each model+replica, evaluates all nodes. Checks: already deployed? In warm cache? Fits on fractional GPU (model_size × 3.0 < 80% GPU)? Needs multi-GPU? Needs evictions?
- **Bound:** CPU, <1ms. Pure bookkeeping.
- **Dependencies:** Step 3a (needs model size).

#### Step 3c: Resource Reservation
- **File:** `src/services/ray/src/ray/deployments/controller/cluster/node.py:209-261`
- **What:** Performs evictions, assigns GPU memory or full GPUs, creates `Deployment` object with `DeploymentLevel.HOT`.
- **Bound:** CPU, <1ms.
- **Dependencies:** Step 3b.

#### Step 4a: Build Delta
- **File:** `src/services/ray/src/ray/deployments/controller/controller.py:222-279`
- **What:** Compares current state with new cluster state. Produces `DeploymentDelta` with four lists: `to_cache`, `from_cache`, `to_create`, `to_delete`.
- **Bound:** CPU, <1ms.
- **Dependencies:** Step 3c.

#### Step 4b: Apply Delta — Evictions (HOT → WARM)
- **File:** `src/services/ray/src/ray/deployments/controller/controller.py`
- **What:** Calls `actor.to_cache.remote()` for each model being evicted. Uses pinned buffer pool for DMA copies (non_blocking), then empties CUDA cache.
- **Bound:** GPU→CPU DMA transfer over PCIe. 0.7-10s per model with pinned memory (was 5-60s with pageable).
- **Dependencies:** Step 4a.
- **Note:** Evictions still block synchronously (`ray.get(future)`) but new actor creation (Step 5+6) now runs **in parallel** with evictions, overlapping the CPU-only work. Only the final GPU dispatch waits for evictions.

#### Step 4c: Apply Delta — From Cache (WARM → HOT)
- **File:** `src/services/ray/src/ray/deployments/controller/controller.py`
- **What:** Calls `actor.from_cache.remote(target_gpus)` to move model CPU→GPU. Single-GPU: chunk bulk transfer (~8 chunks instead of ~200 params) from pinned memory, skips device_map. Multi-GPU: falls back to `dispatch_model()`.
- **Bound:** CPU→GPU transfer. 0.7-10s with pinned memory (was 5-60s).
- **Dependencies:** Step 4b evictions must complete first.
- **Note:** Monitored asynchronously via `_monitor_deployment()` — this part is already non-blocking.

#### Step 5: Ray Actor Creation (New Deployment) — Overlapped with Evictions
- **File:** `src/services/ray/src/ray/deployments/controller/cluster/deployment.py:100-129`
- **What:** Creates `ModelActor` as a detached Ray actor with `num_cpus=2, num_gpus=0` (NDIF manages GPUs itself). Sets `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`.
- **Bound:** I/O (Ray scheduling). 1-5s.
- **Dependencies:** Step 4a. Starts **immediately** after evictions are fired (not after they complete).
- **Note:** Monitored asynchronously via `_monitor_create_and_dispatch()`.

#### Step 6: ModelActor.__init__ — CPU Phase (Overlapped with Evictions)
- **File:** `src/services/ray/src/ray/deployments/modeling/base.py:46-143`

##### 6a: Provider Connections
- Lines 62-63: `ObjectStoreProvider.connect()` and `SioProvider.connect()`
- **Bound:** I/O, <1s.

##### 6b: CUDA Context Init
- Lines 88-99: Sets default CUDA device, configures cudaHostRegister allocator.
- **Bound:** GPU, <1s (~400MiB CUDA context).

##### 6c: Model Load from Disk to CPU
- **File:** `base.py:214-247`
- **What:** `RemoteableMixin.from_model_key(model_key, device_map="cpu", dispatch=True, torch_dtype=self.dtype)` — loads weights to CPU only:
  1. Downloads model weights from HuggingFace Hub (if not cached on disk)
  2. Deserializes weights from disk into CPU memory
- **Bound:** I/O (disk/network) + CPU (deserialization). No GPU needed.
- **Estimated times:**
  - 7B model (14GB bf16), local disk: 4-8s
  - 70B model (140GB bf16), local disk: 30s-3min
  - First-time download: add minutes to hours depending on bandwidth
- **Dependencies:** Steps 6a, 6b. Runs **in parallel** with evictions.
- **Retry logic:** `load_with_cache_deletion_retry()` in `util.py:63-89` evicts LRU HuggingFace cache entries and retries if disk space is insufficient.

##### 6d: Security Sandbox Setup
- Lines 105-117: Wraps persistent modules in `protect()`, creates `Protector`, disables gradients, creates thread pool.
- **Bound:** CPU, <1s.

#### Step 6e: GPU Dispatch (After Evictions Complete)
- **File:** `base.py:248-291`
- **What:** `dispatch_to_gpu()` — called by the controller after evictions finish. Moves model from CPU to target GPUs:
  - Single-GPU: multi-stream (4 streams) `.to(target, non_blocking=True)` + manual `hf_device_map`
  - Multi-GPU: accelerate `_get_device_map()` + `dispatch_model()`
- **Bound:** GPU (PCIe transfer). 0.5-10s.
- **Dependencies:** Step 6c (CPU load complete) + Step 4b (evictions complete).
- **Note:** Also triggers pinned buffer pool pre-allocation for future eviction cycles.

#### Step 7: Processor Readiness (DEPLOYING → READY)
- **File:** `src/services/api/src/queue/processor.py:428-461`
- **What:** Spawns `_initialize_and_start_replica()` for each replica concurrently. Each polls `__ray_ready__` until the actor's `__init__` completes.
- **Bound:** Waiting for Step 6. Polling adds ≤1s overhead per retry.
- **Note:** Multiple replicas are already initialized concurrently via `asyncio.wait()`.

---

## Delta Team Profiling Findings

NVIDIA NNsight profiling by the Delta team identified the following phases in the model lifecycle:
1. Model download (HuggingFace Hub → disk)
2. Loading model to GPU (disk → CPU → GPU)
3. Hot-swapping model GPU → CPU memory (warm caching)
4. Reloading model CPU → GPU

**Key observations:**
- No concurrent model loading and hot-swapping — these phases are fully serialized
- I/O bandwidth is underutilized during transfers
- ~~No pinned memory is used anywhere, causing stalls in CPU↔GPU communication~~ → **Fixed by P6**

**Synthetic benchmarks** on the Delta parallel file system's NVME storage pool confirmed:
- Pinned memory delivers **substantial performance gains** for host↔device data movement
- Current batching strategies are suboptimal for I/O throughput

| Finding | Proposed Mitigation |
|---------|-------------------|
| I/O bandwidth underutilization | Improved batching strategies for efficient data transfer |
| Stalls in model loading and CPU–GPU communication | Adoption of pinned memory for accelerated data transfers |

### Current Transfer Implementation (After P6+P7 Optimization)

- **`to_cache()` (GPU→CPU):** Pre-allocated pinned buffer pool + `non_blocking=True` DMA copies. Falls back to `.cpu()` if pool unavailable.
- **`from_cache()` (CPU→GPU):** Single-GPU fast path with chunk bulk transfer — transfers ~8 contiguous pinned chunks to GPU instead of ~200 individual `.to()` calls, then reassigns `param.data` to GPU-side views. Multi-GPU falls back to `dispatch_model()`.
- **`load_from_disk()` (disk→CPU):** Uses `from_model_key(..., device_map="cpu")` — loads weights to CPU only. Runs in parallel with evictions.
- **`dispatch_to_gpu()` (CPU→GPU):** Called after evictions complete. Single-GPU: multi-stream (4 streams) round-robin transfer to overlap CUDA API dispatch overhead. Multi-GPU: accelerate `dispatch_model()`.
- **Pool reuse:** Pinned buffer pool is reused across eviction cycles via `matches()` — no reallocation overhead. Pre-allocated after `dispatch_to_gpu()` when GPU placement is known.
- **Chunk layout:** `PinnedBufferPool` stores its packing plan (`_plan`) so `transfer_to_device()` can create GPU-side typed views from bulk-transferred chunks.

---

## Parallelization Opportunities

### P0+P4: Overlap Evictions with New Deployments + Two-Phase Load ✅ IMPLEMENTED

**Status:** Fully implemented (2026-02-15).

**What was done:**
1. **Two-phase model loading** in `base.py`: `load_from_disk()` now uses `device_map="cpu"` (CPU only, no GPU needed). New `dispatch_to_gpu()` method moves model to target GPUs after evictions complete.
   - Single-GPU fast path: direct `.to(target, non_blocking=True)` + manual `hf_device_map`
   - Multi-GPU: accelerate `_get_device_map()` + `dispatch_model()`
   - Pinned buffer pool pre-allocation moved to after `dispatch_to_gpu()` (needs GPU placement)
2. **Overlapped apply()** in `controller.py`: Actor creation (CPU load) starts immediately in parallel with evictions. After evictions complete, `_monitor_create_and_dispatch()` calls `dispatch_to_gpu()` on each new actor, then `from_cache` operations proceed.

**Timeline improvement:**
```
Before:  [==== evict ====] → [== spawn ==][== CPU load ==][== GPU dispatch ==]
After:   [====== evict ======]
         [== spawn ==][== CPU load ==]
                                      → [GPU dispatch]  (after evict done)
```

**Estimated savings:** The full actor spawn (~2.7s) + CPU load (~4-8s for 7B) time overlapped with evictions.

**Key files:**
- `src/services/ray/src/ray/deployments/modeling/base.py` — `load_from_disk()` (CPU), `dispatch_to_gpu()` (GPU)
- `src/services/ray/src/ray/deployments/controller/controller.py` — `apply()`, `_monitor_create_and_dispatch()`

**Metrics:** `disk_cpu_load` (CPU phase), `disk_gpu_dispatch` (GPU phase), `init_cpu_total` (full __init__), `monitor_create` (end-to-end create+dispatch)

### ~~P1: Enable hf_transfer for Faster Downloads~~ (RULED OUT)

**Status:** Investigated and ruled out (2026-02-15).

**Finding:** `hf_transfer` is **obsolete** in our environment. Our `huggingface_hub` version (1.4.1) has completely removed the `HF_HUB_ENABLE_HF_TRANSFER` env var — it is silently ignored. The library has been replaced by `hf_xet`, which is already installed (v1.2.0) and enabled by default (`HF_HUB_DISABLE_XET=False`).

However, `hf_xet` only accelerates downloads for model repos that have opted into **xet-enabled storage** on HuggingFace. Most models (including Qwen/Qwen2.5-7B-Instruct) have not migrated yet, so downloads fall back to standard HTTPS. We tested with `HF_HUB_ENABLE_HF_TRANSFER=1` and `hf-transfer` installed — download time was 115.9s vs the 117s baseline, confirming no benefit.

**What would help instead:**
- Pre-cache models on persistent storage so downloads only happen once
- Wait for HuggingFace to roll out xet storage more broadly (migration is ongoing)
- Use a model mirror/proxy closer to the network

### P2: Parallelize Multi-Model Evaluation ✅ IMPLEMENTED

**Status:** Fully implemented (2026-02-15).

**What was done:**
1. **Thread-safe evaluator cache** in `evaluator.py`: Added `threading.Lock` to protect `self.cache` reads/writes, allowing concurrent evaluations without race conditions. The lock is only held during cache lookups and writes — the actual evaluation (meta-device model loading) runs without the lock held.
2. **Parallel evaluation** in `cluster.py`: Replaced sequential dict comprehension with `ThreadPoolExecutor` (capped at 4 workers to avoid overwhelming HuggingFace Hub or CPU).

**Estimated savings:** N×(1-30s) → max(1-30s) for N uncached models. Only relevant when deploying multiple models at once (e.g., startup via `NDIF_DEPLOYMENTS`).

**Key files:**
- `src/services/ray/src/ray/deployments/controller/cluster/evaluator.py` — `_cache_lock` for thread-safety
- `src/services/ray/src/ray/deployments/controller/cluster/cluster.py` — `ThreadPoolExecutor` parallel evaluation

### P3: Pre-download Model Weights During Evaluation (MEDIUM-HIGH IMPACT)

**Problem:** Model evaluation (Step 3a) uses meta device (no actual download), but the real download happens much later in Step 6c. The time between evaluation and actual loading is wasted — evictions, scheduling, actor creation all happen while the disk sits idle.

**Proposed fix:** After evaluation, trigger a background `snapshot_download()` or `from_pretrained(..., device_map="meta")` that pulls the weight files to disk cache without loading into memory. By the time the actor starts `load_from_disk()`, files are already on disk.

**Estimated savings:** Potentially minutes for first-time loads (overlapping download with eviction + actor creation).

**Complexity:** Medium-High. Need to manage background downloads without interfering with cache eviction logic.

### ~~P4: Separate Download/Deserialize from GPU Dispatch~~ (MERGED INTO P0)

**Status:** Implemented as part of P0+P4 above. The two-phase load (`device_map="cpu"` → `dispatch_to_gpu()`) is the core mechanism that enables overlapping CPU work with evictions.

### P5: Overlap Provider Connections with CUDA Init (LOW IMPACT)

**Problem:** MinIO/Socket.IO connections (6a) and CUDA context init (6b) are sequential in `__init__`.

**Proposed fix:** Run them concurrently.

**Estimated savings:** <1s.

**Complexity:** Low.

### P6: Use Pinned Memory for CPU↔GPU Transfers ✅ IMPLEMENTED

**Status:** Fully implemented and measured (2026-02-15).

**What was done:**
1. **`PinnedBufferPool`** (`pinned_pool.py`): Pre-allocates contiguous pinned (page-locked) memory chunks (≤2 GiB each, 4 KiB aligned) sized to the model's full state. Background thread allocation so serving isn't blocked. Uses `cudaHostRegister` with 8 threads for efficient page registration.
2. **`to_cache()` (GPU→CPU):** DMA copies each parameter/buffer into the pre-allocated pinned buffer via `non_blocking=True`, then points `param.data` at the pinned view. Falls back to `.cpu()` if the pool isn't ready.
3. **`from_cache()` (CPU→GPU):** Single-GPU fast path — directly calls `.to(target_device, non_blocking=True)` from pinned memory, skips `_get_device_map()` and `dispatch_model()` entirely, manually sets `hf_device_map`. Multi-GPU falls back to `dispatch_model()`.
4. **Pool reuse:** After reload, the pool is reused across evict/reload cycles via `matches()` check (same parameter names, shapes, dtypes) instead of freeing and reallocating. Only reallocates if the model changes (which doesn't happen for the same actor).

**Measured results (Qwen2.5-7B-Instruct, ~15 GB, single RTX 4090):**

| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| GPU→CPU transfer | 5.13s | 0.59s | **8.7x** |
| CPU→GPU transfer | 1.58s | 0.59s | **2.7x** |
| device_map computation | 0.003s | 0.00s (skipped) | — |
| Evict total | 5.26s | 0.73s | **7.2x** |
| Reload total | 1.73s | 0.73s | **2.4x** |
| **Full warm cycle** | **~7s** | **~1.5s** | **~5x** |

**Key files:**
- `src/services/ray/src/ray/deployments/modeling/pinned_pool.py` — PinnedBufferPool
- `src/services/ray/src/ray/deployments/modeling/base.py` — to_cache, from_cache, pool lifecycle

### P7: Chunk-Based Bulk Transfers for CPU↔GPU ✅ IMPLEMENTED

**Status:** Fully implemented (2026-02-15).

**What was done:**
1. **`from_cache()` chunk bulk transfer:** Instead of ~200 individual `.to()` calls (one per parameter), `PinnedBufferPool.transfer_to_device()` transfers ~8 contiguous uint8 chunks to GPU in bulk, then creates typed GPU-side views. `from_cache()` reassigns `param.data` and `buf.data` to these views. Reduces Python loop overhead, CUDA API overhead, and driver dispatch overhead.
2. **`dispatch_to_gpu()` multi-stream:** Uses 4 CUDA streams with round-robin assignment to overlap CUDA API/driver dispatch overhead across streams. Since model tensors loaded from disk are scattered in CPU memory (not in the pinned pool), chunk bulk transfer isn't applicable here.
3. **`to_cache()` skipped:** GPU tensors are scattered across GPU memory, so coalescing would require an extra GPU staging buffer. The current 0.59s is already near PCIe 4.0 limits.

**Key implementation details:**
- `PinnedBufferPool._plan` now persists the chunk packing layout (chunk_idx, offset, nbytes, dtype, shape, name) from `preallocate()`.
- `transfer_to_device(device)` copies raw chunks to GPU, then creates typed views matching the plan offsets — no extra GPU memory needed beyond the model itself.

**Expected impact:** ~10-20% reduction in `from_cache` transfer overhead (fewer CUDA API calls), ~5-10% for `dispatch_to_gpu` (multi-stream overlap). Needs measurement.

**Key files:**
- `src/services/ray/src/ray/deployments/modeling/pinned_pool.py` — `_plan`, `transfer_to_device()`
- `src/services/ray/src/ray/deployments/modeling/base.py` — `from_cache()` bulk path, `dispatch_to_gpu()` multi-stream

---

## Priority Matrix

| ID | Optimization | Savings | Complexity | Status |
|----|-------------|---------|------------|--------|
| **P6** | Pinned memory for CPU↔GPU transfers | ~5x faster warm cycle | Medium | ✅ Done |
| **P1** | ~~Enable `hf_transfer`~~ | — | — | ❌ Ruled out |
| **P0+P4** | Overlap evictions + two-phase load | Overlaps spawn+CPU load with evictions | Medium | ✅ Done |
| **P7** | Chunk bulk transfers + multi-stream dispatch | 10-20% on top of P6 | Medium | ✅ Done |
| **P2** | Parallelize multi-model evaluation | Seconds per extra model | Low | ✅ Done |
| **P3** | Pre-download weights during evaluation | Minutes (first load) | Medium-High | — |
| **P5** | Overlap provider connections + CUDA init | <1s | Low | — |

**Recommended next:** P3 → P5

---

## Key File References

| File | What |
|------|------|
| `src/services/api/src/queue/dispatcher.py` | Request dispatch, Processor lifecycle |
| `src/services/api/src/queue/processor.py` | State machine, provision/initialize flow |
| `src/services/api/src/queue/replicas.py` | Replica readiness polling |
| `src/services/ray/src/ray/deployments/controller/controller.py` | Deploy orchestration, apply delta |
| `src/services/ray/src/ray/deployments/controller/cluster/cluster.py` | Scheduling, model sorting, evaluation loop |
| `src/services/ray/src/ray/deployments/controller/cluster/evaluator.py` | Meta-device model sizing |
| `src/services/ray/src/ray/deployments/controller/cluster/node.py` | GPU allocation, eviction planning |
| `src/services/ray/src/ray/deployments/controller/cluster/deployment.py` | Ray actor creation |
| `src/services/ray/src/ray/deployments/modeling/base.py` | ModelActor init, load_from_disk, to_cache, from_cache |
| `src/services/ray/src/ray/deployments/modeling/pinned_pool.py` | PinnedBufferPool: pre-allocated pinned memory for transfers |
| `src/services/ray/src/ray/deployments/modeling/util.py` | Cache eviction retry logic |
| `src/services/ray/requirements.in` | Ray service dependencies |
| `docker/docker-compose.yml` | Container env vars, volume mounts |
