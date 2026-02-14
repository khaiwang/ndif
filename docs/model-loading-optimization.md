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
  → Evictions: HOT→WARM  ★ BLOCKING ★       [GPU→CPU, 5-60s per model]
  → Ray actor creation                       [I/O, 1-5s]
  → Provider connections (MinIO, Socket.IO)  [I/O, <1s]
  → CUDA context init                        [GPU, <1s]
  → Model load from disk + GPU dispatch  ★   [I/O+CPU+GPU, 30s-10min+]
  → Security sandbox setup                   [CPU, <1s]
  → Polling for actor readiness              [I/O, 0-few seconds]
  → Worker loops start                       [CPU, <1ms]
  → READY
```

**Total cold-deploy time:** ~40s (7B model) to ~15min+ (70B model)

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

#### Step 4b: Apply Delta — Evictions (HOT → WARM) ★ CRITICAL BOTTLENECK ★
- **File:** `src/services/ray/src/ray/deployments/controller/controller.py:287-325`
- **What:** Calls `actor.to_cache.remote()` for each model being evicted. This moves the entire model from GPU to CPU (`self.model._module.cpu()`), then empties CUDA cache.
- **Bound:** GPU→CPU transfer over PCIe. 5-60s per model depending on size and PCIe bandwidth.
- **Dependencies:** Step 4a.
- **PROBLEM:** The Controller **blocks synchronously** (`ray.get(future)` at line 312) waiting for ALL evictions to complete before proceeding to any new deployments. This means new actor creation cannot even begin until every eviction finishes.

#### Step 4b: Apply Delta — From Cache (WARM → HOT)
- **File:** `src/services/ray/src/ray/deployments/controller/controller.py:328-340`
- **What:** Calls `actor.from_cache.remote(gpu_mem_bytes_by_id)` to move model CPU→GPU via `dispatch_model()`.
- **Bound:** CPU→GPU transfer. 5-60s depending on model size.
- **Dependencies:** Step 4b evictions must complete first (currently).
- **Note:** Monitored asynchronously via `_monitor_deployment()` — this part is already non-blocking.

#### Step 5: Ray Actor Creation (New Deployment)
- **File:** `src/services/ray/src/ray/deployments/controller/cluster/deployment.py:110-135`
- **What:** Creates `ModelActor` as a detached Ray actor with `num_cpus=0, num_gpus=0` (NDIF manages GPUs itself). Sets `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`.
- **Bound:** I/O (Ray scheduling). 1-5s.
- **Dependencies:** Step 4b evictions must complete first (currently).
- **Note:** Also monitored asynchronously via `_monitor_deployment()`.

#### Step 6: ModelActor.__init__ — The Heavy Lift
- **File:** `src/services/ray/src/ray/deployments/modeling/base.py:46-123`

##### 6a: Provider Connections
- Lines 62-63: `ObjectStoreProvider.connect()` and `SioProvider.connect()`
- **Bound:** I/O, <1s.

##### 6b: CUDA Context Init
- Lines 88-99: Sets default CUDA device, optionally sets per-process GPU memory fraction.
- **Bound:** GPU, <1s (~400MiB CUDA context).

##### 6c: Model Load from Disk ★ DOMINANT BOTTLENECK ★
- **File:** `base.py:169-200`
- **What:** `RemoteableMixin.from_model_key(model_key, device_map="auto", max_memory=max_memory, dispatch=True, torch_dtype=self.dtype)` — this single call does everything:
  1. Downloads model weights from HuggingFace Hub (if not cached on disk)
  2. Deserializes weights from disk into CPU memory
  3. Dispatches (moves) weights to GPU(s) via accelerate `device_map="auto"`
- **Bound:** Mixed — I/O (disk/network), CPU (deserialization), GPU (PCIe transfer).
- **Estimated times:**
  - 7B model (14GB bf16), local disk: 30-120s
  - 70B model (140GB bf16), local disk: 3-10min
  - First-time download: add minutes to hours depending on bandwidth
- **Dependencies:** Steps 6a, 6b.
- **Retry logic:** `load_with_cache_deletion_retry()` in `util.py:63-89` evicts LRU HuggingFace cache entries and retries if disk space is insufficient.

##### 6d: Security Sandbox Setup
- Lines 103-123: Wraps persistent modules in `protect()`, creates `Protector`, disables gradients, empties CUDA cache, creates thread pool.
- **Bound:** CPU, <1s.

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
- No pinned memory is used anywhere, causing stalls in CPU↔GPU communication

**Synthetic benchmarks** on the Delta parallel file system's NVME storage pool confirmed:
- Pinned memory delivers **substantial performance gains** for host↔device data movement
- Current batching strategies are suboptimal for I/O throughput

| Finding | Proposed Mitigation |
|---------|-------------------|
| I/O bandwidth underutilization | Improved batching strategies for efficient data transfer |
| Stalls in model loading and CPU–GPU communication | Adoption of pinned memory for accelerated data transfers |

### Current Transfer Implementation (No Optimizations)

Inspection of `base.py` confirms zero transfer optimizations are in use:

- **`to_cache()` (GPU→CPU):** Uses bare `.cpu()` — no pinned memory, no non-blocking, no streams
- **`from_cache()` (CPU→GPU):** Uses `accelerate.dispatch_model()` which internally calls `.to(device)` — no pinned memory, no non-blocking
- **`load_from_disk()` (disk→GPU):** Uses `from_model_key(..., dispatch=True)` — single monolithic call, no pipeline
- **No CUDA streams** for overlapping transfers with compute
- **`torch.cuda.synchronize()` calls commented out** in `to_cache()` but active in `from_cache()`

---

## Parallelization Opportunities

### P0: Overlap Evictions with New Actor Creation (HIGH IMPACT)

**Problem:** `controller.py:312` — `ray.get(future)` blocks on ALL HOT→WARM evictions before starting any new deployments or from-cache operations.

**Why it's wasteful:** Evicting model A (GPU→CPU) and creating model B's actor (Ray scheduling + provider connections + CUDA init + disk read) are independent operations, often on different GPUs. The new actor doesn't need the freed GPU memory until the final `dispatch_model()` step.

**Proposed fix:**
1. Start evictions (non-blocking)
2. Immediately start new actor creation / from-cache operations
3. Have the actor's `load_from_disk()` wait for a signal that GPU memory is actually available before the final GPU dispatch step
4. Or: restructure `from_model_key` to do download+deserialize (CPU) separately from dispatch (GPU)

**Estimated savings:** 5-60s (the full eviction time, overlapped with actor startup)

**Complexity:** Medium. Requires restructuring `apply()` and potentially splitting the model load into CPU-stage and GPU-stage.

### P1: Enable hf_transfer for Faster Downloads (MEDIUM IMPACT)

**Problem:** Model downloads from HuggingFace Hub use default single-threaded HTTP. For first-time loads or cache misses, this is a major bottleneck.

**Current state:**
- `huggingface_hub` 0.36.2 is installed (supports `hf_transfer`)
- `hf_transfer` is NOT installed
- No `HF_HUB_ENABLE_HF_TRANSFER` env var is set

**What `hf_transfer` does:**
- Rust-based parallel download library
- Splits each large file into chunks downloaded concurrently (~100 parallel chunks by default)
- Can reach 500MB/s+ on high-bandwidth networks
- 2-10x speedup per shard file

**Limitation:** `hf_transfer` parallelizes within a single file but downloads shard files sequentially ([huggingface_hub#1831](https://github.com/huggingface/huggingface_hub/issues/1831)). For a 70B model with 15 shards, you still download one shard at a time.

**Future:** `hf_xet` (in `huggingface_hub` v1.0+) replaces `hf_transfer` entirely and parallelizes at the chunk level across ALL files. This is the long-term solution but requires upgrading `huggingface_hub` and testing `nnsight` compatibility.

**Proposed fix (Option A — quick win):**
1. Add `hf-transfer` to `src/services/ray/requirements.in`
2. Set `HF_HUB_ENABLE_HF_TRANSFER=1` in Ray container env
3. Optionally set `HF_HUB_DOWNLOAD_TIMEOUT=3600` (default 10s is too short for large models)

**Estimated savings:** 2-10x faster per-file download on first load. Most impactful for models not yet in disk cache.

**Complexity:** Low. Just a package install + env var.

### P2: Parallelize Multi-Model Evaluation (MEDIUM IMPACT)

**Problem:** `cluster.py:193` calls `self.evaluator(model_key)` sequentially in a loop. Each evaluation loads the model on meta device (1-30s uncached).

**Why it matters:** When deploying multiple models simultaneously (e.g., at startup via `NDIF_DEPLOYMENTS`), evaluations are serialized even though they're independent.

**Proposed fix:** Use `concurrent.futures.ThreadPoolExecutor` or `asyncio.gather()` to evaluate models in parallel.

**Estimated savings:** N×(1-30s) → max(1-30s) for N models. Only relevant when deploying multiple models at once.

**Complexity:** Low.

### P3: Pre-download Model Weights During Evaluation (MEDIUM-HIGH IMPACT)

**Problem:** Model evaluation (Step 3a) uses meta device (no actual download), but the real download happens much later in Step 6c. The time between evaluation and actual loading is wasted — evictions, scheduling, actor creation all happen while the disk sits idle.

**Proposed fix:** After evaluation, trigger a background `snapshot_download()` or `from_pretrained(..., device_map="meta")` that pulls the weight files to disk cache without loading into memory. By the time the actor starts `load_from_disk()`, files are already on disk.

**Estimated savings:** Potentially minutes for first-time loads (overlapping download with eviction + actor creation).

**Complexity:** Medium-High. Need to manage background downloads without interfering with cache eviction logic.

### P4: Separate Download/Deserialize from GPU Dispatch (MEDIUM IMPACT)

**Problem:** `from_model_key(..., dispatch=True)` in Step 6c does three things in one call:
1. Download weights (I/O bound — network)
2. Deserialize weights (CPU bound)
3. Dispatch to GPU (GPU bound — PCIe)

These have different resource requirements and could overlap with other operations.

**Proposed fix:** Split into two phases:
1. `from_model_key(..., dispatch=False)` — download + deserialize to CPU
2. `dispatch_model()` — move from CPU to GPU

Phase 1 can start before GPU memory is available (e.g., while evictions are in progress).

**Estimated savings:** Overlaps download+deserialize time with eviction time. For a 70B model, this could save 2-5 minutes.

**Complexity:** Medium. The `dispatch=False` path already exists (used by the evaluator), and `dispatch_model()` is already a separate function in accelerate.

### P5: Overlap Provider Connections with CUDA Init (LOW IMPACT)

**Problem:** MinIO/Socket.IO connections (6a) and CUDA context init (6b) are sequential in `__init__`.

**Proposed fix:** Run them concurrently.

**Estimated savings:** <1s.

**Complexity:** Low.

### P6: Use Pinned Memory for CPU↔GPU Transfers (HIGH IMPACT)

**Problem:** All CPU↔GPU transfers use standard pageable memory. The Delta team's profiling confirmed this causes stalls and underutilizes PCIe bandwidth. Standard `.cpu()` and `.to(device)` allocate pageable host memory, which requires an extra copy through a pinned staging buffer internally — doubling the effective transfer work.

**Current code (no pinned memory):**
```python
# to_cache(): GPU → pageable CPU memory
self.model._module = self.model._module.cpu()

# from_cache(): pageable CPU memory → GPU via accelerate
self.model._module = dispatch_model(self.model._module, device_map)
```

**What pinned memory does:**
- Pinned (page-locked) memory bypasses the OS page cache, allowing DMA transfers directly between GPU and host memory
- Eliminates the extra copy through the staging buffer
- Enables `non_blocking=True` transfers that overlap with computation
- Typical speedup: **1.5-3x** for large contiguous transfers on PCIe 4.0

**Proposed fix — `to_cache()` (GPU→CPU with pinned memory):**
```python
# Pre-allocate pinned CPU tensors, then copy non-blocking
for name, param in self.model._module.named_parameters():
    pinned = torch.empty_like(param, device='cpu', pin_memory=True)
    pinned.copy_(param.data, non_blocking=True)
    param.data = pinned
torch.cuda.synchronize()  # wait for all async copies
```

**Proposed fix — `from_cache()` (CPU→GPU with pinned memory):**
```python
# If model is already in pinned memory (from to_cache), transfers are fast
# Ensure non_blocking=True is used in dispatch
for name, param in self.model._module.named_parameters():
    param.data = param.data.to(target_device, non_blocking=True)
torch.cuda.synchronize()
```

**Note:** This requires either patching accelerate's `dispatch_model()` or replacing it with a custom dispatch that uses `non_blocking=True`. Alternatively, pin the memory during `to_cache()` so that subsequent `from_cache()` transfers automatically benefit.

**Estimated savings:** 1.5-3x faster CPU↔GPU transfers. For a 70B model (140GB), reducing transfer from ~60s to ~20-40s.

**Complexity:** Medium. Need to manage pinned memory lifecycle (it's a limited resource) and potentially replace `dispatch_model()` with custom transfer logic.

### P7: Improved I/O Batching for Data Transfer (MEDIUM IMPACT)

**Problem:** The Delta team's benchmarks show I/O bandwidth underutilization during model loading. Current transfers move one parameter tensor at a time through `dispatch_model()`, which iterates parameters sequentially. Small tensors (bias terms, layer norms) create many small transfers with high per-transfer overhead.

**Proposed fix:**
1. **Batch small tensors:** Coalesce small parameter tensors into larger contiguous buffers before transfer, then scatter back to individual tensors on the target device.
2. **Pipeline transfers with CUDA streams:** Use multiple CUDA streams to overlap PCIe transfers with GPU-side memory operations:
   ```python
   streams = [torch.cuda.Stream() for _ in range(num_streams)]
   for i, (name, param) in enumerate(model.named_parameters()):
       with torch.cuda.stream(streams[i % num_streams]):
           param.data = param.data.to(device, non_blocking=True)
   torch.cuda.synchronize()
   ```
3. **Double-buffering:** While one batch of parameters is being transferred, prepare the next batch in a second pinned buffer. This keeps the PCIe bus continuously saturated.

**Estimated savings:** 10-30% additional throughput on top of pinned memory gains, primarily from reducing per-transfer overhead for small tensors.

**Complexity:** Medium-High. Requires custom transfer pipeline replacing `dispatch_model()`.

---

## Priority Matrix

| ID | Optimization | Savings | Complexity | Dependencies |
|----|-------------|---------|------------|--------------|
| **P1** | Enable `hf_transfer` | 2-10x download speed | Low | None |
| **P6** | Pinned memory for CPU↔GPU transfers | 1.5-3x faster transfers | Medium | None |
| **P0** | Overlap evictions with actor creation | 5-60s | Medium | None |
| **P4** | Separate download/deserialize from GPU dispatch | Minutes (large models) | Medium | Benefits from P0 |
| **P7** | I/O batching + CUDA streams for transfers | 10-30% on top of P6 | Medium-High | Benefits from P6 |
| **P2** | Parallelize multi-model evaluation | Seconds per extra model | Low | None |
| **P3** | Pre-download weights during evaluation | Minutes (first load) | Medium-High | Benefits from P1 |
| **P5** | Overlap provider connections + CUDA init | <1s | Low | None |

**Recommended execution order:** P1 → P6 → P0 → P4 → P7 → P2 → P3 → P5

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
| `src/services/ray/src/ray/deployments/modeling/util.py` | Cache eviction retry logic |
| `src/services/ray/requirements.in` | Ray service dependencies |
| `docker/docker-compose.yml` | Container env vars, volume mounts |
