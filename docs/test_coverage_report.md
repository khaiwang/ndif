# NDIF Test Coverage Report

## Current Coverage Summary

| Layer | Tests | Source modules covered |
|-------|------:|----------------------|
| Unit | 570 | 14 source modules |
| Component | 19 | 2 integration paths |
| Contract | 16 | 2 service boundaries |
| Integration | ~81 | Remote nnsight + security guards |
| **Total** | **~686** | |

---

## What Is Tested

### Unit Tests (570 tests across 14 files)

| Test file | Tests | Source file | Coverage |
|-----------|------:|-----------|----------|
| `test_controller.py` | 82 | `ray/.../controller/controller.py` | **Thorough.** deploy() async wrapper, evict() with actor deletion, check_nodes() periodic discovery, get_state() aggregation, _adjust_desired_for_cant_accommodate(), environment/status endpoints. |
| `test_node.py` | 80 | `ray/.../cluster/node.py` | **Thorough.** `Resources` (gpus_required, assign_full_gpus, assign_memory), `Node` (evaluate all CandidateLevel outcomes, deploy, evict, cache lifecycle), `Candidate`, `CandidateLevel`. |
| `test_processor.py` | 52 | `api/.../queue/processor.py` | **Thorough.** Full state machine (UNINITIALIZED→CANCELLED), enqueue with dedicated/hotswap logic, provision with all DeploymentStatus variants, remove/add replica, kill_request (in-queue, in-flight, not-found), purge, get_state. |
| `test_replicas.py` | 47 | `api/.../queue/replicas.py` | **Thorough.** `Replica` (init, set/clear request, get_handle, submit, wait_until_ready). `Replicas` (add/remove, set_replica_ids, begin/end request, properties, get_state). |
| `test_dispatcher.py` | 47 | `api/.../queue/dispatcher.py` | **Thorough.** Init/connect, dispatch routing, remove/purge, handle_evictions, handle_errors (connection vs non-connection), get_state, all 5 event handlers (_handle_deploy_event, _handle_evict_event, _handle_kill_request, _handle_queue_state_request, _handle_env_event). |
| `test_cluster.py` | 37 | `ray/.../cluster/cluster.py` | **Good.** Construction, target_replica_ids_for, deploy (single/multi model, replicas, evictions, dedicated, cant_accommodate, errors), evict (by model, by replica, not-found), get_state, integration lifecycle tests. |
| `test_processor_workers.py` | 35 | `api/.../queue/processor.py` | **Thorough.** processor_worker() lifecycle, reply_worker() status updates, _replica_worker() dequeue-execute loop, _execute_on_replica() submit/timeout/response, _initialize_and_start_replica() wait-until-ready. |
| `test_request_parsing.py` | 34 | `common/schema/request.py`, `api/.../dependencies.py` | **Good.** from_request() deserialization (multipart body, session_id, callback, hotswapping, model_key), validate_request() full pipeline integration. |
| `test_config.py` | 33 | `api/.../config.py`, `api/.../queue/config.py` | **Thorough.** AppConfig/QueueConfig from_env() parsing, _parse_positive_int() validation, default values, invalid/missing env var handling. |
| `test_object_storage.py` | 30 | `common/schema/mixins.py` | **Thorough.** ObjectStorageMixin (object_name, url, _save/_load, save JSON/PyTorch paths, load with streaming, delete), TensorStoragePickler (GPU→CPU, CPU passthrough, non-tensor), cpu_pickle_module, TelemetryMixin (all log levels). |
| `test_deployment.py` | 29 | `ray/.../cluster/deployment.py` | **Thorough.** DeploymentLevel enum, construction, properties (name, actor, gpus), get_state() serialization, end_time() calculation, delete/restart (with error handling), cache/from_cache (with error handling), create() with Ray options, env vars, provider integration. |
| `test_dispatcher_workers.py` | 28 | `api/.../queue/dispatcher.py` | **Good.** dispatch_worker() Redis BRPOP→deserialize→dispatch loop, status_worker() periodic cluster status push, events_worker() Redis stream read→event routing, error handling and recovery within each loop. |
| `test_dependencies.py` | 20 | `api/.../dependencies.py` | **Good.** authenticate_api_key (dev mode, valid, invalid, missing), validate_python_version, validate_nnsight_version, check_hotswapping_access, require_ray_connection. |
| `test_schema.py` | 16 | `common/schema/response.py`, `common/schema/request.py` | **Partial.** BackendResponseModel (blocking property, str, respond with blocking/non-blocking/callback variants). BackendRequestModel (create_response status transitions). is_email helper. |

### Component Tests (19 tests across 2 files)

| Test file | What it covers |
|-----------|---------------|
| `test_dispatch_flow.py` (14) | Dispatcher↔Processor coordination: request routing, processor reuse, eviction handling (full and per-replica), error recovery (connection vs transient), event handlers (deploy, evict, queue_state, kill). |
| `test_api.py` (5) | FastAPI endpoints via ASGI: `/ping`, `/connected` (ray up/down), `/response/{id}` (found/not-found). |

### Contract Tests (16 tests across 2 files)

| Test file | What it verifies |
|-----------|-----------------|
| `test_controller_contract.py` (9) | `Cluster.deploy()` return format, `Cluster.evict()` return format, `Processor.check_dedicated()` parsing of controller responses, `DeploymentStatus` ↔ `CandidateLevel` enum alignment. |
| `test_model_actor_contract.py` (7) | `ModelActor` interface completeness (__call__, cancel, __ray_ready__, to_cache, from_cache), replica actor naming convention consistency between API and Ray. |

### Integration Tests (~81 tests, require `--run-remote`)

| Test file | What it covers |
|-----------|---------------|
| `test_nnsight.py` | Remote NNsight execution: tracing, generation, activation modification, gradients, sessions, caching, invokers, iteration, adhoc modules, edge cases. |
| `test_security_guards.py` | Protected environment: allowed ops (tensor, torch, numpy, print), blocked ops (os, subprocess, sys, socket, eval, exec, open), guarded attribute access, restricted compilation. |

---

## What Is NOT Tested (Gap Analysis)

### Priority 1 — High-value gaps with moderate effort

#### 1. `BaseModelDeployment` — Model Execution Pipeline
**Source:** `src/services/ray/src/ray/deployments/modeling/base.py`
**Risk:** Core inference execution. Requires heavy mocking of torch/accelerate.
**Missing:**
- `__call__()` full pipeline: pre → execute → post → cleanup
- `to_cache()` / `from_cache()` GPU↔CPU cache transfers
- `load_from_disk()` model loading with device map
- `cancel()` execution cancellation
- `pre()` request deserialization
- `post()` result saving and response sending
- `exception()` error handling
- `stream_send()` / `stream_receive()` streaming data
- `_build_max_memory()` device map construction
- `_verify_device_placement()` GPU placement validation

#### 2. Provider Classes — Connection Management
**Source:** `src/common/providers/` (redis.py, ray.py, objectstore.py, socketio.py, mailgun.py)
**Risk:** Provider bugs cause silent failures or connection leaks.
**Missing:**
- `RedisProvider`: connect(), connected(), reset()
- `RayProvider`: connect(), connected(), reset(), is_connection_error(), is_listening()
- `ObjectStoreProvider`: connect(), from_env()
- `SioProvider`: connect(), disconnect(), connected(), call(), emit()
- `MailgunProvider`: connected(), send_email()

#### 3. `AccountsDB` — Database Access
**Source:** `src/services/api/src/db.py`
**Risk:** Authentication and authorization decisions depend on this.
**Missing:**
- `api_key_exists()` with mocked psycopg2
- `key_has_hotswapping_access()` tier lookup
- `tier_id_from_name()` mapping
- Connection failure handling

### Priority 2 — Nice to have

#### 4. `ModelEvaluator` — Model Size Estimation
**Source:** `src/services/ray/src/ray/deployments/controller/cluster/evaluator.py`
**Risk:** Incorrect size estimates affect deployment decisions.
**Missing:**
- `__call__()` model size calculation and caching
- `CacheEntry` storage
- Error handling for unknown models

#### 5. Metrics Classes
**Source:** `src/common/metrics/` (6 metric classes)
**Risk:** Low — metrics are observability, not control flow.
**Missing:**
- `GPUMemMetric.update()`
- `ModelLoadTimeMetric.update()`
- `ExecutionTimeMetric.update()`
- `RequestResponseSizeMetric.update()`
- `RequestStatusTimeMetric.update()`
- `NetworkStatusMetric.update()`

#### 6. Logging Infrastructure
**Source:** `src/common/logging/logger.py`
**Risk:** Low — structured logging utilities.
**Missing:**
- `CustomJSONFormatter.format()` output format
- `RetryingLokiHandler.emit()` retry logic
- `set_logger()` configuration

#### 7. Security / Protected Environment
**Source:** `src/services/ray/src/ray/nn/security/`
**Risk:** Already well-tested in integration tests (`test_security_guards.py`). Unit tests for internal mechanics could add confidence.
**Missing (unit-level):**
- `Protector` initialization and escape mechanism
- `Importer.__call__()` whitelist enforcement
- `SafeBuiltins` attribute filtering
- `ProtectedModule` attribute restrictions
- `ProtectedObject` runtime wrapping

#### 8. Google Calendar Scheduler
**Source:** `src/services/ray/src/ray/deployments/controller/gcal/`
**Risk:** Low — optional scheduling feature.
**Missing:**
- `SchedulingActor.check_calendar()` event parsing
- `SchedulingControllerActor` scheduled deployment

#### 9. Utility Functions
**Source:** `src/services/api/src/queue/util.py`, `src/services/ray/src/ray/deployments/modeling/util.py`
**Missing:**
- `patch()` Ray deadlock workaround
- `submit()` actor RPC wrapper
- `controller_handle()` / `get_model_actor_handle()` handle caching
- `remove_accelerate_hooks()` model cleanup
- `downloaded()` / `get_downloaded_models()` HF cache checks
- `make_room()` / `load_with_cache_deletion_retry()` disk space management

---

## Coverage by Source Module

| Source module | Unit | Component | Contract | Integration | Overall |
|--------------|:----:|:---------:|:--------:|:-----------:|:-------:|
| `cluster/node.py` | Full | — | — | — | **Strong** |
| `cluster/cluster.py` | Full | — | Partial | — | **Strong** |
| `queue/replicas.py` | Full | — | — | — | **Strong** |
| `queue/processor.py` | Full | Partial | Partial | — | **Strong** |
| `queue/dispatcher.py` | Full | Good | — | — | **Strong** |
| `controller/controller.py` | Full | — | Indirect | — | **Strong** |
| `cluster/deployment.py` | Full | — | — | — | **Strong** |
| `api/config.py` | Full | — | — | — | **Strong** |
| `schema/request.py` | Good | — | — | — | **Good** |
| `api/dependencies.py` | Good | — | — | — | **Good** |
| `schema/response.py` | Partial | — | — | — | **Moderate** |
| `api/app.py` | — | Partial | — | — | **Moderate** |
| `nn/security/*` | — | — | — | Full | **Good** (integration only) |
| `schema/mixins.py` | Full | — | — | — | **Strong** |
| `modeling/base.py` | — | — | Interface | — | **Weak** |
| `cluster/evaluator.py` | — | — | — | — | **None** |
| `providers/*` | — | — | — | — | **None** |
| `api/db.py` | — | — | — | — | **None** |
| `queue/util.py` | — | — | — | — | **None** |
| `metrics/*` | — | — | — | — | **None** |
| `logging/*` | — | — | — | — | **None** |

---

## Recommended Next Steps

**If you have time for 3 things:**
1. **BaseModelDeployment** — core inference execution pipeline, requires heavy torch mocking
2. **Provider classes** — connection management for Redis, Ray, ObjectStore, Sio, Mailgun
3. **AccountsDB** — authentication/authorization with mocked psycopg2

**If you have time for 1 thing:**
1. **BaseModelDeployment** — it's the core inference execution pipeline and currently has only interface-level contract coverage
