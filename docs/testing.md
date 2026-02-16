# NDIF Test Framework

## Overview

NDIF uses a layered testing strategy that allows thorough verification of individual components without requiring the full distributed stack (Ray, Redis, MinIO, PostgreSQL) to be running. The framework consists of four test layers, three of which run entirely offline.

**575 tests run in ~2.4 seconds with zero external dependencies.**

| Layer | Tests | Directory | What it verifies |
|-------|------:|-----------|------------------|
| Unit | 540 | `tests/unit/` | Pure logic of individual classes and functions |
| Component | 19 | `tests/component/` | Service-internal wiring with mocked boundaries |
| Contract | 16 | `tests/contract/` | Message format compatibility across API and Ray |
| Integration | — | `tests/` (root) | End-to-end against a running NDIF stack |

## Running Tests

```bash
# All offline tests (unit + component + contract)
pytest -m "not integration"

# Individual layers
pytest tests/unit/
pytest tests/component/
pytest tests/contract/

# By marker
pytest -m unit
pytest -m component
pytest -m contract

# Integration tests (requires running NDIF stack)
pytest --run-remote

# Single test file or class
pytest tests/unit/test_node.py
pytest tests/unit/test_processor.py::TestEnqueue

# Verbose output
pytest tests/unit/ -v
```

Markers are auto-applied based on directory — no need to decorate individual tests.

## Test Layers

### Unit Tests (540 tests)

Test individual classes with all external dependencies mocked. No Redis, Ray, or network calls.

| File | Tests | Source under test |
|------|------:|-------------------|
| `test_controller.py` | 82 | `_ControllerActor` — deploy/evict async wrappers, check_nodes, get_state, replica adjustment |
| `test_node.py` | 80 | `Node`, `Resources`, `Candidate`, `CandidateLevel` — GPU allocation, eviction planning, deployment evaluation |
| `test_processor.py` | 52 | `Processor` state machine — enqueue, provision, replica management, kill, purge |
| `test_replicas.py` | 47 | `Replica` and `Replicas` — add/remove, set_replica_ids, begin/end request, worker tasks |
| `test_dispatcher.py` | 47 | `Dispatcher` — dispatch routing, eviction/error handling, Redis stream event handlers |
| `test_cluster.py` | 37 | `Cluster` — multi-node deploy, evict, target replica ID generation |
| `test_processor_workers.py` | 35 | `Processor` worker loops — processor_worker, reply_worker, replica_worker, execute_on_replica |
| `test_request_parsing.py` | 34 | `BackendRequestModel.from_request()`, `validate_request()` — deserialization, validation pipeline |
| `test_config.py` | 33 | `AppConfig`, `QueueConfig` — from_env() parsing, validation, defaults |
| `test_deployment.py` | 29 | `Deployment` actor lifecycle — create, delete, restart, cache, from_cache, get_state |
| `test_dispatcher_workers.py` | 28 | `Dispatcher` worker loops — dispatch_worker, status_worker, events_worker |
| `test_dependencies.py` | 20 | API dependency validators — auth, Python/nnsight version checks, Ray connection |
| `test_schema.py` | 16 | `BackendResponseModel`, `BackendRequestModel` — respond(), create_response(), is_email() |

### Component Tests (19 tests)

Test service-internal integration with mocked external boundaries.

| File | Tests | What it covers |
|------|------:|----------------|
| `test_dispatch_flow.py` | 14 | Dispatcher + Processor coordination: request routing, eviction events, error recovery, Redis stream events (deploy, evict, kill, queue state) |
| `test_api.py` | 5 | FastAPI endpoints via httpx ASGITransport: `/ping`, `/connected`, `/response/{id}` |

### Contract Tests (16 tests)

Verify that the data formats exchanged between the API and Ray services remain compatible. These tests import from **both** sides of the boundary and check that producer output matches consumer expectations.

| File | Tests | What it verifies |
|------|------:|------------------|
| `test_controller_contract.py` | 9 | `Cluster.deploy()` / `evict()` return format, `Processor.check_dedicated()` parsing, `DeploymentStatus` ↔ `CandidateLevel` enum value alignment |
| `test_model_actor_contract.py` | 7 | `ModelActor` interface methods (`__call__`, `cancel`, `__ray_ready__`, `to_cache`, `from_cache`), replica actor naming conventions |

### Integration Tests

Pre-existing tests in `tests/` root that run against a live NDIF stack. Gated behind `--run-remote` and skipped by default.

## Architecture

### Mocking Infrastructure

All module-level mocking lives in `tests/conftest.py` and is shared by every test layer. This ensures any test file can `from src.services.* import ...` without triggering real connections.

**What gets mocked:**

| Dependency | Why | How |
|-----------|-----|-----|
| Redis | Prevent connections | `patch("redis.Redis.from_url")`, `patch("redis.asyncio.Redis.from_url")` |
| Ray | Not installed in test env | `patch.dict("sys.modules", {"ray": ..., "ray.util": ..., ...})` |
| nnsight | Circular import with accelerate | `patch.dict("sys.modules", ...)` with real pydantic BaseModel stubs for `ResponseModel`/`RequestModel` |
| transformers | Version mismatch | `patch.dict("sys.modules", ...)` |
| psycopg2 | Not installed | `sys.modules.setdefault(...)` |
| boto3 | Prevent S3 connections | `patch("boto3.client")` |

**Why nnsight stubs are real pydantic models:**
`BackendResponseModel` inherits from `(ResponseModel, ObjectStorageMixin, TelemetryMixin)`. If `ResponseModel` were a `MagicMock`, the metaclass conflict between `MagicMock` and pydantic's `ModelMetaclass` would crash at class definition time. The stubs in conftest are minimal pydantic `BaseModel` subclasses with the fields and methods that the NDIF schema classes rely on.

### Conftest Hierarchy

```
tests/
  conftest.py          # Module mocking, env vars, pytest hooks, markers
  unit/
    conftest.py        # Factory fixtures: make_resources, make_node, make_deployment,
                       #   make_request, make_processor, mock_redis_*
  component/
    conftest.py        # mock_redis_async, mock_redis_sync (fresh per test)
  contract/
    conftest.py        # make_resources, make_node (for Cluster tests)
```

Sub-directory conftest files contain **only fixtures** — no mocking. All mocking is centralized in the root conftest.

### Factory Fixtures

Unit and contract tests use factory fixtures to build domain objects with sensible defaults:

```python
# Create a node with 4x A100 GPUs (80 GiB each)
node = make_node(node_id="n1", name="gpu-node", total_gpus=4)

# Create a node with custom resources
node = make_node(total_gpus=2, gpu_memory_bytes=40 * 1024**3)

# Create a deployment
dep = make_deployment(model_key="llama-7b", replica_id="r1", size_bytes=10 * 1024**3)

# Create a mock request
req = make_request(model_key="meta-llama/Llama-2-7b", hotswapping=True)

# Create a Processor with real asyncio queues
processor = make_processor(model_key="meta-llama/Llama-2-7b", replica_count=2)
```

## Adding New Tests

### Adding a unit test

1. Create or edit a file in `tests/unit/test_<module>.py`
2. Import the source module inside the test (or at file top) — the root conftest mocking ensures imports work
3. Use `unittest.mock.patch` for any provider calls your code path hits
4. Use factory fixtures from `tests/unit/conftest.py` for domain objects

```python
class TestMyFeature:
    def test_something(self, make_node):
        node = make_node(total_gpus=2)
        # ... test logic
        assert node.resources.total_gpus == 2
```

### Adding a component test

1. Create or edit a file in `tests/component/test_<feature>.py`
2. Mock provider classes at the module level where the source code imports them
3. For API endpoint tests, use `httpx.AsyncClient` with `ASGITransport(app=app)`

### Adding a contract test

1. Create or edit a file in `tests/contract/test_<boundary>.py`
2. Import from **both** sides of the boundary (API and Ray)
3. Verify that the producer's output format matches what the consumer parses

### Adding a new module mock

If a new external dependency is introduced:

1. Add the mock to `tests/conftest.py` in the appropriate section
2. For packages not installed in the test env: use `patch.dict("sys.modules", ...)`
3. For packages that are installed but should not connect: use `patch("module.Class.method")`
4. If the mock needs to be a real class (e.g., for metaclass compatibility), define a stub class

## pytest Configuration

Defined in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "unit: Unit tests (no external services)",
    "component: Component tests (with fakes/mocks for external services)",
    "integration: Integration tests (requires running NDIF stack)",
    "contract: Contract tests (verify inter-service message formats)",
]
```

Dev dependencies:

```toml
[dependency-groups]
dev = [
    "pytest>=8.4.0",
    "pytest-asyncio>=1.3.0",
    "pytest-mock>=3.12",
    "httpx>=0.28.1",
    "fakeredis[lua]>=2.0",
    "ruff>=0.14.5",
]
```

## Known Caveats

- **Pydantic v2 instance patching**: You cannot `patch.object(pydantic_instance, "method")`. Pydantic v2 restricts `__setattr__`/`__delattr__`. Patch at the class level instead: `patch.object(MyModel, "method")`.
- **ASGITransport exception handling**: `httpx.AsyncClient` with `ASGITransport` does not convert unhandled exceptions to 500 responses — they propagate as Python exceptions. Test accordingly with `pytest.raises`.
- **Coroutine warnings**: Some component tests produce `RuntimeWarning: coroutine was never awaited` for mocked `Processor.processor_worker`. These are harmless — the worker coroutine is created but never started since the event loop isn't running in sync tests.
