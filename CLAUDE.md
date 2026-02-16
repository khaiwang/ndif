# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. Please update this file as you make changes to the codebase with the rules you realized.

## Workflow

Always explain your plan before writing code. Do not start implementing until the plan is approved. If the task involves multiple files or steps, present a numbered plan first.

## Project Overview

NDIF (National Deep Inference Fabric) is a distributed inference server that lets researchers run transparent experiments on large AI models via [nnsight](https://github.com/ndif-team/nnsight). Users submit serialized intervention code that executes in a security sandbox on shared GPU infrastructure.

This is a Python project. Primary language is Python. Tests use pytest. Always run the full test suite (`pytest`) after making test changes to verify no regressions or hanging tests.

## Build & Run Commands

```bash
# Build Docker images (api + ray)
make build

# Start all services (Redis, MinIO, Ray, API, Prometheus, Grafana)
make up

# Stop all services
make down

# Full rebuild cycle: down → build → up
make ta

# Install for local development
pip install -e .            # installs the `ndif` CLI
pip install -e ".[dev]"     # includes ruff, pytest, httpx (dev group via pyproject.toml)
```

## Testing

```bash
# Run local-only tests (no server needed)
pytest tests/

# Run all tests including remote (requires running NDIF server)
pytest --run-remote tests/

# Run against a specific host
pytest --run-remote --ndif-host http://server:port tests/

# Run a single test file
pytest tests/test_security_guards.py

# Run a single test class
pytest --run-remote tests/test_nnsight.py::TestBasicTracing
```

Tests requiring a remote server are auto-skipped unless `--run-remote` is passed. The host defaults to `http://localhost:5001` and can also be set via `NDIF_HOST` env var.

When adding tests, keep test files at a manageable size. If a test file exceeds ~300 lines or ~20 test methods, proactively suggest splitting it into logical sub-files before committing.

## Linting

```bash
ruff check .      # lint
ruff format .     # format
```

## CLI Commands

The `ndif` CLI (installed via `pip install -e .`) manages services:

```bash
ndif start [service]    # start services
ndif stop               # stop services
ndif status             # cluster status
ndif deploy <model>     # deploy a model
ndif evict <model>      # evict a model
ndif logs <service>     # view logs
ndif queue              # show pending requests
```

## Architecture

### Three-service design

1. **API Service** (`src/services/api/`) — FastAPI + Gunicorn. Accepts requests at `POST /request`, validates API keys, pushes to a Redis queue. A `Dispatcher` async loop reads the queue and routes requests to per-model `Processor` instances.

2. **Ray Service** (`src/services/ray/`) — Distributed compute layer.
   - `ControllerActor` (head node): manages cluster state, GPU scheduling, model deployment/eviction, warm caching.
   - `ModelActor` (worker nodes): loads models, deserializes user requests, executes intervention code in a security sandbox, uploads results to MinIO.

3. **CLI** (`cli/`) — Click-based tool for managing services and deployments.

### Request flow

Client → `POST /request` → Redis queue → Dispatcher → per-model Processor → ControllerActor (deploy if needed) → ModelActor (execute in sandbox) → results to MinIO → presigned URL returned via Socket.IO or polling.

### Shared code

- `src/common/schema/` — Pydantic request/response/result models
- `src/common/types.py` — Type aliases (`MODEL_KEY`, `API_KEY`, `REPLICA_ID`, etc.)
- `src/common/providers/` — Static provider classes for external services (Redis, MinIO, Socket.IO, PostgreSQL, Ray)
- `src/common/metrics/` — Prometheus metrics collection

### Key patterns

- **Deployment states**: `COLD` (disk) → `HOT` (GPU, serving) → `WARM` (CPU cache). Warm caching controlled by `NDIF_MODEL_CACHE_PERCENTAGE`.
- **Processor state machine**: `UNINITIALIZED → PROVISIONING → DEPLOYING → READY ↔ BUSY → CANCELLED`
- **Model evaluation**: Loads on meta device (no GPU) to measure size, then bin-packs across GPUs.
- **Async/thread hybrid**: Dispatcher is async for I/O; ModelActor runs execution in a thread pool for timeout/cancellation support.

### Security sandbox

User code runs inside a `Protector` context manager (`src/services/ray/src/ray/nn/security/`):
- **Import whitelist** (`whitelist.yaml`): only torch, numpy, collections, math, etc.
- **Builtin restrictions**: `SafeBuiltins` blocks open, exec, compile, raw `__import__`
- **Dunder guards**: blocks `__class__`, `__globals__`, `__dict__`, `__reduce__`, etc.
- **Protected objects**: model/tokenizer wrapped to prevent `.to()`, weight modification

## Configuration

All config via environment variables. See `.env.example` for defaults. Key variables:
- `NDIF_DEV_MODE=true` — skip API key validation (local dev)
- `NDIF_DEPLOYMENTS` — pipe-separated model keys to deploy at startup
- `NDIF_EXECUTION_TIMEOUT_SECONDS` — max execution time (default 3600)
- `NDIF_BROKER_URL` — Redis URL (default `redis://localhost:6379`)
- `NDIF_API_PORT` — API port (default 5001)

## Pinned Dependencies

- `fastapi==0.108.0` and `python-socketio==5.13.0` are pinned for Socket.IO compatibility — do not upgrade without testing Socket.IO.
- Python >=3.12, <3.14

## Git

When working with git worktrees, always confirm the current worktree/branch before making changes. Run `git worktree list` and verify you are in the correct directory before any edits.

## General Rules

When creating or editing files, double-check you are modifying the correct file. Do not edit existing CLAUDE.md or config files meant for other purposes — create new files when instructed to do so.

## Communication Style

Be concise and precise in responses. Avoid verbose explanations unless the user asks for detail. Prefer actionable, specific output over general commentary.
