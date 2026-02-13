"""Integration test fixtures — Docker Compose lifecycle + Ray connection.

Architecture:
    pytest session start
      └─ session fixture: build compose cmd (base + override if exists)
      └─ load env from .env.example + docker/.env
      └─ detect HOST_IP and N_DEVICES dynamically
      └─ docker compose -p ndif-test up -d (broker, minio, ray, api)
      └─ poll /ping and /connected until ready (timeout 120s)
      └─ connect to Ray via ray.init()
      └─ run tests
      └─ session fixture teardown: docker compose -p ndif-test down -v

Prerequisites:
    - Docker images must be pre-built (``make build``)
    - Default ports must be free (stop any running dev stack first)
    - GPU required (N_DEVICES >= 1); tests skip if no GPU
"""

import os
import subprocess
import time

import pytest

from .models import get_enabled_models

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COMPOSE_BASE = os.path.join(PROJECT_ROOT, "docker", "docker-compose.yml")
COMPOSE_OVERRIDE = os.path.join(PROJECT_ROOT, "docker", "docker-compose.override.yml")
PROJECT_NAME = "ndif-test"
CORE_SERVICES = ["message_broker", "minio", "ray", "api"]


def _build_compose_files():
    """Detect compose files the same way GNUmakefile does."""
    files = ["-f", COMPOSE_BASE]
    if os.path.exists(COMPOSE_OVERRIDE):
        files += ["-f", COMPOSE_OVERRIDE]
    return files


def _load_env():
    """Load env vars from .env.example and docker/.env, same as Makefile."""
    env = {}
    for envfile in [
        os.path.join(PROJECT_ROOT, ".env.example"),
        os.path.join(PROJECT_ROOT, "docker", ".env"),
    ]:
        if os.path.exists(envfile):
            with open(envfile) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        env[k.strip()] = v.strip().strip("'\"")

    # Dynamic detection (same as Makefile)
    try:
        env["HOST_IP"] = subprocess.check_output(
            ["hostname", "-I"], text=True
        ).split()[0]
    except (subprocess.CalledProcessError, IndexError):
        env["HOST_IP"] = "127.0.0.1"

    env["N_DEVICES"] = subprocess.check_output(
        "command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L | wc -l || echo 0",
        shell=True,
        text=True,
    ).strip()

    # Override: no auto-deploy, tests control deployment
    env["NDIF_DEPLOYMENTS"] = ""
    return env


def _wait_for_ready(api_base, timeout=120):
    """Poll /ping then /connected until both succeed."""
    import requests

    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{api_base}/ping", timeout=2)
            if r.status_code == 200:
                r2 = requests.get(f"{api_base}/connected", timeout=2)
                if r2.status_code == 200:
                    return
        except requests.ConnectionError as e:
            last_error = e
        time.sleep(2)
    raise TimeoutError(
        f"Services not ready after {timeout}s. Last error: {last_error}"
    )


@pytest.fixture(scope="session")
def docker_services(request):
    """Start Docker Compose stack, wait for readiness, tear down after.

    Skips if:
    - --run-remote not passed
    - N_DEVICES == 0 (no GPU)
    """
    if not request.config.getoption("--run-remote", default=False):
        pytest.skip("Need --run-remote option to run integration tests")

    env_vars = _load_env()
    if env_vars.get("N_DEVICES", "0") == "0":
        pytest.skip("No GPU available (N_DEVICES=0)")

    merged_env = {**os.environ, **env_vars}
    compose_files = _build_compose_files()
    compose_cmd = ["docker", "compose", "-p", PROJECT_NAME] + compose_files

    # Start only core services (skip telemetry)
    subprocess.run(
        compose_cmd + ["up", "-d"] + CORE_SERVICES,
        env=merged_env,
        check=True,
        cwd=PROJECT_ROOT,
    )

    api_port = env_vars.get("NDIF_API_PORT", "5001")
    try:
        _wait_for_ready(f"http://localhost:{api_port}", timeout=120)
    except TimeoutError:
        # Dump logs on failure before tearing down
        subprocess.run(
            compose_cmd + ["logs", "--tail=50"],
            env=merged_env,
            cwd=PROJECT_ROOT,
        )
        subprocess.run(
            compose_cmd + ["down", "-v"],
            env=merged_env,
            cwd=PROJECT_ROOT,
        )
        raise

    yield env_vars

    # Tear down with volumes
    subprocess.run(
        compose_cmd + ["down", "-v"],
        env=merged_env,
        cwd=PROJECT_ROOT,
    )


@pytest.fixture(scope="session")
def controller(docker_services):
    """Connect to Ray and get Controller handle."""
    import ray

    env = docker_services
    ray_port = env.get("NDIF_RAY_CLIENT_PORT", "10001")
    ray.init(address=f"ray://localhost:{ray_port}", ignore_reinit_error=True)
    handle = ray.get_actor("Controller", namespace="NDIF")
    yield handle
    ray.shutdown()


@pytest.fixture(params=get_enabled_models(), ids=lambda m: m.name)
def test_model(request):
    """Parametrize tests over enabled models."""
    return request.param
