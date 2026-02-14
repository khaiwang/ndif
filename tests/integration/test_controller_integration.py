"""Integration tests for ControllerActor deploy/evict via Docker Compose.

Tests run sequentially since deploy/evict is stateful. Each test class operates
on a single model (parametrized via the ``test_model`` fixture).

Prerequisites:
    - Docker images pre-built (``make build``)
    - Run with: ``pytest tests/integration/ --run-remote -v``
    - GPU required (auto-detected, skipped if unavailable)
"""

from typing import Any


import pytest
import ray


DEPLOY_TIMEOUT = 120  # seconds


# ============================================================================
# Infrastructure Tests (no model needed)
# ============================================================================


class TestControllerInfrastructure:
    """Basic connectivity tests that don't require model deployment."""

    def test_controller_exists(self, controller):
        """Can get Controller actor handle from Ray."""
        assert controller is not None

    def test_get_state_returns_valid_dict(self, controller):
        """get_state() returns dict with expected keys."""
        state = ray.get(controller.get_state.remote())
        assert isinstance(state, dict)
        assert "cluster" in state
        assert "execution_timeout_seconds" in state


# ============================================================================
# Deploy / Evict Tests (per model)
# ============================================================================


class TestDeployEvict:
    """Ordered deploy/evict tests parametrized over enabled models."""

    def test_deploy_model(self, controller, test_model):
        """deploy() returns result dict, status is not CANT_ACCOMMODATE."""
        result = ray.get(
            controller.deploy.remote([test_model.model_key]),
            timeout=DEPLOY_TIMEOUT,
        )
        assert isinstance(result, dict)
        assert "result" in result
        for (_model_key, _replica_id), status in result["result"].items():
            assert str(status).upper() != "CANT_ACCOMMODATE", (
                f"Model {test_model.model_key} cannot be accommodated"
            )

    def test_deploy_result_format(self, controller, test_model):
        """Deploy result has 'result' and 'evictions' keys."""
        result = ray.get(
            controller.deploy.remote([test_model.model_key]),
            timeout=DEPLOY_TIMEOUT,
        )
        assert "result" in result
        assert "evictions" in result

    def test_deployed_model_in_state(self, controller, test_model):
        """After deploy, get_state() shows model in cluster."""
        # Ensure deployed
        ray.get(
            controller.deploy.remote([test_model.model_key]),
            timeout=DEPLOY_TIMEOUT,
        )
        state = ray.get(controller.get_state.remote())
        cluster_state = state["cluster"]
        # Model should appear somewhere in the cluster nodes
        found = False
        for node_state in cluster_state.get("nodes", []):
            deployments = node_state.get("deployments", {})
            for model_key in deployments:
                if test_model.model_key in str(model_key):
                    found = True
                    break
        assert found, (
            f"Model {test_model.model_key} not found in cluster state after deploy"
        )

    def test_deploy_already_deployed(self, controller, test_model):
        """Re-deploying returns DEPLOYED status (idempotent)."""
        # First deploy
        ray.get(
            controller.deploy.remote([test_model.model_key]),
            timeout=DEPLOY_TIMEOUT,
        )
        # Second deploy
        result = ray.get(
            controller.deploy.remote([test_model.model_key]),
            timeout=DEPLOY_TIMEOUT,
        )
        statuses: list[Any] = list(result["result"].values())
        assert any("DEPLOYED" in str(s).upper() for s in statuses), (
            f"Expected DEPLOYED status on re-deploy, got: {statuses}"
        )

    def test_evict_model(self, controller, test_model):
        """evict() succeeds without error."""
        # Ensure deployed first
        ray.get(
            controller.deploy.remote([test_model.model_key]),
            timeout=DEPLOY_TIMEOUT,
        )
        # Evict
        result = ray.get(controller.evict.remote([test_model.model_key]))
        assert isinstance(result, dict)

    def test_evicted_model_removed(self, controller, test_model):
        """After evict, model no longer in HOT deployments."""
        # Deploy then evict
        ray.get(
            controller.deploy.remote([test_model.model_key]),
            timeout=DEPLOY_TIMEOUT,
        )
        ray.get(controller.evict.remote([test_model.model_key]))

        state = ray.get(controller.get_state.remote())
        cluster_state = state["cluster"]
        for node_state in cluster_state.get("nodes", []):
            deployments = node_state.get("deployments", {})
            for model_key, model_map in deployments.items():
                if test_model.model_key in str(model_key):
                    # If the model is still here, it should be cached (WARM) not HOT
                    for dep_info in model_map.values():
                        if isinstance(dep_info, dict):
                            assert dep_info.get("deployment_level") != "hot", (
                                f"Model {test_model.model_key} still HOT after evict"
                            )

    def test_evict_nonexistent_model(self, controller):
        """Evicting unknown model returns not_found status."""
        result = ray.get(controller.evict.remote(["nonexistent/model-xyz"]))
        assert isinstance(result, dict)
        found_not_found = any(
            "not_found" in str(v) for v in result.values()
        )
        assert found_not_found, f"Expected not_found status, got: {result}"

    def test_deploy_evict_cycle(self, controller, test_model):
        """Deploy -> evict -> deploy cycle works cleanly."""
        # Deploy
        r1 = ray.get(
            controller.deploy.remote([test_model.model_key]),
            timeout=DEPLOY_TIMEOUT,
        )
        assert "result" in r1

        # Evict
        ray.get(controller.evict.remote([test_model.model_key]))

        # Re-deploy
        r2 = ray.get(
            controller.deploy.remote([test_model.model_key]),
            timeout=DEPLOY_TIMEOUT,
        )
        assert "result" in r2
        for (_model_key, _replica_id), status in r2["result"].items():
            assert str(status).upper() != "CANT_ACCOMMODATE"
