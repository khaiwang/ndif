"""Comprehensive unit tests for the _ControllerActor class.

Tests cover:
- _deploy: sets desired_replicas, delegates to cluster.deploy, calls apply on change
- deploy: async wrapper around _deploy
- evict: with/without replica_keys, desired_replicas tracking, apply on change
- _adjust_desired_for_cant_accommodate: decrements desired for CANT_ACCOMMODATE results
- _current_replica_count: counts replicas across nodes
- build: computes DeploymentDelta from state vs cluster nodes
- apply: executes delta (delete, cache, from_cache, create) with error cleanup
- _remove_deployment_from_state: cleans state and node resources
- _monitor_deployment: async monitoring with cleanup on failure
- scale / scale_up: replica scaling with idempotency
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from src.services.ray.src.ray.deployments.controller.cluster.deployment import (
    Deployment,
    DeploymentLevel,
)

_MOD = "src.services.ray.src.ray.deployments.controller.controller"


# ============================================================================
# Helpers
# ============================================================================

GIB = 1024**3


def _make_deployment(
    model_key="model-a",
    replica_id="r1",
    deployment_level=DeploymentLevel.HOT,
    gpu_mem_bytes_by_id=None,
    gpu_memory_fraction=0.5,
    size_bytes=10 * GIB,
    dedicated=False,
    node_id="node-1",
):
    if gpu_mem_bytes_by_id is None:
        gpu_mem_bytes_by_id = {0: 80 * GIB}
    return Deployment(
        model_key=model_key,
        replica_id=replica_id,
        deployment_level=deployment_level,
        gpu_mem_bytes_by_id=gpu_mem_bytes_by_id,
        gpu_memory_fraction=gpu_memory_fraction,
        size_bytes=size_bytes,
        dedicated=dedicated,
        node_id=node_id,
    )


@pytest.fixture
def controller():
    """Instantiate _ControllerActor with all external deps mocked."""
    with patch(f"{_MOD}.set_logger", return_value=MagicMock()):
        with patch("asyncio.create_task"):
            with patch(f"{_MOD}.Cluster") as MockCluster:
                mock_cluster = MagicMock()
                mock_cluster.nodes = {}
                MockCluster.return_value = mock_cluster
                with patch(f"{_MOD}.ray") as mock_ray:
                    mock_ray.get_runtime_context.return_value = MagicMock()
                    from src.services.ray.src.ray.deployments.controller.controller import (
                        _ControllerActor,
                    )

                    actor = _ControllerActor(
                        deployments=[],
                        model_import_path="test:app",
                        execution_timeout_seconds=3600.0,
                        model_cache_percentage=0.5,
                        minimum_deployment_time_seconds=3600.0,
                    )
    # Replace cluster with a fresh mock for test isolation
    actor.cluster = MagicMock()
    actor.cluster.nodes = {}
    return actor


# ============================================================================
# TestControllerDeploy
# ============================================================================


class TestControllerDeploy:
    """Tests for _deploy() and async deploy()."""

    def test_sets_desired_replicas(self, controller):
        """Verify _deploy sets desired_replicas for each model key."""
        controller.cluster.deploy.return_value = ({"result": {}}, False)
        controller._deploy(["model-a", "model-b"], replicas=2)
        assert controller.desired_replicas["model-a"] == 2
        assert controller.desired_replicas["model-b"] == 2

    def test_delegates_to_cluster_deploy(self, controller):
        """Verify _deploy forwards model keys, dedicated, and replicas to cluster.deploy."""
        controller.cluster.deploy.return_value = ({"result": {}}, False)
        controller._deploy(["model-a"], dedicated=True, replicas=3)
        controller.cluster.deploy.assert_called_once_with(
            ["model-a"], dedicated=True, replicas=3
        )

    def test_calls_adjust_desired(self, controller):
        """Verify _deploy decrements desired_replicas for CANT_ACCOMMODATE results."""
        results = {"result": {("model-a", "r1"): "CANT_ACCOMMODATE"}}
        controller.cluster.deploy.return_value = (results, False)
        controller._deploy(["model-a"], replicas=2)
        # desired should be decremented from 2 to 1
        assert controller.desired_replicas["model-a"] == 1

    def test_calls_apply_on_change(self, controller):
        """Verify _deploy calls apply() when cluster reports a change."""
        controller.cluster.deploy.return_value = ({"result": {}}, True)
        with patch.object(controller, "apply") as mock_apply:
            controller._deploy(["model-a"])
            mock_apply.assert_called_once()

    def test_does_not_call_apply_when_no_change(self, controller):
        """Verify _deploy skips apply() when cluster reports no change."""
        controller.cluster.deploy.return_value = ({"result": {}}, False)
        with patch.object(controller, "apply") as mock_apply:
            controller._deploy(["model-a"])
            mock_apply.assert_not_called()

    def test_returns_cluster_results(self, controller):
        """Verify _deploy returns the results dict from cluster.deploy."""
        expected = {"result": {("model-a", "r1"): "FREE"}}
        controller.cluster.deploy.return_value = (expected, False)
        result = controller._deploy(["model-a"])
        assert result == expected

    def test_default_replicas_is_one(self, controller):
        """Verify _deploy defaults to replicas=1 when not specified."""
        controller.cluster.deploy.return_value = ({"result": {}}, False)
        controller._deploy(["model-a"])
        assert controller.desired_replicas["model-a"] == 1

    def test_default_dedicated_is_false(self, controller):
        """Verify _deploy defaults to dedicated=False when not specified."""
        controller.cluster.deploy.return_value = ({"result": {}}, False)
        controller._deploy(["model-a"])
        controller.cluster.deploy.assert_called_once_with(
            ["model-a"], dedicated=False, replicas=1
        )

    @pytest.mark.asyncio
    async def test_async_deploy_wraps_sync(self, controller):
        """Verify async deploy() delegates to the synchronous _deploy logic."""
        controller.cluster.deploy.return_value = ({"result": {}}, False)
        with patch.object(controller, "apply"):
            result = await controller.deploy(["model-a"], replicas=2, dedicated=True)
        controller.cluster.deploy.assert_called_once_with(
            ["model-a"], dedicated=True, replicas=2
        )


# ============================================================================
# TestControllerEvict
# ============================================================================


class TestControllerEvict:
    """Tests for evict()."""

    def test_without_replica_keys_sets_desired_zero(self, controller):
        """Verify evict without replica_keys sets desired_replicas to zero."""
        controller.cluster.evict.return_value = ({}, False)
        controller.evict(["model-a", "model-b"])
        assert controller.desired_replicas["model-a"] == 0
        assert controller.desired_replicas["model-b"] == 0

    def test_with_replica_keys_decrements_desired(self, controller):
        """Verify evict with replica_keys decrements desired_replicas by the count evicted."""
        controller.desired_replicas["model-a"] = 3
        controller.cluster.evict.return_value = ({}, False)
        controller.evict(
            model_keys=["model-a"],
            replica_keys=[("model-a", "r1")],
        )
        assert controller.desired_replicas["model-a"] == 2

    def test_with_replica_keys_clamps_to_zero(self, controller):
        """Verify evict with replica_keys does not decrement desired_replicas below zero."""
        controller.desired_replicas["model-a"] = 0
        controller.cluster.evict.return_value = ({}, False)
        controller.evict(
            model_keys=["model-a"],
            replica_keys=[("model-a", "r1")],
        )
        assert controller.desired_replicas["model-a"] == 0

    def test_with_replica_keys_falls_back_to_current_count(self, controller):
        """When model not in desired_replicas, uses _current_replica_count."""
        mock_node = MagicMock()
        mock_node.deployments = {"model-a": {"r1": MagicMock(), "r2": MagicMock()}}
        controller.cluster.nodes = {"n1": mock_node}
        controller.cluster.evict.return_value = ({}, False)

        controller.evict(
            model_keys=["model-a"],
            replica_keys=[("model-a", "r1")],
        )
        # _current_replica_count returns 2, evicting 1 → 1
        assert controller.desired_replicas["model-a"] == 1

    def test_delegates_to_cluster_evict(self, controller):
        """Verify evict forwards model_keys and replica_keys to cluster.evict."""
        controller.cluster.evict.return_value = ({}, False)
        replica_keys = [("model-a", "r1")]
        controller.evict(["model-a"], replica_keys=replica_keys)
        controller.cluster.evict.assert_called_once_with(
            ["model-a"], replica_keys=replica_keys
        )

    def test_calls_apply_on_change(self, controller):
        """Verify evict calls apply() when cluster reports a change."""
        controller.cluster.evict.return_value = ({}, True)
        with patch.object(controller, "apply") as mock_apply:
            controller.evict(["model-a"])
            mock_apply.assert_called_once()

    def test_does_not_call_apply_when_no_change(self, controller):
        """Verify evict skips apply() when cluster reports no change."""
        controller.cluster.evict.return_value = ({}, False)
        with patch.object(controller, "apply") as mock_apply:
            controller.evict(["model-a"])
            mock_apply.assert_not_called()

    def test_returns_results(self, controller):
        """Verify evict returns the results dict from cluster.evict."""
        expected = {("model-a", "r1"): {"status": "evicted"}}
        controller.cluster.evict.return_value = (expected, True)
        with patch.object(controller, "apply"):
            result = controller.evict(["model-a"])
        assert result == expected

    def test_multiple_replica_keys_same_model(self, controller):
        """Verify evicting multiple replicas of one model decrements desired by that count."""
        controller.desired_replicas["model-a"] = 3
        controller.cluster.evict.return_value = ({}, False)
        controller.evict(
            model_keys=["model-a"],
            replica_keys=[("model-a", "r1"), ("model-a", "r2")],
        )
        assert controller.desired_replicas["model-a"] == 1


# ============================================================================
# TestAdjustDesired
# ============================================================================


class TestAdjustDesired:
    """Tests for _adjust_desired_for_cant_accommodate()."""

    def test_decrements_cant_accommodate(self, controller):
        """Verify desired_replicas is decremented once per CANT_ACCOMMODATE result."""
        controller.desired_replicas["model-a"] = 3
        results = {
            "result": {
                ("model-a", "r1"): "CANT_ACCOMMODATE",
                ("model-a", "r2"): "FREE",
            }
        }
        controller._adjust_desired_for_cant_accommodate(results)
        assert controller.desired_replicas["model-a"] == 2

    def test_multiple_cant_accommodate(self, controller):
        """Verify multiple CANT_ACCOMMODATE results decrement desired_replicas accordingly."""
        controller.desired_replicas["model-a"] = 3
        results = {
            "result": {
                ("model-a", "r1"): "CANT_ACCOMMODATE",
                ("model-a", "r2"): "CANT_ACCOMMODATE",
            }
        }
        controller._adjust_desired_for_cant_accommodate(results)
        assert controller.desired_replicas["model-a"] == 1

    def test_clamps_to_zero(self, controller):
        """Verify desired_replicas does not go below zero after CANT_ACCOMMODATE adjustments."""
        controller.desired_replicas["model-a"] = 1
        results = {
            "result": {
                ("model-a", "r1"): "CANT_ACCOMMODATE",
                ("model-a", "r2"): "CANT_ACCOMMODATE",
            }
        }
        controller._adjust_desired_for_cant_accommodate(results)
        assert controller.desired_replicas["model-a"] == 0

    def test_no_cant_accommodate_no_change(self, controller):
        """Verify desired_replicas is unchanged when no CANT_ACCOMMODATE results exist."""
        controller.desired_replicas["model-a"] = 3
        results = {"result": {("model-a", "r1"): "FREE"}}
        controller._adjust_desired_for_cant_accommodate(results)
        assert controller.desired_replicas["model-a"] == 3


# ============================================================================
# TestCurrentReplicaCount
# ============================================================================


class TestCurrentReplicaCount:
    """Tests for _current_replica_count()."""

    def test_counts_replicas_across_nodes(self, controller):
        """Verify replica count sums replicas for a model across all cluster nodes."""
        node1 = MagicMock()
        node1.deployments = {"model-a": {"r1": MagicMock()}}
        node2 = MagicMock()
        node2.deployments = {"model-a": {"r2": MagicMock()}}
        controller.cluster.nodes = {"n1": node1, "n2": node2}
        assert controller._current_replica_count("model-a") == 2

    def test_returns_zero_for_unknown_model(self, controller):
        """Verify replica count returns zero when the model has no deployments."""
        node = MagicMock()
        node.deployments = {}
        controller.cluster.nodes = {"n1": node}
        assert controller._current_replica_count("nonexistent") == 0

    def test_empty_cluster(self, controller):
        """Verify replica count returns zero when the cluster has no nodes."""
        controller.cluster.nodes = {}
        assert controller._current_replica_count("model-a") == 0


# ============================================================================
# TestControllerBuild
# ============================================================================


class TestControllerBuild:
    """Tests for build() — computes DeploymentDelta from state vs nodes."""

    def test_new_deployment_creates(self, controller):
        """Deployment in node.deployments but not in state → to_create."""
        dep = _make_deployment(model_key="model-a", replica_id="r1", node_id="n1")
        node = MagicMock()
        node.name = "test-node"
        node.deployments = {"model-a": {"r1": dep}}
        node.cache = {}
        controller.cluster.nodes = {"n1": node}
        controller.state = {}

        delta = controller.build()
        assert len(delta.deployments_to_create) == 1
        assert delta.deployments_to_create[0] == ("test-node", dep)

    def test_missing_from_nodes_deletes(self, controller):
        """Deployment in state but not in any node → to_delete."""
        dep = _make_deployment(model_key="model-a", replica_id="r1", node_id="n1")
        controller.state = {("n1", "model-a", "r1"): dep}
        controller.cluster.nodes = {}

        delta = controller.build()
        assert len(delta.deployments_to_delete) == 1
        assert delta.deployments_to_delete[0] is dep

    def test_hot_in_cache_becomes_to_cache(self, controller):
        """HOT deployment moved to cache → to_cache."""
        dep_hot = _make_deployment(
            model_key="model-a", replica_id="r1",
            deployment_level=DeploymentLevel.HOT, node_id="n1",
        )
        dep_cached = _make_deployment(
            model_key="model-a", replica_id="r1",
            deployment_level=DeploymentLevel.WARM, node_id="n1",
        )
        controller.state = {("n1", "model-a", "r1"): dep_hot}

        node = MagicMock()
        node.cache = {"model-a": {"r1": dep_cached}}
        node.deployments = {}
        controller.cluster.nodes = {"n1": node}

        delta = controller.build()
        assert len(delta.deployments_to_cache) == 1
        assert delta.deployments_to_cache[0] is dep_cached

    def test_warm_in_deployments_becomes_from_cache(self, controller):
        """WARM deployment moved to deployments → from_cache."""
        dep_warm = _make_deployment(
            model_key="model-a", replica_id="r1",
            deployment_level=DeploymentLevel.WARM, node_id="n1",
        )
        dep_hot = _make_deployment(
            model_key="model-a", replica_id="r1",
            deployment_level=DeploymentLevel.HOT, node_id="n1",
        )
        controller.state = {("n1", "model-a", "r1"): dep_warm}

        node = MagicMock()
        node.cache = {}
        node.deployments = {"model-a": {"r1": dep_hot}}
        controller.cluster.nodes = {"n1": node}

        delta = controller.build()
        assert len(delta.deployments_from_cache) == 1
        assert delta.deployments_from_cache[0] is dep_hot

    def test_matching_state_no_deltas(self, controller):
        """HOT→HOT (same level, same location) → no deltas."""
        dep = _make_deployment(
            model_key="model-a", replica_id="r1",
            deployment_level=DeploymentLevel.HOT, node_id="n1",
        )
        controller.state = {("n1", "model-a", "r1"): dep}

        node = MagicMock()
        node.cache = {}
        node.deployments = {"model-a": {"r1": dep}}
        controller.cluster.nodes = {"n1": node}

        delta = controller.build()
        assert delta.deployments_to_create == []
        assert delta.deployments_to_delete == []
        assert delta.deployments_to_cache == []
        assert delta.deployments_from_cache == []

    def test_empty_cluster_empty_state(self, controller):
        """No nodes, no state → all empty."""
        controller.state = {}
        controller.cluster.nodes = {}

        delta = controller.build()
        assert delta.deployments_to_create == []
        assert delta.deployments_to_delete == []
        assert delta.deployments_to_cache == []
        assert delta.deployments_from_cache == []


# ============================================================================
# TestControllerApply
# ============================================================================


class TestControllerApply:
    """Tests for apply() — executes DeploymentDelta actions."""

    def test_calls_build(self, controller):
        """Verify apply() invokes build() to compute the deployment delta."""
        with patch.object(controller, "build") as mock_build:
            from src.services.ray.src.ray.deployments.controller.controller import (
                DeploymentDelta,
            )

            mock_build.return_value = DeploymentDelta([], [], [], [])
            controller.apply()
            mock_build.assert_called_once()

    def test_deletes_deployments(self, controller):
        """Verify apply() calls delete() on each deployment in the to_delete list."""
        from src.services.ray.src.ray.deployments.controller.controller import (
            DeploymentDelta,
        )

        dep = MagicMock()
        with patch.object(controller, "build") as mock_build:
            mock_build.return_value = DeploymentDelta([], [], [], [dep])
            controller.apply()
        dep.delete.assert_called_once()

    def test_caches_deployment_success(self, controller):
        """cache() returns a future → ray.get() is called on it."""
        from src.services.ray.src.ray.deployments.controller.controller import (
            DeploymentDelta,
        )

        dep = MagicMock()
        dep.model_key = "model-a"
        future = MagicMock()
        dep.cache.return_value = future

        with patch.object(controller, "build") as mock_build:
            mock_build.return_value = DeploymentDelta([dep], [], [], [])
            with patch(f"{_MOD}.ray") as mock_ray:
                controller.apply()
                mock_ray.get.assert_called_once_with(future)

    def test_cache_returns_none_cleans_up(self, controller):
        """cache() returns None → deployment is deleted and removed from state."""
        from src.services.ray.src.ray.deployments.controller.controller import (
            DeploymentDelta,
        )

        dep = MagicMock()
        dep.model_key = "model-a"
        dep.cache.return_value = None

        with patch.object(controller, "build") as mock_build:
            mock_build.return_value = DeploymentDelta([dep], [], [], [])
            with patch.object(controller, "_remove_deployment_from_state") as mock_rm:
                controller.apply()
        dep.delete.assert_called_once()
        mock_rm.assert_called_once_with(dep)

    def test_cache_ray_get_failure_cleans_up(self, controller):
        """ray.get(future) raises → deployment is deleted and removed from state."""
        from src.services.ray.src.ray.deployments.controller.controller import (
            DeploymentDelta,
        )

        dep = MagicMock()
        dep.model_key = "model-a"
        future = MagicMock()
        dep.cache.return_value = future

        with patch.object(controller, "build") as mock_build:
            mock_build.return_value = DeploymentDelta([dep], [], [], [])
            with patch(f"{_MOD}.ray") as mock_ray:
                mock_ray.get.side_effect = RuntimeError("cache failed")
                with patch.object(
                    controller, "_remove_deployment_from_state"
                ) as mock_rm:
                    controller.apply()
        dep.delete.assert_called_once()
        mock_rm.assert_called_once_with(dep)

    def test_from_cache_spawns_monitor_task(self, controller):
        """from_cache() returns a future → asyncio.create_task is called."""
        from src.services.ray.src.ray.deployments.controller.controller import (
            DeploymentDelta,
        )

        dep = MagicMock()
        dep.model_key = "model-a"
        future = MagicMock()
        dep.from_cache.return_value = future

        with patch.object(controller, "build") as mock_build:
            mock_build.return_value = DeploymentDelta([], [dep], [], [])
            with patch("asyncio.create_task") as mock_ct:
                controller.apply()
                mock_ct.assert_called_once()

    def test_from_cache_returns_none_cleans_up(self, controller):
        """from_cache() returns None → deployment is deleted and removed."""
        from src.services.ray.src.ray.deployments.controller.controller import (
            DeploymentDelta,
        )

        dep = MagicMock()
        dep.model_key = "model-a"
        dep.from_cache.return_value = None

        with patch.object(controller, "build") as mock_build:
            mock_build.return_value = DeploymentDelta([], [dep], [], [])
            with patch.object(controller, "_remove_deployment_from_state") as mock_rm:
                controller.apply()
        dep.delete.assert_called_once()
        mock_rm.assert_called_once_with(dep)

    def test_create_spawns_monitor_task(self, controller):
        """create() deployment → gets actor handle → asyncio.create_task."""
        from src.services.ray.src.ray.deployments.controller.controller import (
            DeploymentDelta,
        )

        dep = MagicMock()
        dep.model_key = "model-a"
        dep.replica_id = "r1"
        dep.gpu_mem_bytes_by_id = {0: 80 * GIB}
        dep.gpu_memory_fraction = 0.5
        mock_actor = MagicMock()
        dep.actor = mock_actor
        mock_actor.__ray_ready__ = MagicMock()
        ready_future = MagicMock()
        mock_actor.__ray_ready__.remote.return_value = ready_future

        with patch.object(controller, "build") as mock_build:
            mock_build.return_value = DeploymentDelta([], [], [("test-node", dep)], [])
            with patch("asyncio.create_task") as mock_ct:
                controller.apply()
                mock_ct.assert_called_once()
        dep.create.assert_called_once()


# ============================================================================
# TestRemoveDeploymentFromState
# ============================================================================


class TestRemoveDeploymentFromState:
    """Tests for _remove_deployment_from_state()."""

    def test_removes_from_state_dict(self, controller):
        """Verify the deployment's key is removed from the state dictionary."""
        dep = _make_deployment(model_key="model-a", replica_id="r1", node_id="n1")
        controller.state = {("n1", "model-a", "r1"): dep}
        controller.cluster.nodes = {}

        controller._remove_deployment_from_state(dep)
        assert ("n1", "model-a", "r1") not in controller.state

    def test_removes_from_node_deployments(self, controller):
        """Verify the deployment is also removed from the node's deployments map."""
        from src.services.ray.src.ray.deployments.controller.cluster.node import (
            Node,
            Resources,
        )

        resources = Resources(
            total_gpus=4,
            gpu_type="A100",
            gpu_memory_bytes=80 * GIB,
            cpu_memory_bytes=256 * GIB,
            available_cpu_memory_bytes=256 * GIB,
            available_gpus=[1, 2, 3],  # gpu 0 used by deployment
            gpu_memory_available_bytes_by_id={i: 80 * GIB for i in range(4)},
        )
        node = Node("n1", "test-node", resources)
        dep = _make_deployment(
            model_key="model-a", replica_id="r1", node_id="n1",
            gpu_mem_bytes_by_id={0: 80 * GIB},
        )
        node._set_in_map(node.deployments, dep)
        controller.cluster.nodes = {"n1": node}
        controller.state = {("n1", "model-a", "r1"): dep}

        controller._remove_deployment_from_state(dep)
        assert "model-a" not in node.deployments

    def test_noop_for_missing_key(self, controller):
        """Removing a deployment not in state is a no-op."""
        dep = _make_deployment(model_key="model-a", replica_id="r1", node_id="n1")
        controller.state = {}
        controller.cluster.nodes = {}

        # Should not raise
        controller._remove_deployment_from_state(dep)


# ============================================================================
# TestMonitorDeployment
# ============================================================================


class TestMonitorDeployment:
    """Tests for _monitor_deployment()."""

    @pytest.mark.asyncio
    async def test_success_does_not_cleanup(self, controller):
        """Verify successful monitoring does not trigger delete or state removal."""
        dep = MagicMock()
        dep.model_key = "model-a"
        future = MagicMock()

        async def noop():
            return None

        with patch(f"{_MOD}.ray"):
            with patch(f"{_MOD}.asyncio") as mock_asyncio:
                loop = MagicMock()
                mock_asyncio.get_event_loop.return_value = loop
                loop.run_in_executor.return_value = noop()
                await controller._monitor_deployment(future, dep, "create")

        dep.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_failure_cleans_up(self, controller):
        """Verify monitoring failure triggers delete and state removal."""
        dep = MagicMock()
        dep.model_key = "model-a"
        future = MagicMock()

        with patch(f"{_MOD}.ray"):
            with patch(f"{_MOD}.asyncio") as mock_asyncio:
                loop = MagicMock()
                mock_asyncio.get_event_loop.return_value = loop

                async def raise_error():
                    raise RuntimeError("deploy failed")

                loop.run_in_executor.return_value = raise_error()
                with patch.object(
                    controller, "_remove_deployment_from_state"
                ) as mock_rm:
                    await controller._monitor_deployment(future, dep, "create")

        dep.delete.assert_called_once()
        mock_rm.assert_called_once_with(dep)

    @pytest.mark.asyncio
    async def test_delete_error_swallowed(self, controller):
        """If delete() raises during cleanup, it's caught."""
        dep = MagicMock()
        dep.model_key = "model-a"
        dep.delete.side_effect = RuntimeError("already gone")
        future = MagicMock()

        with patch(f"{_MOD}.ray"):
            with patch(f"{_MOD}.asyncio") as mock_asyncio:
                loop = MagicMock()
                mock_asyncio.get_event_loop.return_value = loop

                async def raise_error():
                    raise RuntimeError("deploy failed")

                loop.run_in_executor.return_value = raise_error()
                with patch.object(
                    controller, "_remove_deployment_from_state"
                ) as mock_rm:
                    # Should not raise despite delete failing
                    await controller._monitor_deployment(future, dep, "create")

        mock_rm.assert_called_once_with(dep)


# ============================================================================
# TestControllerScale
# ============================================================================


class TestControllerScale:
    """Tests for scale() and scale_up()."""

    def test_scale_no_op_when_already_at_target(self, controller):
        """scale() returns without deploying if current >= target."""
        node = MagicMock()
        node.deployments = {"model-a": {"r1": MagicMock(), "r2": MagicMock()}}
        controller.cluster.nodes = {"n1": node}

        result = controller.scale("model-a", replicas=2)
        assert result["changed"] is False
        assert result["current_replicas"] == 2
        controller.cluster.deploy.assert_not_called()

    def test_scale_deploys_when_below_target(self, controller):
        """Verify scale() triggers deploy when current replicas are below the target."""
        node = MagicMock()
        node.deployments = {"model-a": {"r1": MagicMock()}}
        controller.cluster.nodes = {"n1": node}
        controller.cluster.deploy.return_value = ({"result": {}}, True)

        with patch.object(controller, "apply"):
            result = controller.scale("model-a", replicas=3)
        assert result["changed"] is True
        assert result["target_replicas"] == 3
        assert controller.desired_replicas["model-a"] == 3

    def test_scale_rejects_zero_replicas(self, controller):
        """Verify scale() raises ValueError when replicas is zero."""
        with pytest.raises(ValueError, match="positive"):
            controller.scale("model-a", replicas=0)

    def test_scale_up_adds_replicas(self, controller):
        """Verify scale_up() adds the specified count to the current replica count."""
        node = MagicMock()
        node.deployments = {"model-a": {"r1": MagicMock()}}
        controller.cluster.nodes = {"n1": node}
        controller.cluster.deploy.return_value = ({"result": {}}, True)

        with patch.object(controller, "apply"):
            result = controller.scale_up("model-a", replicas=2)
        # current=1 + 2 = target 3
        assert result["target_replicas"] == 3
        assert result["current_replicas"] == 1
        assert controller.desired_replicas["model-a"] == 3

    def test_scale_up_rejects_zero(self, controller):
        """Verify scale_up() raises ValueError when replicas is zero."""
        with pytest.raises(ValueError, match="positive"):
            controller.scale_up("model-a", replicas=0)

    def test_scale_adjusts_for_cant_accommodate(self, controller):
        """scale() calls _adjust_desired_for_cant_accommodate on results."""
        node = MagicMock()
        node.deployments = {}
        controller.cluster.nodes = {"n1": node}
        results = {"result": {("model-a", "r1"): "CANT_ACCOMMODATE"}}
        controller.cluster.deploy.return_value = (results, False)

        result = controller.scale("model-a", replicas=2)
        # desired was set to 2, then decremented by 1 for CANT_ACCOMMODATE
        assert controller.desired_replicas["model-a"] == 1


# ============================================================================
# TestGetState
# ============================================================================


class TestGetState:
    """Tests for get_state()."""

    def test_returns_expected_keys(self, controller):
        """Verify get_state() returns all expected top-level keys."""
        controller.cluster.get_state.return_value = {"nodes": []}
        state = controller.get_state()
        assert "cluster" in state
        assert "execution_timeout_seconds" in state
        assert "model_cache_percentage" in state
        assert "minimum_deployment_time_seconds" in state
        assert "replica_count" in state
        assert "datetime" in state

    def test_values_match_init(self, controller):
        """Verify get_state() returns the configuration values passed at init."""
        controller.cluster.get_state.return_value = {}
        state = controller.get_state()
        assert state["execution_timeout_seconds"] == 3600.0
        assert state["model_cache_percentage"] == 0.5
        assert state["minimum_deployment_time_seconds"] == 3600.0
        assert state["replica_count"] == 1


# ============================================================================
# TestControllerInit
# ============================================================================


class TestControllerInit:
    """Tests for __init__() — verifying initialization flow."""

    def _make_actor(self, deployments=None, **overrides):
        """Construct a _ControllerActor with all external deps mocked."""
        if deployments is None:
            deployments = []
        kwargs = dict(
            deployments=deployments,
            model_import_path="test:app",
            execution_timeout_seconds=3600.0,
            model_cache_percentage=0.5,
            minimum_deployment_time_seconds=3600.0,
        )
        kwargs.update(overrides)
        with patch(f"{_MOD}.set_logger", return_value=MagicMock()):
            with patch("asyncio.create_task") as mock_ct:
                with patch(f"{_MOD}.Cluster") as MockCluster:
                    mock_cluster = MagicMock()
                    mock_cluster.nodes = {}
                    MockCluster.return_value = mock_cluster
                    with patch(f"{_MOD}.ray") as mock_ray:
                        mock_ray.get_runtime_context.return_value = MagicMock()
                        from src.services.ray.src.ray.deployments.controller.controller import (
                            _ControllerActor,
                        )

                        actor = _ControllerActor(**kwargs)
        return actor, MockCluster, mock_ct, mock_cluster

    def test_instance_vars_set(self):
        """Verify instance variables are set from constructor arguments."""
        actor, _, _, _ = self._make_actor()
        assert actor.model_import_path == "test:app"
        assert actor.execution_timeout_seconds == 3600.0
        assert actor.minimum_deployment_time_seconds == 3600.0
        assert actor.model_cache_percentage == 0.5
        assert actor.replica_count == 1
        assert actor.desired_replicas == {}
        assert actor.state == {}

    def test_cluster_created_with_correct_params(self):
        """Verify Cluster is created with minimum_deployment_time_seconds and model_cache_percentage."""
        _, MockCluster, _, _ = self._make_actor(
            minimum_deployment_time_seconds=7200.0,
            model_cache_percentage=0.8,
        )
        MockCluster.assert_called_once_with(
            minimum_deployment_time_seconds=7200.0,
            model_cache_percentage=0.8,
        )

    def test_update_nodes_called_during_init(self):
        """Verify cluster.update_nodes() is called during initialization."""
        _, _, _, mock_cluster = self._make_actor()
        mock_cluster.update_nodes.assert_called_once()

    def test_empty_deployments_no_deploy(self):
        """Verify _deploy is NOT called when deployments list is empty."""
        with patch(f"{_MOD}.set_logger", return_value=MagicMock()):
            with patch("asyncio.create_task"):
                with patch(f"{_MOD}.Cluster") as MockCluster:
                    mock_cluster = MagicMock()
                    mock_cluster.nodes = {}
                    MockCluster.return_value = mock_cluster
                    with patch(f"{_MOD}.ray") as mock_ray:
                        mock_ray.get_runtime_context.return_value = MagicMock()
                        from src.services.ray.src.ray.deployments.controller.controller import (
                            _ControllerActor,
                        )

                        actor = _ControllerActor(
                            deployments=[],
                            model_import_path="test:app",
                            execution_timeout_seconds=3600.0,
                            model_cache_percentage=0.5,
                            minimum_deployment_time_seconds=3600.0,
                        )
        # cluster.deploy should not have been called
        mock_cluster.deploy.assert_not_called()

    def test_nonempty_deployments_calls_deploy(self):
        """Verify _deploy is called with dedicated=True for non-empty deployments."""
        with patch(f"{_MOD}.set_logger", return_value=MagicMock()):
            with patch("asyncio.create_task"):
                with patch(f"{_MOD}.Cluster") as MockCluster:
                    mock_cluster = MagicMock()
                    mock_cluster.nodes = {}
                    mock_cluster.deploy.return_value = ({"result": {}}, False)
                    MockCluster.return_value = mock_cluster
                    with patch(f"{_MOD}.ray") as mock_ray:
                        mock_ray.get_runtime_context.return_value = MagicMock()
                        from src.services.ray.src.ray.deployments.controller.controller import (
                            _ControllerActor,
                        )

                        actor = _ControllerActor(
                            deployments=["model-a", "model-b"],
                            model_import_path="test:app",
                            execution_timeout_seconds=3600.0,
                            model_cache_percentage=0.5,
                            minimum_deployment_time_seconds=3600.0,
                        )
        mock_cluster.deploy.assert_called_once_with(
            ["model-a", "model-b"], dedicated=True, replicas=1
        )

    def test_empty_string_deployments_no_deploy(self):
        """Verify _deploy is NOT called when deployments is [''] (env var split artifact)."""
        with patch(f"{_MOD}.set_logger", return_value=MagicMock()):
            with patch("asyncio.create_task"):
                with patch(f"{_MOD}.Cluster") as MockCluster:
                    mock_cluster = MagicMock()
                    mock_cluster.nodes = {}
                    MockCluster.return_value = mock_cluster
                    with patch(f"{_MOD}.ray") as mock_ray:
                        mock_ray.get_runtime_context.return_value = MagicMock()
                        from src.services.ray.src.ray.deployments.controller.controller import (
                            _ControllerActor,
                        )

                        actor = _ControllerActor(
                            deployments=[""],
                            model_import_path="test:app",
                            execution_timeout_seconds=3600.0,
                            model_cache_percentage=0.5,
                            minimum_deployment_time_seconds=3600.0,
                        )
        mock_cluster.deploy.assert_not_called()

    def test_check_nodes_task_created(self):
        """Verify asyncio.create_task is called for check_nodes()."""
        _, _, mock_ct, _ = self._make_actor()
        mock_ct.assert_called_once()


# ============================================================================
# TestCheckNodes
# ============================================================================


class TestCheckNodes:
    """Tests for check_nodes() async loop."""

    @pytest.mark.asyncio
    async def test_calls_update_nodes(self, controller):
        """Verify check_nodes calls cluster.update_nodes() each iteration."""
        with patch("asyncio.sleep", side_effect=StopAsyncIteration):
            with pytest.raises(StopAsyncIteration):
                await controller.check_nodes()
        controller.cluster.update_nodes.assert_called_once()

    @pytest.mark.asyncio
    async def test_sleeps_default_interval(self, controller):
        """Verify check_nodes sleeps for 30s by default."""
        with patch("asyncio.sleep", side_effect=StopAsyncIteration) as mock_sleep:
            with pytest.raises(StopAsyncIteration):
                await controller.check_nodes()
        mock_sleep.assert_called_once_with(30)

    @pytest.mark.asyncio
    async def test_respects_env_var_override(self, controller):
        """Verify check_nodes respects NDIF_CONTROLLER_SYNC_INTERVAL_S env var."""
        with patch.dict(os.environ, {"NDIF_CONTROLLER_SYNC_INTERVAL_S": "60"}):
            with patch("asyncio.sleep", side_effect=StopAsyncIteration) as mock_sleep:
                with pytest.raises(StopAsyncIteration):
                    await controller.check_nodes()
        mock_sleep.assert_called_once_with(60)


# ============================================================================
# TestGetDeploymentForReplica
# ============================================================================


class TestGetDeploymentForReplica:
    """Tests for get_deployment_for_replica()."""

    def test_found_returns_get_state(self, controller):
        """Verify returns deployment.get_state() when found."""
        dep = _make_deployment(model_key="model-a", replica_id="r1")
        node = MagicMock()
        node.deployments = {"model-a": {"r1": dep}}
        controller.cluster.nodes = {"n1": node}

        result = controller.get_deployment_for_replica("model-a", "r1")
        assert result == dep.get_state()

    def test_not_found_returns_not_found_dict(self, controller):
        """Verify returns not_found dict when replica is not in any node."""
        node = MagicMock()
        node.deployments = {}
        controller.cluster.nodes = {"n1": node}

        result = controller.get_deployment_for_replica("model-a", "r1")
        assert result == {
            "model_key": "model-a",
            "replica_id": "r1",
            "deployment_state": "not_found",
        }

    def test_searches_across_nodes(self, controller):
        """Verify searches multiple nodes to find the replica."""
        dep = _make_deployment(model_key="model-a", replica_id="r1", node_id="n2")
        node1 = MagicMock()
        node1.deployments = {}
        node2 = MagicMock()
        node2.deployments = {"model-a": {"r1": dep}}
        controller.cluster.nodes = {"n1": node1, "n2": node2}

        result = controller.get_deployment_for_replica("model-a", "r1")
        assert result == dep.get_state()


# ============================================================================
# TestGetDeployment
# ============================================================================


class TestGetDeployment:
    """Tests for get_deployment()."""

    def test_found_returns_all_replicas(self, controller):
        """Verify returns dict of replica_id → get_state() for all replicas."""
        dep1 = _make_deployment(model_key="model-a", replica_id="r1")
        dep2 = _make_deployment(model_key="model-a", replica_id="r2")
        node = MagicMock()
        node.deployments = {"model-a": {"r1": dep1, "r2": dep2}}
        controller.cluster.nodes = {"n1": node}

        result = controller.get_deployment("model-a")
        assert result == {
            "r1": dep1.get_state(),
            "r2": dep2.get_state(),
        }

    def test_not_found_returns_not_found_dict(self, controller):
        """Verify returns not_found dict when model has no deployments."""
        node = MagicMock()
        node.deployments = {}
        controller.cluster.nodes = {"n1": node}

        result = controller.get_deployment("model-a")
        assert result == {
            "model_key": "model-a",
            "deployments_state": "not_found",
        }

    def test_multiple_replicas_across_nodes(self, controller):
        """Verify collects replicas from multiple nodes."""
        dep1 = _make_deployment(model_key="model-a", replica_id="r1", node_id="n1")
        dep2 = _make_deployment(model_key="model-a", replica_id="r2", node_id="n2")
        node1 = MagicMock()
        node1.deployments = {"model-a": {"r1": dep1}}
        node2 = MagicMock()
        node2.deployments = {"model-a": {"r2": dep2}}
        controller.cluster.nodes = {"n1": node1, "n2": node2}

        result = controller.get_deployment("model-a")
        assert result == {
            "r1": dep1.get_state(),
            "r2": dep2.get_state(),
        }


# ============================================================================
# TestEnv
# ============================================================================


class TestEnv:
    """Tests for env()."""

    def test_returns_python_version_and_packages(self, controller):
        """Verify env() returns dict with python_version and packages keys."""
        mock_dist = MagicMock()
        mock_dist.metadata = {"Name": "requests"}
        mock_dist.version = "2.31.0"

        with patch(f"{_MOD}.distributions", return_value=[mock_dist]):
            with patch(
                f"{_MOD}.packages_distributions",
                return_value={"requests": ["requests"]},
            ):
                result = controller.env()

        assert "python_version" in result
        assert "packages" in result

    def test_python_version_matches_sys(self, controller):
        """Verify python_version matches sys.version."""
        with patch(f"{_MOD}.distributions", return_value=[]):
            with patch(f"{_MOD}.packages_distributions", return_value={}):
                result = controller.env()
        assert result["python_version"] == sys.version

    def test_packages_maps_import_name_to_version(self, controller):
        """Verify packages maps import names to versions using packages_distributions."""
        mock_dist = MagicMock()
        mock_dist.metadata = {"Name": "Pillow"}
        mock_dist.version = "10.0.0"

        with patch(f"{_MOD}.distributions", return_value=[mock_dist]):
            with patch(
                f"{_MOD}.packages_distributions",
                return_value={"PIL": ["Pillow"]},
            ):
                result = controller.env()

        assert result["packages"]["PIL"] == "10.0.0"

    def test_packages_fallback_to_dist_name(self, controller):
        """Verify packages falls back to dist name when no import mapping exists."""
        mock_dist = MagicMock()
        mock_dist.metadata = {"Name": "my-tool"}
        mock_dist.version = "1.0.0"

        with patch(f"{_MOD}.distributions", return_value=[mock_dist]):
            with patch(f"{_MOD}.packages_distributions", return_value={}):
                result = controller.env()

        assert result["packages"]["my-tool"] == "1.0.0"


# ============================================================================
# TestStatus
# ============================================================================


class TestStatus:
    """Tests for status() — comprehensive deployment/cluster status."""

    def _mock_actor_state(self, name, state):
        """Create a mock actor state object."""
        actor = MagicMock()
        actor.name = name
        actor.state = state
        return actor

    def _mock_evaluator_cache_entry(
        self, repo_id="org/model", revision="main", n_params=1000000
    ):
        """Create a mock evaluator cache entry."""
        entry = MagicMock()
        entry.config._name_or_path = repo_id
        entry.revision = revision
        entry.config.to_json_string.return_value = '{"key": "value"}'
        entry.n_params = n_params
        return entry

    def test_empty_cluster_returns_empty(self, controller):
        """Verify status() returns empty dicts when cluster has no nodes."""
        controller.cluster.nodes = {}
        with patch(f"{_MOD}.list_actors", return_value=[]):
            with patch(f"{_MOD}.get_downloaded_models", return_value=[]):
                result = controller.status()
        assert result["deployments"] == {}
        assert result["cluster"]["nodes"] == {}

    def test_hot_deployment_included(self, controller):
        """Verify HOT deployment includes all expected fields."""
        dep = _make_deployment(
            model_key="model-a", replica_id="r1",
            deployment_level=DeploymentLevel.HOT, dedicated=True,
        )
        node = MagicMock()
        node.deployments = {"model-a": {"r1": dep}}
        node.cache = {}
        node.resources = MagicMock()
        node.resources.total_gpus = 4
        node.resources.gpu_memory_bytes = 80 * GIB
        node.resources.available_gpus = [2, 3]
        controller.cluster.nodes = {"n1": node}

        cache_entry = self._mock_evaluator_cache_entry()
        controller.cluster.evaluator = MagicMock()
        controller.cluster.evaluator.cache = {"model-a": cache_entry}

        actor_state = self._mock_actor_state("ModelActor:model-a:r1", "ALIVE")
        with patch(f"{_MOD}.list_actors", return_value=[actor_state]):
            with patch(f"{_MOD}.get_downloaded_models", return_value=[]):
                result = controller.status()

        app_name = "ModelActor:model-a:r1"
        dep_status = result["deployments"][app_name]
        assert dep_status["deployment_level"] == "HOT"
        assert dep_status["dedicated"] is True
        assert dep_status["model_key"] == "model-a"
        assert dep_status["replica_id"] == "r1"
        assert dep_status["repo_id"] == "org/model"
        assert dep_status["revision"] == "main"
        assert dep_status["n_params"] == 1000000
        assert dep_status["application_state"] == "RUNNING"

    def test_warm_cached_deployment(self, controller):
        """Verify WARM cached deployment includes deployment_level=WARM."""
        cached_dep = _make_deployment(
            model_key="model-b", replica_id="r1",
            deployment_level=DeploymentLevel.WARM,
        )
        node = MagicMock()
        node.deployments = {}
        node.cache = {"model-b": {"r1": cached_dep}}
        node.resources = MagicMock()
        node.resources.total_gpus = 4
        node.resources.gpu_memory_bytes = 80 * GIB
        node.resources.available_gpus = [0, 1, 2, 3]
        controller.cluster.nodes = {"n1": node}

        cache_entry = self._mock_evaluator_cache_entry(repo_id="org/model-b")
        controller.cluster.evaluator = MagicMock()
        controller.cluster.evaluator.cache = {"model-b": cache_entry}

        with patch(f"{_MOD}.list_actors", return_value=[]):
            with patch(f"{_MOD}.get_downloaded_models", return_value=[]):
                result = controller.status()

        app_name = "ModelActor:model-b:r1"
        dep_status = result["deployments"][app_name]
        assert dep_status["deployment_level"] == "WARM"
        assert dep_status["model_key"] == "model-b"
        assert dep_status["repo_id"] == "org/model-b"

    def test_non_dedicated_has_schedule(self, controller):
        """Verify non-dedicated deployment with minimum_deployment_time has schedule.end_time."""
        dep = _make_deployment(
            model_key="model-a", replica_id="r1",
            deployment_level=DeploymentLevel.HOT, dedicated=False,
        )
        node = MagicMock()
        node.deployments = {"model-a": {"r1": dep}}
        node.cache = {}
        node.resources = MagicMock()
        node.resources.total_gpus = 4
        node.resources.gpu_memory_bytes = 80 * GIB
        node.resources.available_gpus = [2, 3]
        controller.cluster.nodes = {"n1": node}
        controller.minimum_deployment_time_seconds = 3600.0

        cache_entry = self._mock_evaluator_cache_entry()
        controller.cluster.evaluator = MagicMock()
        controller.cluster.evaluator.cache = {"model-a": cache_entry}

        actor_state = self._mock_actor_state("ModelActor:model-a:r1", "ALIVE")
        with patch(f"{_MOD}.list_actors", return_value=[actor_state]):
            with patch(f"{_MOD}.get_downloaded_models", return_value=[]):
                result = controller.status()

        app_name = "ModelActor:model-a:r1"
        assert "schedule" in result["deployments"][app_name]
        assert "end_time" in result["deployments"][app_name]["schedule"]

    def test_dedicated_no_schedule(self, controller):
        """Verify dedicated deployment does NOT have a schedule field."""
        dep = _make_deployment(
            model_key="model-a", replica_id="r1",
            deployment_level=DeploymentLevel.HOT, dedicated=True,
        )
        node = MagicMock()
        node.deployments = {"model-a": {"r1": dep}}
        node.cache = {}
        node.resources = MagicMock()
        node.resources.total_gpus = 4
        node.resources.gpu_memory_bytes = 80 * GIB
        node.resources.available_gpus = [2, 3]
        controller.cluster.nodes = {"n1": node}

        cache_entry = self._mock_evaluator_cache_entry()
        controller.cluster.evaluator = MagicMock()
        controller.cluster.evaluator.cache = {"model-a": cache_entry}

        actor_state = self._mock_actor_state("ModelActor:model-a:r1", "ALIVE")
        with patch(f"{_MOD}.list_actors", return_value=[actor_state]):
            with patch(f"{_MOD}.get_downloaded_models", return_value=[]):
                result = controller.status()

        app_name = "ModelActor:model-a:r1"
        assert "schedule" not in result["deployments"][app_name]

    def test_cold_downloaded_model(self, controller):
        """Verify downloaded but not deployed model is listed as COLD."""
        controller.cluster.nodes = {}

        with patch(f"{_MOD}.list_actors", return_value=[]):
            with patch(
                f"{_MOD}.get_downloaded_models", return_value=["org/cold-model"]
            ):
                result = controller.status()

        assert result["deployments"]["org/cold-model"] == {
            "deployment_level": "COLD",
            "repo_id": "org/cold-model",
        }

    def test_already_deployed_repo_not_duplicated_as_cold(self, controller):
        """Verify a repo_id that is already deployed is NOT also listed as COLD."""
        dep = _make_deployment(
            model_key="model-a", replica_id="r1",
            deployment_level=DeploymentLevel.HOT, dedicated=True,
        )
        node = MagicMock()
        node.deployments = {"model-a": {"r1": dep}}
        node.cache = {}
        node.resources = MagicMock()
        node.resources.total_gpus = 4
        node.resources.gpu_memory_bytes = 80 * GIB
        node.resources.available_gpus = [2, 3]
        controller.cluster.nodes = {"n1": node}

        cache_entry = self._mock_evaluator_cache_entry(repo_id="org/model-a")
        controller.cluster.evaluator = MagicMock()
        controller.cluster.evaluator.cache = {"model-a": cache_entry}

        actor_state = self._mock_actor_state("ModelActor:model-a:r1", "ALIVE")
        with patch(f"{_MOD}.list_actors", return_value=[actor_state]):
            with patch(
                f"{_MOD}.get_downloaded_models", return_value=["org/model-a"]
            ):
                result = controller.status()

        # org/model-a should NOT appear as a separate COLD entry
        cold_entries = [
            k for k, v in result["deployments"].items()
            if v.get("deployment_level") == "COLD"
        ]
        assert len(cold_entries) == 0

    def test_ray_actor_states_mapped(self, controller):
        """Verify Ray actor states are mapped correctly."""
        dep_alive = _make_deployment(
            model_key="model-a", replica_id="r1",
            deployment_level=DeploymentLevel.HOT, dedicated=True,
        )
        dep_pending = _make_deployment(
            model_key="model-b", replica_id="r1",
            deployment_level=DeploymentLevel.HOT, dedicated=True,
        )
        dep_dead = _make_deployment(
            model_key="model-c", replica_id="r1",
            deployment_level=DeploymentLevel.HOT, dedicated=True,
        )
        node = MagicMock()
        node.deployments = {
            "model-a": {"r1": dep_alive},
            "model-b": {"r1": dep_pending},
            "model-c": {"r1": dep_dead},
        }
        node.cache = {}
        node.resources = MagicMock()
        node.resources.total_gpus = 8
        node.resources.gpu_memory_bytes = 80 * GIB
        node.resources.available_gpus = [4, 5, 6, 7]
        controller.cluster.nodes = {"n1": node}

        cache_a = self._mock_evaluator_cache_entry(repo_id="org/a")
        cache_b = self._mock_evaluator_cache_entry(repo_id="org/b")
        cache_c = self._mock_evaluator_cache_entry(repo_id="org/c")
        controller.cluster.evaluator = MagicMock()
        controller.cluster.evaluator.cache = {
            "model-a": cache_a,
            "model-b": cache_b,
            "model-c": cache_c,
        }

        actors = [
            self._mock_actor_state("ModelActor:model-a:r1", "ALIVE"),
            self._mock_actor_state("ModelActor:model-b:r1", "PENDING_CREATION"),
            self._mock_actor_state("ModelActor:model-c:r1", "DEAD"),
        ]
        with patch(f"{_MOD}.list_actors", return_value=actors):
            with patch(f"{_MOD}.get_downloaded_models", return_value=[]):
                result = controller.status()

        assert result["deployments"]["ModelActor:model-a:r1"]["application_state"] == "RUNNING"
        assert result["deployments"]["ModelActor:model-b:r1"]["application_state"] == "DEPLOYING"
        assert result["deployments"]["ModelActor:model-c:r1"]["application_state"] == "UNHEALTHY"

    def test_cluster_node_resources_in_output(self, controller):
        """Verify cluster section includes node resources and deployment info."""
        dep = _make_deployment(
            model_key="model-a", replica_id="r1",
            deployment_level=DeploymentLevel.HOT, dedicated=True,
        )
        node = MagicMock()
        node.deployments = {"model-a": {"r1": dep}}
        node.cache = {}
        node.resources = MagicMock()
        node.resources.total_gpus = 4
        node.resources.gpu_memory_bytes = 80 * GIB
        node.resources.available_gpus = [2, 3]
        controller.cluster.nodes = {"n1": node}

        cache_entry = self._mock_evaluator_cache_entry()
        controller.cluster.evaluator = MagicMock()
        controller.cluster.evaluator.cache = {"model-a": cache_entry}

        actor_state = self._mock_actor_state("ModelActor:model-a:r1", "ALIVE")
        with patch(f"{_MOD}.list_actors", return_value=[actor_state]):
            with patch(f"{_MOD}.get_downloaded_models", return_value=[]):
                result = controller.status()

        cluster_node = result["cluster"]["nodes"]["n1"]
        assert cluster_node["resources"]["total_gpus"] == 4
        assert cluster_node["resources"]["gpu_memory_bytes"] == 80 * GIB
        assert cluster_node["resources"]["available_gpus"] == [2, 3]
        assert "model-a:r1" in cluster_node["deployments"]
