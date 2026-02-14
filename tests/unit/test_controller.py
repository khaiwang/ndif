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
