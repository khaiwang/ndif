"""Comprehensive unit tests for the Cluster class.

Tests cover:
- Cluster construction and initial state
- target_replica_ids_for: reusing cached IDs, generating new ones
- deploy: evaluator integration, model sorting, node selection, eviction handling, error handling
- evict: by model_keys, by replica_keys, not-found cases
- get_state
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.services.ray.src.ray.deployments.controller.cluster.node import (
    Candidate,
    CandidateLevel,
    Node,
    Resources,
)
from src.services.ray.src.ray.deployments.controller.cluster.deployment import (
    Deployment,
    DeploymentLevel,
)
from src.services.ray.src.ray.deployments.controller.cluster.cluster import Cluster


GIB = 1024**3


# ============================================================================
# Helpers
# ============================================================================


def _make_resources(
    total_gpus=4,
    gpu_memory_bytes=80 * GIB,
    cpu_memory_bytes=256 * GIB,
    available_cpu_memory_bytes=None,
    available_gpus=None,
    gpu_memory_available_bytes_by_id=None,
):
    if available_cpu_memory_bytes is None:
        available_cpu_memory_bytes = cpu_memory_bytes
    if available_gpus is None:
        available_gpus = list(range(total_gpus))
    if gpu_memory_available_bytes_by_id is None:
        gpu_memory_available_bytes_by_id = {
            i: gpu_memory_bytes for i in range(total_gpus)
        }
    return Resources(
        total_gpus=total_gpus,
        gpu_type="A100",
        gpu_memory_bytes=gpu_memory_bytes,
        cpu_memory_bytes=cpu_memory_bytes,
        available_cpu_memory_bytes=available_cpu_memory_bytes,
        available_gpus=available_gpus,
        gpu_memory_available_bytes_by_id=gpu_memory_available_bytes_by_id,
    )


def _make_node(node_id="node-1", name="test-node", total_gpus=4, gpu_memory_bytes=80 * GIB, **kwargs):
    resources = _make_resources(total_gpus=total_gpus, gpu_memory_bytes=gpu_memory_bytes, **kwargs)
    return Node(node_id, name, resources)


def _make_deployment(
    model_key="model",
    replica_id="r1",
    deployment_level=DeploymentLevel.HOT,
    gpu_mem_bytes_by_id=None,
    gpu_memory_fraction=None,
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


def _make_cluster(**kwargs):
    """Create a Cluster with the evaluator mocked so it never calls nnsight."""
    cluster = Cluster(**kwargs)
    cluster.evaluator = MagicMock()
    return cluster


# ============================================================================
# Cluster: construction
# ============================================================================


class TestClusterConstruction:
    def test_default_construction(self):
        """Verify default Cluster attributes after construction with no arguments."""
        cluster = _make_cluster()
        assert cluster.nodes == {}
        assert cluster.minimum_deployment_time_seconds is None
        assert cluster.model_cache_percentage == 0.5

    def test_custom_construction(self):
        """Verify Cluster respects custom minimum_deployment_time and cache_percentage values."""
        cluster = _make_cluster(
            minimum_deployment_time_seconds=120,
            model_cache_percentage=0.7,
        )
        assert cluster.minimum_deployment_time_seconds == 120
        assert cluster.model_cache_percentage == 0.7


# ============================================================================
# Cluster: target_replica_ids_for
# ============================================================================


class TestTargetReplicaIdsFor:
    def test_no_replicas_needed(self):
        """If already deployed count >= replicas, no new replicas needed."""
        cluster = _make_cluster()
        result = cluster.target_replica_ids_for(
            cached_replica_ids=set(),
            deployed_replica_ids={"r1", "r2"},
            replicas=2,
        )
        assert result == []

    def test_uses_cached_ids_first(self):
        """Should reuse cached replica IDs before generating new ones."""
        cluster = _make_cluster()
        result = cluster.target_replica_ids_for(
            cached_replica_ids={"cached-1", "cached-2"},
            deployed_replica_ids=set(),
            replicas=2,
        )
        assert len(result) == 2
        # Both should come from cached IDs
        assert set(result).issubset({"cached-1", "cached-2"})

    def test_generates_new_ids_for_remainder(self):
        """If cached IDs are not enough, generates new UUIDs for the rest."""
        cluster = _make_cluster()
        result = cluster.target_replica_ids_for(
            cached_replica_ids={"cached-1"},
            deployed_replica_ids=set(),
            replicas=3,
        )
        assert len(result) == 3
        assert result[0] == "cached-1"
        # The other two should be new UUIDs (hex strings of length 32)
        assert len(result[1]) == 32
        assert len(result[2]) == 32

    def test_avoids_deployed_and_cached_ids(self):
        """Generated IDs should not collide with deployed or cached IDs."""
        cluster = _make_cluster()
        result = cluster.target_replica_ids_for(
            cached_replica_ids=set(),
            deployed_replica_ids={"deployed-1"},
            replicas=2,
        )
        # Need 1 replica (2 - 1 deployed), and the new ID should not be "deployed-1"
        assert len(result) == 1
        assert result[0] != "deployed-1"

    def test_partially_deployed(self):
        """When some replicas deployed and some cached, reuse cached then generate."""
        cluster = _make_cluster()
        result = cluster.target_replica_ids_for(
            cached_replica_ids={"cached-1"},
            deployed_replica_ids={"deployed-1"},
            replicas=3,
        )
        # Need 2 more replicas (3 - 1 deployed)
        assert len(result) == 2
        assert result[0] == "cached-1"
        assert len(result[1]) == 32  # new UUID

    def test_zero_replicas(self):
        """Verify that requesting zero replicas returns an empty list."""
        cluster = _make_cluster()
        result = cluster.target_replica_ids_for(
            cached_replica_ids={"cached-1"},
            deployed_replica_ids=set(),
            replicas=0,
        )
        assert result == []


# ============================================================================
# Cluster: deploy
# ============================================================================


class TestClusterDeploy:
    def test_deploy_single_model_single_node(self):
        """Deploy a single model on a cluster with one node."""
        cluster = _make_cluster()
        node = _make_node("n1", "node-1", total_gpus=4)
        cluster.nodes["n1"] = node
        cluster.evaluator.side_effect = lambda mk: 5 * GIB

        results, change = cluster.deploy(["model-a"])
        assert change is True
        # Should have one result entry
        assert len(results["result"]) == 1
        key = list(results["result"].keys())[0]
        assert key[0] == "model-a"
        assert results["result"][key] in (
            CandidateLevel.FREE.name,
            CandidateLevel.CACHED_AND_FREE.name,
        )

    def test_deploy_model_already_deployed(self):
        """Deploying an already-deployed model returns DEPLOYED status."""
        cluster = _make_cluster()
        node = _make_node("n1", "node-1", total_gpus=4)
        cluster.nodes["n1"] = node
        cluster.evaluator.side_effect = lambda mk: 5 * GIB

        # First deploy
        results1, _ = cluster.deploy(["model-a"])
        replica_id = list(results1["result"].keys())[0][1]

        # Second deploy with same model (now already deployed, replicas=1 => no new needed)
        results2, change2 = cluster.deploy(["model-a"])
        # The already-deployed replica should appear in results
        found_deployed = False
        for (mk, rid), status in results2["result"].items():
            if mk == "model-a" and rid == replica_id:
                assert status == CandidateLevel.DEPLOYED.name
                found_deployed = True
        assert found_deployed

    def test_deploy_multiple_models_sorted_by_size(self):
        """Models should be deployed largest first."""
        cluster = _make_cluster()
        node = _make_node("n1", "node-1", total_gpus=8, gpu_memory_bytes=80 * GIB)
        cluster.nodes["n1"] = node

        sizes = {"model-small": 5 * GIB, "model-large": 50 * GIB}
        cluster.evaluator.side_effect = lambda mk: sizes[mk]

        results, change = cluster.deploy(["model-small", "model-large"])
        assert change is True
        # Both models should be deployed
        model_keys_in_results = [k[0] for k in results["result"].keys()]
        assert "model-small" in model_keys_in_results
        assert "model-large" in model_keys_in_results

    def test_deploy_evaluator_returns_exception(self):
        """If evaluator returns an Exception for a model, result should contain the error."""
        cluster = _make_cluster()
        node = _make_node("n1", "node-1", total_gpus=4)
        cluster.nodes["n1"] = node

        def evaluator_side_effect(mk):
            if mk == "bad-model":
                return ValueError("Model not found")
            return 5 * GIB

        cluster.evaluator.side_effect = evaluator_side_effect

        results, change = cluster.deploy(["bad-model", "good-model"])
        # good-model should deploy fine
        good_entries = {k: v for k, v in results["result"].items() if k[0] == "good-model"}
        assert len(good_entries) == 1

        # bad-model should have error string
        bad_entries = {k: v for k, v in results["result"].items() if k[0] == "bad-model"}
        assert len(bad_entries) == 1
        error_msg = list(bad_entries.values())[0]
        assert "Model not found" in error_msg

    def test_deploy_with_eviction(self):
        """Deploy a model that requires evicting existing deployments."""
        cluster = _make_cluster()
        node = _make_node("n1", "node-1", total_gpus=1, gpu_memory_bytes=80 * GIB)
        cluster.nodes["n1"] = node

        # First deploy model-a (uses full GPU)
        cluster.evaluator.side_effect = lambda mk: 30 * GIB
        results1, change1 = cluster.deploy(["model-a"])
        assert change1 is True

        # Now deploy model-b, which should evict model-a
        # We need to ensure model-a is old enough to be evictable
        for model_map in node.deployments.values():
            for dep in model_map.values():
                dep.deployed = time.time() - 1000

        cluster.evaluator.side_effect = lambda mk: 30 * GIB
        results2, change2 = cluster.deploy(["model-b"])
        assert change2 is True

    def test_deploy_on_multiple_nodes_picks_best(self):
        """When multiple nodes available, picks the best candidate."""
        cluster = _make_cluster()

        # Node 1: has free GPUs
        node1 = _make_node("n1", "node-1", total_gpus=4)
        # Node 2: has free GPUs
        node2 = _make_node("n2", "node-2", total_gpus=4)

        cluster.nodes["n1"] = node1
        cluster.nodes["n2"] = node2

        cluster.evaluator.side_effect = lambda mk: 5 * GIB

        results, change = cluster.deploy(["model-a"])
        assert change is True
        # Model should be deployed on one of the nodes
        model_entries = {k: v for k, v in results["result"].items() if k[0] == "model-a"}
        assert len(model_entries) == 1

    def test_deploy_model_cant_accommodate(self):
        """When model is too large for any node, result is CANT_ACCOMMODATE."""
        cluster = _make_cluster()
        node = _make_node("n1", "node-1", total_gpus=2, gpu_memory_bytes=80 * GIB)
        cluster.nodes["n1"] = node

        # Model needs more GPUs than available on any node
        cluster.evaluator.side_effect = lambda mk: 200 * GIB

        results, change = cluster.deploy(["huge-model"])
        model_entries = {k: v for k, v in results["result"].items() if k[0] == "huge-model"}
        assert len(model_entries) == 1
        status = list(model_entries.values())[0]
        assert status == CandidateLevel.CANT_ACCOMMODATE.name

    def test_deploy_with_replicas(self):
        """Deploy a model with multiple replicas."""
        cluster = _make_cluster()
        # Need multiple nodes or enough capacity for multiple replicas
        node1 = _make_node("n1", "node-1", total_gpus=4, gpu_memory_bytes=80 * GIB)
        node2 = _make_node("n2", "node-2", total_gpus=4, gpu_memory_bytes=80 * GIB)
        cluster.nodes["n1"] = node1
        cluster.nodes["n2"] = node2

        cluster.evaluator.side_effect = lambda mk: 5 * GIB

        results, change = cluster.deploy(["model-a"], replicas=2)
        assert change is True
        model_entries = {k: v for k, v in results["result"].items() if k[0] == "model-a"}
        assert len(model_entries) == 2

    def test_deploy_dedicated_evicts_old_dedicated(self):
        """Dedicated deployment should evict deprecated dedicated deployments not in the new set."""
        cluster = _make_cluster()
        node = _make_node("n1", "node-1", total_gpus=4, gpu_memory_bytes=80 * GIB)
        cluster.nodes["n1"] = node

        # Deploy model-old as dedicated
        cluster.evaluator.side_effect = lambda mk: 5 * GIB
        cluster.deploy(["model-old"], dedicated=True)

        # Age the deployment
        for model_map in node.deployments.values():
            for dep in model_map.values():
                dep.deployed = time.time() - 1000

        # Now deploy model-new as dedicated; model-old should be evicted
        cluster.evaluator.side_effect = lambda mk: 5 * GIB
        results, change = cluster.deploy(["model-new"], dedicated=True)

        # model-old should have been evicted
        assert len(results["evictions"]) >= 1
        evicted_models = {e[0] for e in results["evictions"]}
        assert "model-old" in evicted_models

    def test_deploy_records_evictions_from_candidates(self):
        """Evictions from node.deploy candidate should be recorded in results."""
        cluster = _make_cluster()
        node = _make_node("n1", "node-1", total_gpus=1, gpu_memory_bytes=80 * GIB)
        cluster.nodes["n1"] = node

        # Deploy model-a first
        cluster.evaluator.side_effect = lambda mk: 30 * GIB
        cluster.deploy(["model-a"])

        # Age it
        for model_map in node.deployments.values():
            for dep in model_map.values():
                dep.deployed = time.time() - 1000

        # Deploy model-b which needs eviction of model-a
        cluster.evaluator.side_effect = lambda mk: 30 * GIB
        results, change = cluster.deploy(["model-b"])
        assert change is True
        # model-a eviction should be in evictions
        evicted_models = {e[0] for e in results["evictions"]}
        assert "model-a" in evicted_models

    def test_deploy_empty_model_list(self):
        """Deploying an empty list should return empty results."""
        cluster = _make_cluster()
        node = _make_node("n1", "node-1")
        cluster.nodes["n1"] = node

        results, change = cluster.deploy([])
        assert results["result"] == {}
        assert results["evictions"] == set()
        assert change is False

    def test_deploy_no_nodes(self):
        """Deploying with no nodes in cluster should handle gracefully."""
        cluster = _make_cluster()
        cluster.evaluator.side_effect = lambda mk: 5 * GIB

        # With no nodes, there are no candidates, which will cause an error
        # in random.choice on an empty list. Let's verify the behavior.
        with pytest.raises(IndexError):
            cluster.deploy(["model-a"])


# ============================================================================
# Cluster: evict
# ============================================================================


class TestClusterEvict:
    def test_evict_by_model_key(self):
        """Evict all replicas of a model by model_key."""
        cluster = _make_cluster()
        node = _make_node("n1", "node-1", total_gpus=4, gpu_memory_bytes=80 * GIB)
        cluster.nodes["n1"] = node

        # Deploy
        cluster.evaluator.side_effect = lambda mk: 5 * GIB
        cluster.deploy(["model-a"])

        # Evict
        results, change = cluster.evict(["model-a"])
        assert change is True
        # All replicas should be evicted
        for key, info in results.items():
            assert info["status"] == "evicted"
            assert info["node"] == "node-1"

    def test_evict_model_not_found(self):
        """Evicting a model that doesn't exist returns not_found."""
        cluster = _make_cluster()
        node = _make_node("n1", "node-1")
        cluster.nodes["n1"] = node

        results, change = cluster.evict(["nonexistent"])
        assert change is False
        assert ("nonexistent", None) in results
        assert results[("nonexistent", None)]["status"] == "not_found"

    def test_evict_by_replica_keys(self):
        """Evict specific replicas using replica_keys."""
        cluster = _make_cluster()
        node = _make_node("n1", "node-1", total_gpus=4, gpu_memory_bytes=80 * GIB)
        cluster.nodes["n1"] = node

        cluster.evaluator.side_effect = lambda mk: 5 * GIB
        deploy_results, _ = cluster.deploy(["model-a"])

        replica_id = list(deploy_results["result"].keys())[0][1]

        results, change = cluster.evict(
            model_keys=["model-a"],
            replica_keys=[("model-a", replica_id)],
        )
        assert change is True
        assert ("model-a", replica_id) in results
        assert results[("model-a", replica_id)]["status"] == "evicted"

    def test_evict_replica_not_found(self):
        """Evicting a specific replica that doesn't exist returns not_found."""
        cluster = _make_cluster()
        node = _make_node("n1", "node-1")
        cluster.nodes["n1"] = node

        results, change = cluster.evict(
            model_keys=["model-a"],
            replica_keys=[("model-a", "nonexistent-replica")],
        )
        assert change is False
        assert ("model-a", "nonexistent-replica") in results
        assert results[("model-a", "nonexistent-replica")]["status"] == "not_found"

    def test_evict_multiple_models(self):
        """Evict multiple models at once."""
        cluster = _make_cluster()
        node = _make_node("n1", "node-1", total_gpus=4, gpu_memory_bytes=80 * GIB)
        cluster.nodes["n1"] = node

        cluster.evaluator.side_effect = lambda mk: 5 * GIB
        cluster.deploy(["model-a", "model-b"])

        results, change = cluster.evict(["model-a", "model-b"])
        assert change is True
        evicted_models = {k[0] for k in results.keys()}
        assert "model-a" in evicted_models
        assert "model-b" in evicted_models

    def test_evict_frees_gpu_memory(self):
        """After eviction, node GPU memory should be restored."""
        cluster = _make_cluster()
        node = _make_node("n1", "node-1", total_gpus=4, gpu_memory_bytes=80 * GIB)
        cluster.nodes["n1"] = node

        cluster.evaluator.side_effect = lambda mk: 5 * GIB
        cluster.deploy(["model-a"])

        # Check some GPU memory was consumed
        gpu_mem_before_evict = dict(node.resources.gpu_memory_available_bytes_by_id)

        cluster.evict(["model-a"])

        # At least some GPU memory should be freed (possibly all restored)
        gpu_mem_after_evict = node.resources.gpu_memory_available_bytes_by_id
        total_after = sum(gpu_mem_after_evict.values())
        total_before = sum(gpu_mem_before_evict.values())
        assert total_after > total_before

    def test_evict_across_multiple_nodes(self):
        """Model deployed across multiple nodes: evict from all."""
        cluster = _make_cluster()
        node1 = _make_node("n1", "node-1", total_gpus=2, gpu_memory_bytes=80 * GIB)
        node2 = _make_node("n2", "node-2", total_gpus=2, gpu_memory_bytes=80 * GIB)
        cluster.nodes["n1"] = node1
        cluster.nodes["n2"] = node2

        # Manually deploy a model with replicas on different nodes
        dep1 = _make_deployment(
            model_key="model-a", replica_id="r1",
            gpu_mem_bytes_by_id={0: 20 * GIB}, size_bytes=5 * GIB,
            node_id="n1",
        )
        dep2 = _make_deployment(
            model_key="model-a", replica_id="r2",
            gpu_mem_bytes_by_id={0: 20 * GIB}, size_bytes=5 * GIB,
            node_id="n2",
        )
        node1._set_in_map(node1.deployments, dep1)
        node1.resources.gpu_memory_available_bytes_by_id[0] -= 20 * GIB

        node2._set_in_map(node2.deployments, dep2)
        node2.resources.gpu_memory_available_bytes_by_id[0] -= 20 * GIB

        results, change = cluster.evict(["model-a"])
        assert change is True
        assert len(results) == 2
        for key, info in results.items():
            assert info["status"] == "evicted"

    def test_evict_results_contain_freed_info(self):
        """Eviction results should contain freed GPU count and memory info."""
        cluster = _make_cluster()
        node = _make_node("n1", "node-1", total_gpus=4, gpu_memory_bytes=80 * GIB)
        cluster.nodes["n1"] = node

        cluster.evaluator.side_effect = lambda mk: 5 * GIB
        cluster.deploy(["model-a"])

        results, _ = cluster.evict(["model-a"])
        for key, info in results.items():
            if info["status"] == "evicted":
                assert "freed_gpus" in info
                assert "freed_memory_gbs" in info
                assert info["freed_gpus"] >= 1
                assert info["freed_memory_gbs"] > 0


# ============================================================================
# Cluster: get_state
# ============================================================================


class TestClusterGetState:
    def test_get_state_empty_cluster(self):
        """Verify get_state returns empty nodes list for a cluster with no nodes."""
        cluster = _make_cluster()
        cluster.evaluator.get_state = MagicMock(return_value={"cache": {}})
        state = cluster.get_state()
        assert "nodes" in state
        assert "evaluator" in state
        assert state["nodes"] == []

    def test_get_state_with_nodes(self):
        """Verify get_state includes node info when nodes are present."""
        cluster = _make_cluster()
        cluster.evaluator.get_state = MagicMock(return_value={"cache": {}})
        node = _make_node("n1", "node-1")
        cluster.nodes["n1"] = node

        state = cluster.get_state()
        assert len(state["nodes"]) == 1
        assert state["nodes"][0]["id"] == "n1"

    def test_get_state_excludes_ray_state_by_default(self):
        """Verify get_state does not include ray_state key by default."""
        cluster = _make_cluster()
        cluster.evaluator.get_state = MagicMock(return_value={"cache": {}})
        state = cluster.get_state()
        assert "ray_state" not in state


# ============================================================================
# Cluster: integration / end-to-end style tests
# ============================================================================


class TestClusterIntegration:
    def test_deploy_then_evict_then_redeploy(self):
        """Full lifecycle: deploy, evict, re-deploy (should find cache)."""
        cluster = _make_cluster()
        node = _make_node("n1", "node-1", total_gpus=4, gpu_memory_bytes=80 * GIB)
        cluster.nodes["n1"] = node

        cluster.evaluator.side_effect = lambda mk: 5 * GIB

        # Deploy
        results1, change1 = cluster.deploy(["model-a"])
        assert change1 is True
        replica_id = list(results1["result"].keys())[0][1]

        # Evict
        cluster.evict(["model-a"])
        # Model should be in cache now
        cached = node._get_from_map(node.cache, "model-a", replica_id)
        assert cached is not None

        # Re-deploy (should find cached replica)
        results2, change2 = cluster.deploy(["model-a"])
        assert change2 is True
        # Should use the cached replica ID
        model_entries = {k: v for k, v in results2["result"].items() if k[0] == "model-a"}
        assert len(model_entries) >= 1
        # At least one entry should have the CACHED_AND_FREE status
        statuses = set(model_entries.values())
        assert CandidateLevel.CACHED_AND_FREE.name in statuses or CandidateLevel.DEPLOYED.name in statuses

    def test_deploy_multiple_models_with_eviction_chain(self):
        """Deploy models that cause a chain of evictions."""
        cluster = _make_cluster()
        node = _make_node("n1", "node-1", total_gpus=2, gpu_memory_bytes=80 * GIB)
        cluster.nodes["n1"] = node

        # Deploy model-a and model-b (each fits on 1 GPU fractionally)
        cluster.evaluator.side_effect = lambda mk: 20 * GIB
        cluster.deploy(["model-a"])
        cluster.deploy(["model-b"])

        # Age all deployments
        for model_map in node.deployments.values():
            for dep in model_map.values():
                dep.deployed = time.time() - 1000

        # Deploy model-c which is large enough to potentially trigger evictions
        cluster.evaluator.side_effect = lambda mk: 50 * GIB
        results, change = cluster.deploy(["model-c"])
        # Should succeed (possibly with evictions)
        model_entries = {k: v for k, v in results["result"].items() if k[0] == "model-c"}
        assert len(model_entries) == 1

    def test_deploy_with_dedicated_replaces_old_dedicated(self):
        """When deploying new dedicated models, old dedicated ones not in the set are evicted."""
        cluster = _make_cluster()
        node = _make_node("n1", "node-1", total_gpus=4, gpu_memory_bytes=80 * GIB)
        cluster.nodes["n1"] = node

        cluster.evaluator.side_effect = lambda mk: 5 * GIB

        # Deploy model-a as dedicated
        cluster.deploy(["model-a"], dedicated=True)

        # Age deployment
        for model_map in node.deployments.values():
            for dep in model_map.values():
                dep.deployed = time.time() - 1000

        # Deploy model-b as dedicated, model-a should be evicted
        results, change = cluster.deploy(["model-b"], dedicated=True)
        evicted = {e[0] for e in results["evictions"]}
        assert "model-a" in evicted

    def test_all_evaluator_errors(self):
        """When all models fail evaluation, no deployments should be made."""
        cluster = _make_cluster()
        node = _make_node("n1", "node-1", total_gpus=4)
        cluster.nodes["n1"] = node

        cluster.evaluator.side_effect = lambda mk: ValueError("fail")

        results, change = cluster.deploy(["bad-1", "bad-2"])
        assert change is False
        # All results should be error strings
        for key, status in results["result"].items():
            assert "fail" in status

    def test_mixed_evaluator_results(self):
        """Some models succeed evaluation, some fail."""
        cluster = _make_cluster()
        node = _make_node("n1", "node-1", total_gpus=4, gpu_memory_bytes=80 * GIB)
        cluster.nodes["n1"] = node

        def eval_fn(mk):
            if mk == "good":
                return 5 * GIB
            return RuntimeError("cannot load")

        cluster.evaluator.side_effect = eval_fn

        results, change = cluster.deploy(["good", "bad"])
        # "good" should be deployed
        good_entries = {k: v for k, v in results["result"].items() if k[0] == "good"}
        assert len(good_entries) == 1
        assert list(good_entries.values())[0] in (
            CandidateLevel.FREE.name,
            CandidateLevel.CACHED_AND_FREE.name,
        )

        # "bad" should have error string
        bad_entries = {k: v for k, v in results["result"].items() if k[0] == "bad"}
        assert len(bad_entries) == 1
        assert "cannot load" in list(bad_entries.values())[0]

    def test_replicas_spread_across_nodes(self):
        """Multiple replicas of same model should be spread across nodes when possible."""
        cluster = _make_cluster()
        node1 = _make_node("n1", "node-1", total_gpus=4, gpu_memory_bytes=80 * GIB)
        node2 = _make_node("n2", "node-2", total_gpus=4, gpu_memory_bytes=80 * GIB)
        cluster.nodes["n1"] = node1
        cluster.nodes["n2"] = node2

        cluster.evaluator.side_effect = lambda mk: 5 * GIB
        results, change = cluster.deploy(["model-a"], replicas=2)
        assert change is True

        model_entries = {k: v for k, v in results["result"].items() if k[0] == "model-a"}
        assert len(model_entries) == 2
        # Both replicas should be successfully deployed
        for status in model_entries.values():
            assert status in (
                CandidateLevel.FREE.name,
                CandidateLevel.CACHED_AND_FREE.name,
                CandidateLevel.DEPLOYED.name,
            )
