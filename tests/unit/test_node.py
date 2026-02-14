"""Comprehensive unit tests for Node, Resources, Candidate, and CandidateLevel classes.

Tests cover:
- Resources: gpus_required, assign_full_gpus, assign_memory, assign, _update_available_gpus
- Node: evaluate, deploy, evict, evictions_for_gpu_count, evictions_for_fractional_gpu_memory, _is_evictable
- CandidateLevel ordering
- Candidate construction
"""

import time
from unittest.mock import patch

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


GIB = 1024**3


# ============================================================================
# CandidateLevel tests
# ============================================================================


class TestCandidateLevel:
    def test_ordering(self):
        """CandidateLevel values are ordered from best (DEPLOYED=0) to worst (CANT_ACCOMMODATE=5)."""
        assert CandidateLevel.DEPLOYED < CandidateLevel.CACHED_AND_FREE
        assert CandidateLevel.CACHED_AND_FREE < CandidateLevel.FREE
        assert CandidateLevel.FREE < CandidateLevel.CACHED_AND_FULL
        assert CandidateLevel.CACHED_AND_FULL < CandidateLevel.FULL
        assert CandidateLevel.FULL < CandidateLevel.CANT_ACCOMMODATE

    def test_values(self):
        """Verify CandidateLevel integer values match expected 0-5 assignments."""
        assert CandidateLevel.DEPLOYED == 0
        assert CandidateLevel.CACHED_AND_FREE == 1
        assert CandidateLevel.FREE == 2
        assert CandidateLevel.CACHED_AND_FULL == 3
        assert CandidateLevel.FULL == 4
        assert CandidateLevel.CANT_ACCOMMODATE == 5

    def test_is_intenum(self):
        """CandidateLevel values can be used in integer comparisons."""
        assert CandidateLevel.DEPLOYED + 1 == CandidateLevel.CACHED_AND_FREE
        assert int(CandidateLevel.CANT_ACCOMMODATE) == 5


# ============================================================================
# Candidate tests
# ============================================================================


class TestCandidate:
    def test_default_construction(self):
        """Verify a Candidate with only candidate_level has correct defaults."""
        c = Candidate(candidate_level=CandidateLevel.FREE)
        assert c.candidate_level == CandidateLevel.FREE
        assert c.gpus_required is None
        assert c.gpu_ids == []
        assert c.gpu_mem_bytes_by_id == {}
        assert c.gpu_memory_required_bytes is None
        assert c.evictions == []

    def test_full_construction(self):
        """Verify a fully-constructed Candidate stores all provided fields."""
        evictions = [("model-a", "replica-1")]
        c = Candidate(
            candidate_level=CandidateLevel.FULL,
            gpus_required=2,
            gpu_ids=[0, 1],
            gpu_mem_bytes_by_id={0: 80 * GIB, 1: 80 * GIB},
            gpu_memory_required_bytes=40 * GIB,
            evictions=evictions,
        )
        assert c.gpus_required == 2
        assert c.gpu_ids == [0, 1]
        assert len(c.gpu_mem_bytes_by_id) == 2
        assert c.evictions == evictions


# ============================================================================
# Resources tests
# ============================================================================


class TestResourcesGpusRequired:
    def test_basic_calculation(self, make_resources):
        """gpus_required = model_size // gpu_memory + 1"""
        r = make_resources(gpu_memory_bytes=80 * GIB)
        # A 10 GiB model needs 1 GPU (10 // 80 + 1 = 1)
        assert r.gpus_required(10 * GIB) == 1

    def test_model_exactly_one_gpu(self, make_resources):
        """Verify a model exactly equal to GPU memory requires 2 GPUs."""
        r = make_resources(gpu_memory_bytes=80 * GIB)
        # 80 GiB model: 80 // 80 + 1 = 2
        assert r.gpus_required(80 * GIB) == 2

    def test_large_model(self, make_resources):
        """Verify a 200 GiB model needs 3 GPUs at 80 GiB each."""
        r = make_resources(gpu_memory_bytes=80 * GIB)
        # 200 GiB model: 200 // 80 + 1 = 3
        assert r.gpus_required(200 * GIB) == 3

    def test_small_model(self, make_resources):
        """Verify a 1-byte model still requires exactly 1 GPU."""
        r = make_resources(gpu_memory_bytes=80 * GIB)
        # 1 byte model: 1 // (80 GiB) + 1 = 1
        assert r.gpus_required(1) == 1

    def test_zero_gpu_memory_raises(self, make_resources):
        """Verify gpus_required raises ValueError when GPU memory is 0."""
        r = make_resources(gpu_memory_bytes=0)
        with pytest.raises(ValueError, match="GPU memory bytes is 0"):
            r.gpus_required(100)

    def test_model_just_over_one_gpu(self, make_resources):
        """Verify an 81 GiB model just over one GPU boundary requires 2 GPUs."""
        r = make_resources(gpu_memory_bytes=80 * GIB)
        # 81 GiB model: 81 // 80 + 1 = 2
        assert r.gpus_required(81 * GIB) == 2


class TestResourcesAssignFullGpus:
    def test_assign_one_gpu(self, make_resources):
        """Verify assigning 1 full GPU removes it from available and zeroes its memory."""
        r = make_resources(total_gpus=4, gpu_memory_bytes=80 * GIB)
        result = r.assign_full_gpus(1)
        assert len(result) == 1
        assert 0 in result
        assert result[0] == 80 * GIB
        # GPU 0 should be removed from available
        assert 0 not in r.available_gpus
        assert len(r.available_gpus) == 3
        # GPU 0 memory should be zeroed out
        assert r.gpu_memory_available_bytes_by_id[0] == 0

    def test_assign_all_gpus(self, make_resources):
        """Verify assigning all GPUs leaves none available."""
        r = make_resources(total_gpus=2, gpu_memory_bytes=80 * GIB)
        result = r.assign_full_gpus(2)
        assert len(result) == 2
        assert r.available_gpus == []
        for gpu_id in result:
            assert r.gpu_memory_available_bytes_by_id[gpu_id] == 0

    def test_assign_too_many_raises(self, make_resources):
        """Verify requesting more GPUs than available raises ValueError."""
        r = make_resources(total_gpus=2, gpu_memory_bytes=80 * GIB)
        with pytest.raises(ValueError, match="Not enough GPUs available"):
            r.assign_full_gpus(3)

    def test_assign_zero_gpus(self, make_resources):
        """Verify assigning 0 GPUs returns empty dict and keeps all available."""
        r = make_resources(total_gpus=4, gpu_memory_bytes=80 * GIB)
        result = r.assign_full_gpus(0)
        assert result == {}
        assert len(r.available_gpus) == 4

    def test_assign_updates_memory_map(self, make_resources):
        """Verify assigned GPUs have memory zeroed while others remain full."""
        r = make_resources(total_gpus=4, gpu_memory_bytes=80 * GIB)
        r.assign_full_gpus(2)
        # GPUs 0 and 1 zeroed
        assert r.gpu_memory_available_bytes_by_id[0] == 0
        assert r.gpu_memory_available_bytes_by_id[1] == 0
        # GPUs 2 and 3 still full
        assert r.gpu_memory_available_bytes_by_id[2] == 80 * GIB
        assert r.gpu_memory_available_bytes_by_id[3] == 80 * GIB


class TestResourcesAssignMemory:
    def test_assign_fractional_memory(self, make_resources):
        """Verify assigning fractional memory reduces GPU available memory correctly."""
        r = make_resources(total_gpus=2, gpu_memory_bytes=80 * GIB)
        result = r.assign_memory(20 * GIB)
        assert len(result) == 1
        gpu_id = list(result.keys())[0]
        assert result[gpu_id] == 20 * GIB
        assert r.gpu_memory_available_bytes_by_id[gpu_id] == 60 * GIB

    def test_assign_memory_with_specific_gpu(self, make_resources):
        """Verify assigning memory to a specific GPU ID targets that GPU."""
        r = make_resources(total_gpus=2, gpu_memory_bytes=80 * GIB)
        result = r.assign_memory(30 * GIB, gpu_id=1)
        assert result == {1: 30 * GIB}
        assert r.gpu_memory_available_bytes_by_id[1] == 50 * GIB

    def test_assign_memory_picks_gpu_with_most_memory(self, make_resources):
        """When no gpu_id specified, picks the GPU with most available memory."""
        r = make_resources(
            total_gpus=2,
            gpu_memory_bytes=80 * GIB,
            gpu_memory_available_bytes_by_id={0: 30 * GIB, 1: 60 * GIB},
        )
        result = r.assign_memory(20 * GIB)
        # Should pick GPU 1 (60 GiB available > 30 GiB)
        assert 1 in result

    def test_assign_memory_no_eligible_gpu_raises(self, make_resources):
        """Verify ValueError when no GPU has enough available memory."""
        r = make_resources(
            total_gpus=2,
            gpu_memory_bytes=80 * GIB,
            gpu_memory_available_bytes_by_id={0: 10 * GIB, 1: 10 * GIB},
        )
        with pytest.raises(ValueError, match="No GPU has enough available memory"):
            r.assign_memory(20 * GIB)

    def test_assign_memory_specific_gpu_insufficient_raises(self, make_resources):
        """Verify ValueError when the specified GPU has insufficient memory."""
        r = make_resources(
            total_gpus=2,
            gpu_memory_bytes=80 * GIB,
            gpu_memory_available_bytes_by_id={0: 10 * GIB, 1: 80 * GIB},
        )
        with pytest.raises(ValueError, match="GPU 0 does not have enough available memory"):
            r.assign_memory(20 * GIB, gpu_id=0)

    def test_assign_memory_removes_from_available_when_below_threshold(self, make_resources):
        """GPU is removed from available_gpus when memory drops below min_available_gpu_fraction."""
        r = make_resources(total_gpus=1, gpu_memory_bytes=100 * GIB)
        # Default min_available_gpu_fraction=0.3, so threshold is 30 GiB
        # Assign 75 GiB, leaving 25 GiB (< 30 GiB threshold)
        r.assign_memory(75 * GIB)
        assert 0 not in r.available_gpus

    def test_assign_memory_keeps_in_available_when_above_threshold(self, make_resources):
        """GPU stays in available_gpus when memory stays above min_available_gpu_fraction."""
        r = make_resources(total_gpus=1, gpu_memory_bytes=100 * GIB)
        # Assign 60 GiB, leaving 40 GiB (> 30 GiB threshold)
        r.assign_memory(60 * GIB)
        assert 0 in r.available_gpus


class TestResourcesAssign:
    def test_assign_returns_gpu_ids(self, make_resources):
        """Verify assign() returns the correct number of integer GPU IDs."""
        r = make_resources(total_gpus=4, gpu_memory_bytes=80 * GIB)
        gpu_ids = r.assign(2)
        assert len(gpu_ids) == 2
        assert all(isinstance(gid, int) for gid in gpu_ids)

    def test_assign_delegates_to_assign_full_gpus(self, make_resources):
        """Verify assign() delegates to assign_full_gpus and zeroes assigned GPU memory."""
        r = make_resources(total_gpus=4, gpu_memory_bytes=80 * GIB)
        gpu_ids = r.assign(2)
        # After assigning 2 full GPUs, their memory should be zeroed
        for gid in gpu_ids:
            assert r.gpu_memory_available_bytes_by_id[gid] == 0
        assert len(r.available_gpus) == 2


class TestResourcesUpdateAvailableGpus:
    def test_gpu_added_back_when_memory_restored(self, make_resources):
        """If a GPU's available memory goes back above the threshold, it should be re-added."""
        r = make_resources(total_gpus=1, gpu_memory_bytes=100 * GIB)
        # Remove GPU from available list
        r.gpu_memory_available_bytes_by_id[0] = 20 * GIB
        r._update_available_gpus(0)
        assert 0 not in r.available_gpus

        # Restore memory
        r.gpu_memory_available_bytes_by_id[0] = 50 * GIB
        r._update_available_gpus(0)
        assert 0 in r.available_gpus

    def test_no_duplicate_in_available(self, make_resources):
        """Calling _update_available_gpus when GPU is already in list should not duplicate it."""
        r = make_resources(total_gpus=1, gpu_memory_bytes=100 * GIB)
        r._update_available_gpus(0)
        r._update_available_gpus(0)
        assert r.available_gpus.count(0) == 1


class TestResourcesStr:
    def test_str_representation(self, make_resources):
        """Verify Resources str includes total_gpus and gpu_type."""
        r = make_resources(total_gpus=2, gpu_type="A100")
        s = str(r)
        assert "total_gpus=2" in s
        assert "gpu_type=A100" in s


# ============================================================================
# Node: _is_evictable tests
# ============================================================================


class TestNodeIsEvictable:
    def test_dedicated_deployment_not_evictable(self, make_node, make_deployment):
        """Verify a dedicated deployment cannot be evicted by a non-dedicated request."""
        node = make_node()
        dep = make_deployment(dedicated=True)
        assert node._is_evictable(dep, dedicated=False) is False

    def test_non_dedicated_deployment_is_evictable(self, make_node, make_deployment):
        """Verify a non-dedicated deployment is evictable by default."""
        node = make_node()
        dep = make_deployment(dedicated=False)
        assert node._is_evictable(dep, dedicated=False) is True

    def test_minimum_deployment_time_not_expired(self, make_node, make_deployment):
        """Verify a recently-deployed model is not evictable before minimum time."""
        node = make_node(minimum_deployment_time_seconds=60)
        dep = make_deployment(dedicated=False)
        dep.deployed = time.time()  # just deployed
        # Not dedicated request, deployment too young
        assert node._is_evictable(dep, dedicated=False) is False

    def test_minimum_deployment_time_expired(self, make_node, make_deployment):
        """Verify a model deployed past minimum time is evictable."""
        node = make_node(minimum_deployment_time_seconds=60)
        dep = make_deployment(dedicated=False)
        dep.deployed = time.time() - 120  # deployed 2 minutes ago
        assert node._is_evictable(dep, dedicated=False) is True

    def test_dedicated_request_bypasses_minimum_time(self, make_node, make_deployment):
        """When dedicated=True (the requesting deployment is dedicated), minimum time is ignored."""
        node = make_node(minimum_deployment_time_seconds=60)
        dep = make_deployment(dedicated=False)
        dep.deployed = time.time()  # just deployed
        # dedicated=True in the request context bypasses the time check
        assert node._is_evictable(dep, dedicated=True) is True

    def test_no_minimum_time_set(self, make_node, make_deployment):
        """Verify a deployment is immediately evictable when no minimum time is set."""
        node = make_node(minimum_deployment_time_seconds=None)
        dep = make_deployment(dedicated=False)
        dep.deployed = time.time()
        assert node._is_evictable(dep, dedicated=False) is True


# ============================================================================
# Node: evictions_for_gpu_count tests
# ============================================================================


class TestNodeEvictionsForGpuCount:
    def test_enough_available_gpus_no_evictions(self, make_node):
        """Verify no evictions needed when enough GPUs are already available."""
        node = make_node(total_gpus=4)
        result = node.evictions_for_gpu_count(2)
        # 4 available, need 2 -> 0 evictions needed
        assert result == []

    def test_evicts_smallest_deployments_first(self, make_node, make_deployment):
        """Verify evictions start with smallest deployments to free GPU count."""
        node = make_node(total_gpus=4, gpu_memory_bytes=80 * GIB)
        # Use all GPUs
        node.resources.available_gpus = []
        for i in range(4):
            node.resources.gpu_memory_available_bytes_by_id[i] = 0

        # Deploy model-a on GPU 0 and model-b on GPUs 1,2
        dep_a = make_deployment(
            model_key="model-a", replica_id="r1",
            gpu_mem_bytes_by_id={0: 80 * GIB}, size_bytes=10 * GIB
        )
        dep_b = make_deployment(
            model_key="model-b", replica_id="r2",
            gpu_mem_bytes_by_id={1: 80 * GIB, 2: 80 * GIB}, size_bytes=20 * GIB
        )
        node._set_in_map(node.deployments, dep_a)
        node._set_in_map(node.deployments, dep_b)

        # Need 2 GPUs. Should evict model-a (1 GPU) first, then model-b (2 GPUs)
        result = node.evictions_for_gpu_count(2)
        # model-a (1 GPU) evicted first gives 1 GPU, need 1 more -> model-b (2 GPUs) evicted
        assert len(result) >= 1
        assert ("model-a", "r1") in result

    def test_returns_empty_if_cant_free_enough(self, make_node, make_deployment):
        """If not enough evictable deployments to free required GPUs, return empty list."""
        node = make_node(total_gpus=4, gpu_memory_bytes=80 * GIB)
        node.resources.available_gpus = []

        # Deploy one dedicated (non-evictable) deployment using all 4 GPUs
        dep = make_deployment(
            model_key="model-a", replica_id="r1",
            gpu_mem_bytes_by_id={0: 80 * GIB, 1: 80 * GIB, 2: 80 * GIB, 3: 80 * GIB},
            dedicated=True, size_bytes=40 * GIB,
        )
        node._set_in_map(node.deployments, dep)

        result = node.evictions_for_gpu_count(2)
        assert result == []


# ============================================================================
# Node: evictions_for_fractional_gpu_memory tests
# ============================================================================


class TestNodeEvictionsForFractionalGpuMemory:
    def test_gpu_has_enough_memory_no_evictions(self, make_node):
        """Verify no evictions needed when a GPU has enough free memory."""
        node = make_node(total_gpus=2, gpu_memory_bytes=80 * GIB)
        result = node.evictions_for_fractional_gpu_memory(20 * GIB)
        assert result is not None
        gpu_id, evictions = result
        assert evictions == []

    def test_needs_eviction_to_free_memory(self, make_node, make_deployment):
        """Verify evictions are proposed when GPU lacks sufficient free memory."""
        node = make_node(total_gpus=1, gpu_memory_bytes=80 * GIB)
        # GPU 0 has only 10 GiB free
        node.resources.gpu_memory_available_bytes_by_id[0] = 10 * GIB
        node.resources.available_gpus = []

        dep = make_deployment(
            model_key="model-a", replica_id="r1",
            gpu_mem_bytes_by_id={0: 30 * GIB}, size_bytes=30 * GIB,
        )
        dep.deployed = time.time() - 1000
        node._set_in_map(node.deployments, dep)

        result = node.evictions_for_fractional_gpu_memory(30 * GIB)
        assert result is not None
        gpu_id, evictions = result
        assert gpu_id == 0
        assert ("model-a", "r1") in evictions

    def test_returns_none_if_cannot_free(self, make_node, make_deployment):
        """Returns None when no combination of evictions can free enough memory."""
        node = make_node(total_gpus=1, gpu_memory_bytes=80 * GIB)
        node.resources.gpu_memory_available_bytes_by_id[0] = 0
        node.resources.available_gpus = []

        # Dedicated deployment is not evictable
        dep = make_deployment(
            model_key="model-a", replica_id="r1",
            gpu_mem_bytes_by_id={0: 80 * GIB}, size_bytes=80 * GIB, dedicated=True,
        )
        node._set_in_map(node.deployments, dep)

        result = node.evictions_for_fractional_gpu_memory(40 * GIB)
        assert result is None

    def test_picks_gpu_with_fewest_evictions(self, make_node, make_deployment):
        """When multiple GPUs can satisfy the request, prefers the one with fewer evictions."""
        node = make_node(total_gpus=2, gpu_memory_bytes=80 * GIB)
        # GPU 0: 5 GiB free, one model using 40 GiB
        node.resources.gpu_memory_available_bytes_by_id[0] = 5 * GIB
        # GPU 1: 70 GiB free (enough without evictions)
        node.resources.gpu_memory_available_bytes_by_id[1] = 70 * GIB
        node.resources.available_gpus = [1]

        dep = make_deployment(
            model_key="model-a", replica_id="r1",
            gpu_mem_bytes_by_id={0: 40 * GIB}, size_bytes=40 * GIB,
        )
        dep.deployed = time.time() - 1000
        node._set_in_map(node.deployments, dep)

        # Need 30 GiB. GPU 1 has 70 GiB free (0 evictions), GPU 0 would need eviction
        result = node.evictions_for_fractional_gpu_memory(30 * GIB)
        assert result is not None
        gpu_id, evictions = result
        # Should pick GPU 1 (no evictions) since it has enough
        assert evictions == []


# ============================================================================
# Node: evaluate tests
# ============================================================================


class TestNodeEvaluate:
    def test_already_deployed_returns_deployed(self, make_node, make_deployment):
        """Verify evaluating an already-deployed model returns DEPLOYED level."""
        node = make_node(total_gpus=4, gpu_memory_bytes=80 * GIB)
        dep = make_deployment(model_key="model-a", replica_id="r1")
        node._set_in_map(node.deployments, dep)

        candidate = node.evaluate("model-a", "r1", 10 * GIB)
        assert candidate.candidate_level == CandidateLevel.DEPLOYED

    def test_already_deployed_sets_dedicated(self, make_node, make_deployment):
        """Verify evaluating a deployed model with dedicated=True upgrades it."""
        node = make_node(total_gpus=4, gpu_memory_bytes=80 * GIB)
        dep = make_deployment(model_key="model-a", replica_id="r1", dedicated=False)
        node._set_in_map(node.deployments, dep)

        node.evaluate("model-a", "r1", 10 * GIB, dedicated=True)
        assert dep.dedicated is True

    def test_small_model_free_gpu_fractional(self, make_node):
        """A small model should use fractional allocation on a single GPU."""
        node = make_node(total_gpus=4, gpu_memory_bytes=80 * GIB)
        # 5 GiB model: required_bytes = 5*3 = 15 GiB < 80*0.8 = 64 GiB threshold
        candidate = node.evaluate("model-a", "r1", 5 * GIB)
        assert candidate.candidate_level == CandidateLevel.FREE
        assert candidate.gpus_required == 1
        assert len(candidate.gpu_ids) == 1
        assert candidate.gpu_memory_required_bytes == int(5 * GIB * 3.0)

    def test_large_single_gpu_model_uses_full_gpu_memory(self, make_node):
        """If model * gpu_fraction_factor >= fraction_threshold, use full GPU memory."""
        node = make_node(total_gpus=4, gpu_memory_bytes=80 * GIB)
        # 25 GiB model: required_bytes = 25*3 = 75 GiB >= 80*0.8=64 GiB
        candidate = node.evaluate("model-a", "r1", 25 * GIB)
        assert candidate.candidate_level == CandidateLevel.FREE
        assert candidate.gpus_required == 1
        assert candidate.gpu_memory_required_bytes == int(80 * GIB)

    def test_cached_model_free_gpu(self, make_node, make_deployment):
        """A cached model with free GPU space should get CACHED_AND_FREE."""
        node = make_node(total_gpus=4, gpu_memory_bytes=80 * GIB)
        cached = make_deployment(
            model_key="model-a", replica_id="r1",
            deployment_level=DeploymentLevel.WARM,
            gpu_mem_bytes_by_id={}, size_bytes=5 * GIB,
        )
        node._set_in_map(node.cache, cached)

        candidate = node.evaluate("model-a", "r1", 5 * GIB)
        assert candidate.candidate_level == CandidateLevel.CACHED_AND_FREE

    def test_multi_gpu_model_free(self, make_node):
        """Model that doesn't fit on one GPU should get multi-GPU FREE."""
        node = make_node(total_gpus=4, gpu_memory_bytes=80 * GIB)
        # 100 GiB model: doesn't fit single GPU.
        # gpus_required = 100 // 80 + 1 = 2
        candidate = node.evaluate("model-a", "r1", 100 * GIB)
        assert candidate.candidate_level == CandidateLevel.FREE
        assert candidate.gpus_required == 2
        assert len(candidate.gpu_ids) == 2

    def test_multi_gpu_model_needs_eviction(self, make_node, make_deployment):
        """Multi-GPU model when not enough free GPUs but evictable deployments exist."""
        node = make_node(total_gpus=4, gpu_memory_bytes=80 * GIB)
        # Use up 3 GPUs
        for i in range(3):
            dep = make_deployment(
                model_key=f"model-{i}", replica_id=f"r{i}",
                gpu_mem_bytes_by_id={i: 80 * GIB}, size_bytes=10 * GIB,
            )
            dep.deployed = time.time() - 1000
            node._set_in_map(node.deployments, dep)

        node.resources.available_gpus = [3]
        for i in range(3):
            node.resources.gpu_memory_available_bytes_by_id[i] = 0

        # 100 GiB model needs 2 GPUs, only 1 available
        candidate = node.evaluate("model-new", "r-new", 100 * GIB)
        assert candidate.candidate_level == CandidateLevel.FULL
        assert len(candidate.evictions) >= 1

    def test_model_too_large_cant_accommodate(self, make_node):
        """Model requires more GPUs than total on node."""
        node = make_node(total_gpus=2, gpu_memory_bytes=80 * GIB)
        # 200 GiB model: gpus_required = 200 // 80 + 1 = 3 > total 2
        candidate = node.evaluate("model-a", "r1", 200 * GIB)
        assert candidate.candidate_level == CandidateLevel.CANT_ACCOMMODATE

    def test_multi_gpu_needs_eviction_but_all_dedicated(self, make_node, make_deployment):
        """When all deployments are dedicated, evictions return empty and result is CANT_ACCOMMODATE."""
        node = make_node(total_gpus=2, gpu_memory_bytes=80 * GIB)
        # Fill both GPUs with dedicated deployments
        for i in range(2):
            dep = make_deployment(
                model_key=f"model-{i}", replica_id=f"r{i}",
                gpu_mem_bytes_by_id={i: 80 * GIB}, size_bytes=80 * GIB,
                dedicated=True,
            )
            node._set_in_map(node.deployments, dep)
        node.resources.available_gpus = []
        node.resources.gpu_memory_available_bytes_by_id = {0: 0, 1: 0}

        # Need 2 GPUs for a 100 GiB model, but no evictable deployments
        candidate = node.evaluate("model-new", "r-new", 100 * GIB)
        assert candidate.candidate_level == CandidateLevel.CANT_ACCOMMODATE

    def test_cached_multi_gpu_full(self, make_node, make_deployment):
        """Cached model that needs evictions for multi-GPU should be CACHED_AND_FULL."""
        node = make_node(total_gpus=4, gpu_memory_bytes=80 * GIB)
        # Cache model-a
        cached = make_deployment(
            model_key="model-a", replica_id="r1",
            deployment_level=DeploymentLevel.WARM,
            gpu_mem_bytes_by_id={}, size_bytes=100 * GIB,
        )
        node._set_in_map(node.cache, cached)

        # Fill all GPUs
        for i in range(4):
            dep = make_deployment(
                model_key=f"model-{i}", replica_id=f"r{i}",
                gpu_mem_bytes_by_id={i: 80 * GIB}, size_bytes=10 * GIB,
            )
            dep.deployed = time.time() - 1000
            node._set_in_map(node.deployments, dep)
        node.resources.available_gpus = []
        for i in range(4):
            node.resources.gpu_memory_available_bytes_by_id[i] = 0

        # model-a needs 2 GPUs (100 // 80 + 1 = 2), no free GPUs but evictable
        candidate = node.evaluate("model-a", "r1", 100 * GIB)
        assert candidate.candidate_level == CandidateLevel.CACHED_AND_FULL
        assert len(candidate.evictions) >= 1

    def test_fractional_with_eviction_full(self, make_node, make_deployment):
        """Fractional single-GPU model that needs an eviction on that GPU should be FULL."""
        node = make_node(total_gpus=1, gpu_memory_bytes=80 * GIB)
        # GPU 0 has only 5 GiB free
        node.resources.gpu_memory_available_bytes_by_id[0] = 5 * GIB
        node.resources.available_gpus = []

        dep = make_deployment(
            model_key="model-existing", replica_id="r1",
            gpu_mem_bytes_by_id={0: 40 * GIB}, size_bytes=10 * GIB,
        )
        dep.deployed = time.time() - 1000
        node._set_in_map(node.deployments, dep)

        # 5 GiB model: required_bytes = 15 GiB, only 5 GiB free -> needs eviction
        candidate = node.evaluate("model-new", "r-new", 5 * GIB)
        assert candidate.candidate_level == CandidateLevel.FULL
        assert len(candidate.evictions) >= 1


# ============================================================================
# Node: deploy tests
# ============================================================================


class TestNodeDeploy:
    def test_deploy_fractional_model(self, make_node):
        """Verify deploying a small model uses fractional GPU memory allocation."""
        node = make_node(total_gpus=2, gpu_memory_bytes=80 * GIB)
        candidate = Candidate(
            candidate_level=CandidateLevel.FREE,
            gpus_required=1,
            gpu_ids=[0],
            gpu_mem_bytes_by_id={0: 20 * GIB},
            gpu_memory_required_bytes=20 * GIB,
        )
        node.deploy("model-a", "r1", candidate, size_bytes=5 * GIB)

        dep = node._get_from_map(node.deployments, "model-a", "r1")
        assert dep is not None
        assert dep.model_key == "model-a"
        assert dep.replica_id == "r1"
        assert dep.deployment_level == DeploymentLevel.HOT
        assert dep.gpu_memory_fraction is not None
        assert 0 < dep.gpu_memory_fraction <= 0.99

    def test_deploy_full_gpu_model(self, make_node):
        """Verify deploying a multi-GPU model allocates full GPUs with no fraction."""
        node = make_node(total_gpus=4, gpu_memory_bytes=80 * GIB)
        candidate = Candidate(
            candidate_level=CandidateLevel.FREE,
            gpus_required=2,
            gpu_ids=[0, 1],
            gpu_mem_bytes_by_id={0: 80 * GIB, 1: 80 * GIB},
            gpu_memory_required_bytes=200 * GIB,  # larger than single GPU
        )
        node.deploy("model-a", "r1", candidate, size_bytes=100 * GIB)

        dep = node._get_from_map(node.deployments, "model-a", "r1")
        assert dep is not None
        assert len(dep.gpu_mem_bytes_by_id) == 2
        # Multi-GPU: gpu_memory_fraction should be None
        assert dep.gpu_memory_fraction is None

    def test_deploy_evicts_candidates(self, make_node, make_deployment):
        """Verify deploy() evicts listed candidates and deploys the new model."""
        node = make_node(total_gpus=2, gpu_memory_bytes=80 * GIB)
        # Existing deployment on GPU 0
        existing = make_deployment(
            model_key="model-old", replica_id="r1",
            gpu_mem_bytes_by_id={0: 80 * GIB}, size_bytes=10 * GIB,
        )
        existing.deployed = time.time() - 1000
        node._set_in_map(node.deployments, existing)
        node.resources.available_gpus = [1]
        node.resources.gpu_memory_available_bytes_by_id[0] = 0

        candidate = Candidate(
            candidate_level=CandidateLevel.FULL,
            gpus_required=2,
            gpu_ids=[],
            gpu_mem_bytes_by_id={},
            gpu_memory_required_bytes=200 * GIB,
            evictions=[("model-old", "r1")],
        )
        node.deploy("model-new", "r-new", candidate, size_bytes=100 * GIB)

        # Old model should be evicted from deployments (moved to cache)
        assert node._get_from_map(node.deployments, "model-old", "r1") is None
        # New model should be deployed
        assert node._get_from_map(node.deployments, "model-new", "r-new") is not None

    def test_deploy_from_cache_returns_cpu_memory(self, make_node, make_deployment):
        """Verify deploying from cache removes the cache entry and restores CPU memory."""
        node = make_node(
            total_gpus=2, gpu_memory_bytes=80 * GIB,
            cpu_memory_bytes=256 * GIB,
        )
        # Put model in cache (using 10 GiB CPU)
        cached = make_deployment(
            model_key="model-a", replica_id="r1",
            deployment_level=DeploymentLevel.WARM,
            gpu_mem_bytes_by_id={}, size_bytes=10 * GIB,
        )
        node._set_in_map(node.cache, cached)
        node.resources.available_cpu_memory_bytes = 246 * GIB  # 256 - 10

        candidate = Candidate(
            candidate_level=CandidateLevel.CACHED_AND_FREE,
            gpus_required=1,
            gpu_ids=[0],
            gpu_mem_bytes_by_id={0: 20 * GIB},
            gpu_memory_required_bytes=20 * GIB,
        )
        node.deploy("model-a", "r1", candidate, size_bytes=10 * GIB)

        # Cache entry should be removed
        assert node._get_from_map(node.cache, "model-a", "r1") is None
        # CPU memory should be restored
        assert node.resources.available_cpu_memory_bytes == 256 * GIB

    def test_deploy_gpu_memory_fraction_clamped(self, make_node):
        """gpu_memory_fraction should be clamped between 0.01 and 0.99."""
        node = make_node(total_gpus=1, gpu_memory_bytes=80 * GIB)
        # Very small fraction
        candidate = Candidate(
            candidate_level=CandidateLevel.FREE,
            gpus_required=1,
            gpu_ids=[0],
            gpu_mem_bytes_by_id={0: 1},  # tiny amount
            gpu_memory_required_bytes=1,
        )
        node.deploy("model-tiny", "r1", candidate, size_bytes=1)

        dep = node._get_from_map(node.deployments, "model-tiny", "r1")
        assert dep.gpu_memory_fraction >= 0.01

    def test_deploy_full_gpu_sets_fraction_to_099(self, make_node):
        """Deploying with full GPU memory should set fraction to 0.99."""
        node = make_node(total_gpus=2, gpu_memory_bytes=80 * GIB)
        candidate = Candidate(
            candidate_level=CandidateLevel.FREE,
            gpus_required=1,
            gpu_ids=[0],
            gpu_mem_bytes_by_id={0: 80 * GIB},
            gpu_memory_required_bytes=80 * GIB,
        )
        node.deploy("model-a", "r1", candidate, size_bytes=80 * GIB)

        dep = node._get_from_map(node.deployments, "model-a", "r1")
        assert dep.gpu_memory_fraction == 0.99


# ============================================================================
# Node: evict tests
# ============================================================================


class TestNodeEvict:
    def test_evict_frees_gpu_memory(self, make_node, make_deployment):
        """Verify evicting a deployment restores GPU memory and availability."""
        node = make_node(total_gpus=2, gpu_memory_bytes=80 * GIB)
        dep = make_deployment(
            model_key="model-a", replica_id="r1",
            gpu_mem_bytes_by_id={0: 80 * GIB}, size_bytes=10 * GIB,
        )
        node._set_in_map(node.deployments, dep)
        node.resources.gpu_memory_available_bytes_by_id[0] = 0
        node.resources.available_gpus = [1]

        node.evict("model-a", "r1")

        # GPU memory restored
        assert node.resources.gpu_memory_available_bytes_by_id[0] == 80 * GIB
        # GPU 0 back in available
        assert 0 in node.resources.available_gpus

    def test_evict_creates_warm_cache_entry(self, make_node, make_deployment):
        """Verify eviction moves the deployment to cache as WARM with cleared GPU map."""
        node = make_node(
            total_gpus=2, gpu_memory_bytes=80 * GIB,
            cpu_memory_bytes=256 * GIB,
        )
        dep = make_deployment(
            model_key="model-a", replica_id="r1",
            gpu_mem_bytes_by_id={0: 80 * GIB}, size_bytes=10 * GIB,
        )
        node._set_in_map(node.deployments, dep)
        node.resources.gpu_memory_available_bytes_by_id[0] = 0

        node.evict("model-a", "r1")

        # Should be in cache now
        cached = node._get_from_map(node.cache, "model-a", "r1")
        assert cached is not None
        assert cached.deployment_level == DeploymentLevel.WARM
        assert cached.gpu_mem_bytes_by_id == {}
        assert cached.dedicated is False

    def test_evict_reduces_cpu_memory(self, make_node, make_deployment):
        """Verify eviction to cache consumes CPU memory equal to model size."""
        node = make_node(
            total_gpus=2, gpu_memory_bytes=80 * GIB,
            cpu_memory_bytes=256 * GIB,
        )
        dep = make_deployment(
            model_key="model-a", replica_id="r1",
            gpu_mem_bytes_by_id={0: 80 * GIB}, size_bytes=10 * GIB,
        )
        node._set_in_map(node.deployments, dep)

        initial_cpu = node.resources.available_cpu_memory_bytes
        node.evict("model-a", "r1")

        assert node.resources.available_cpu_memory_bytes == initial_cpu - 10 * GIB

    def test_evict_nonexistent_raises(self, make_node):
        """Verify evicting a non-existent deployment raises KeyError."""
        node = make_node()
        with pytest.raises(KeyError, match="Deployment not found"):
            node.evict("nonexistent", "r1")

    def test_evict_removes_from_deployments(self, make_node, make_deployment):
        """Verify eviction removes the deployment from the active deployments map."""
        node = make_node(total_gpus=2, gpu_memory_bytes=80 * GIB)
        dep = make_deployment(
            model_key="model-a", replica_id="r1",
            gpu_mem_bytes_by_id={0: 80 * GIB}, size_bytes=10 * GIB,
        )
        node._set_in_map(node.deployments, dep)

        node.evict("model-a", "r1")
        assert node._get_from_map(node.deployments, "model-a", "r1") is None

    def test_evict_when_no_cpu_memory_evicts_cache_entries(self, make_node, make_deployment):
        """When CPU memory is insufficient, evict cache entries to make room."""
        node = make_node(
            total_gpus=2, gpu_memory_bytes=80 * GIB,
            cpu_memory_bytes=50 * GIB,
            available_cpu_memory_bytes=0,
        )

        # Cache entry using 30 GiB CPU memory
        cache_entry = make_deployment(
            model_key="cached-model", replica_id="r-cache",
            deployment_level=DeploymentLevel.WARM,
            gpu_mem_bytes_by_id={}, size_bytes=30 * GIB,
        )
        node._set_in_map(node.cache, cache_entry)

        # Active deployment to evict: 10 GiB
        dep = make_deployment(
            model_key="model-a", replica_id="r1",
            gpu_mem_bytes_by_id={0: 80 * GIB}, size_bytes=10 * GIB,
        )
        node._set_in_map(node.deployments, dep)
        node.resources.gpu_memory_available_bytes_by_id[0] = 0

        node.evict("model-a", "r1")

        # cached-model should have been evicted from cache to make room
        assert node._get_from_map(node.cache, "cached-model", "r-cache") is None
        # model-a should now be in cache
        assert node._get_from_map(node.cache, "model-a", "r1") is not None

    def test_evict_respects_exclude_set_for_cache(self, make_node, make_deployment):
        """Cache eviction should skip models in the exclude set."""
        node = make_node(
            total_gpus=2, gpu_memory_bytes=80 * GIB,
            cpu_memory_bytes=50 * GIB,
            available_cpu_memory_bytes=0,
        )

        # Two cache entries
        c1 = make_deployment(
            model_key="protected-model", replica_id="r-cache1",
            deployment_level=DeploymentLevel.WARM,
            gpu_mem_bytes_by_id={}, size_bytes=30 * GIB,
        )
        c2 = make_deployment(
            model_key="expendable-model", replica_id="r-cache2",
            deployment_level=DeploymentLevel.WARM,
            gpu_mem_bytes_by_id={}, size_bytes=30 * GIB,
        )
        node._set_in_map(node.cache, c1)
        node._set_in_map(node.cache, c2)

        dep = make_deployment(
            model_key="model-a", replica_id="r1",
            gpu_mem_bytes_by_id={0: 80 * GIB}, size_bytes=10 * GIB,
        )
        node._set_in_map(node.deployments, dep)
        node.resources.gpu_memory_available_bytes_by_id[0] = 0

        node.evict("model-a", "r1", exclude={"protected-model"})

        # Protected model should still be in cache
        assert node._get_from_map(node.cache, "protected-model", "r-cache1") is not None
        # Expendable model should have been evicted from cache
        assert node._get_from_map(node.cache, "expendable-model", "r-cache2") is None

    def test_evict_no_cache_when_cpu_insufficient_and_no_cache_to_evict(self, make_node, make_deployment):
        """When CPU memory is insufficient and no cache entries to evict, skip caching entirely."""
        node = make_node(
            total_gpus=2, gpu_memory_bytes=80 * GIB,
            cpu_memory_bytes=5 * GIB,
            available_cpu_memory_bytes=0,
        )

        # Deploy a large model (10 GiB), but no CPU memory and no cache to evict
        dep = make_deployment(
            model_key="model-a", replica_id="r1",
            gpu_mem_bytes_by_id={0: 80 * GIB}, size_bytes=10 * GIB,
        )
        node._set_in_map(node.deployments, dep)
        node.resources.gpu_memory_available_bytes_by_id[0] = 0

        node.evict("model-a", "r1")

        # Model should NOT be cached (not enough CPU memory)
        assert node._get_from_map(node.cache, "model-a", "r1") is None
        # GPU memory should still be freed
        assert node.resources.gpu_memory_available_bytes_by_id[0] == 80 * GIB

    def test_evict_multi_gpu_deployment(self, make_node, make_deployment):
        """Evicting a multi-GPU deployment frees all its GPU memory."""
        node = make_node(total_gpus=4, gpu_memory_bytes=80 * GIB)
        dep = make_deployment(
            model_key="model-a", replica_id="r1",
            gpu_mem_bytes_by_id={0: 80 * GIB, 1: 80 * GIB},
            size_bytes=100 * GIB,
        )
        node._set_in_map(node.deployments, dep)
        node.resources.gpu_memory_available_bytes_by_id[0] = 0
        node.resources.gpu_memory_available_bytes_by_id[1] = 0
        node.resources.available_gpus = [2, 3]

        node.evict("model-a", "r1")

        assert node.resources.gpu_memory_available_bytes_by_id[0] == 80 * GIB
        assert node.resources.gpu_memory_available_bytes_by_id[1] == 80 * GIB
        assert 0 in node.resources.available_gpus
        assert 1 in node.resources.available_gpus


# ============================================================================
# Node: map helper tests
# ============================================================================


class TestNodeMapHelpers:
    def test_set_and_get(self, make_node, make_deployment):
        """Verify _set_in_map and _get_from_map store and retrieve a deployment."""
        node = make_node()
        dep = make_deployment(model_key="m1", replica_id="r1")
        node._set_in_map(node.deployments, dep)
        assert node._get_from_map(node.deployments, "m1", "r1") is dep

    def test_get_missing_returns_none(self, make_node):
        """Verify _get_from_map returns None for a non-existent deployment."""
        node = make_node()
        assert node._get_from_map(node.deployments, "missing", "r1") is None

    def test_remove_from_map(self, make_node, make_deployment):
        """Verify _remove_from_map removes deployment and cleans up empty model keys."""
        node = make_node()
        dep = make_deployment(model_key="m1", replica_id="r1")
        node._set_in_map(node.deployments, dep)
        removed = node._remove_from_map(node.deployments, "m1", "r1")
        assert removed is dep
        assert node._get_from_map(node.deployments, "m1", "r1") is None
        # Model key should also be removed from top-level dict
        assert "m1" not in node.deployments

    def test_remove_nonexistent_returns_none(self, make_node):
        """Verify _remove_from_map returns None for a non-existent key."""
        node = make_node()
        assert node._remove_from_map(node.deployments, "missing", "r1") is None

    def test_flatten_map(self, make_node, make_deployment):
        """Verify _flatten_map returns all deployments across model keys as a flat list."""
        node = make_node()
        d1 = make_deployment(model_key="m1", replica_id="r1")
        d2 = make_deployment(model_key="m1", replica_id="r2")
        d3 = make_deployment(model_key="m2", replica_id="r3")
        node._set_in_map(node.deployments, d1)
        node._set_in_map(node.deployments, d2)
        node._set_in_map(node.deployments, d3)

        flat = node._flatten_map(node.deployments)
        assert len(flat) == 3
        assert d1 in flat
        assert d2 in flat
        assert d3 in flat

    def test_remove_keeps_other_replicas(self, make_node, make_deployment):
        """Verify removing one replica keeps sibling replicas under the same model key."""
        node = make_node()
        d1 = make_deployment(model_key="m1", replica_id="r1")
        d2 = make_deployment(model_key="m1", replica_id="r2")
        node._set_in_map(node.deployments, d1)
        node._set_in_map(node.deployments, d2)

        node._remove_from_map(node.deployments, "m1", "r1")
        assert node._get_from_map(node.deployments, "m1", "r2") is d2
        assert "m1" in node.deployments


# ============================================================================
# Node: get_state tests
# ============================================================================


class TestNodeGetState:
    def test_empty_state(self, make_node):
        """Verify get_state returns correct defaults for a node with no deployments."""
        node = make_node(node_id="n1", name="test-node")
        state = node.get_state()
        assert state["id"] == "n1"
        assert state["name"] == "test-node"
        assert state["deployments"] == []
        assert state["num_deployments"] == 0
        assert state["cache"] == []
        assert state["cache_size"] == 0

    def test_state_with_deployments(self, make_node, make_deployment):
        """Verify get_state includes deployment info when deployments exist."""
        node = make_node(node_id="n1", name="test-node")
        dep = make_deployment(model_key="m1", replica_id="r1", size_bytes=10 * GIB)
        node._set_in_map(node.deployments, dep)

        state = node.get_state()
        assert state["num_deployments"] == 1
        assert len(state["deployments"]) == 1
        assert state["deployments"][0]["model_key"] == "m1"

    def test_state_with_cache(self, make_node, make_deployment):
        """Verify get_state includes cache info and cache_size when entries exist."""
        node = make_node(node_id="n1", name="test-node")
        cached = make_deployment(
            model_key="m1", replica_id="r1",
            deployment_level=DeploymentLevel.WARM,
            gpu_mem_bytes_by_id={}, size_bytes=5 * GIB,
        )
        node._set_in_map(node.cache, cached)

        state = node.get_state()
        assert len(state["cache"]) == 1
        assert state["cache_size"] == 5 * GIB


# ============================================================================
# Integration-style tests: evaluate -> deploy -> evict cycle
# ============================================================================


class TestNodeLifecycle:
    def test_evaluate_deploy_evict_cycle(self, make_node):
        """Full lifecycle: evaluate a model, deploy it, then evict it."""
        node = make_node(
            total_gpus=2, gpu_memory_bytes=80 * GIB,
            cpu_memory_bytes=256 * GIB,
        )

        # Evaluate
        candidate = node.evaluate("model-a", "r1", 5 * GIB)
        assert candidate.candidate_level == CandidateLevel.FREE

        # Deploy
        node.deploy("model-a", "r1", candidate, size_bytes=5 * GIB)
        assert node._get_from_map(node.deployments, "model-a", "r1") is not None

        # Evict
        node.evict("model-a", "r1")
        assert node._get_from_map(node.deployments, "model-a", "r1") is None
        assert node._get_from_map(node.cache, "model-a", "r1") is not None

    def test_deploy_from_cache_then_re_evict(self, make_node):
        """Deploy a model, evict to cache, re-deploy from cache, evict again."""
        node = make_node(
            total_gpus=2, gpu_memory_bytes=80 * GIB,
            cpu_memory_bytes=256 * GIB,
        )

        # First deploy
        candidate = node.evaluate("model-a", "r1", 5 * GIB)
        node.deploy("model-a", "r1", candidate, size_bytes=5 * GIB)

        # Evict to cache
        node.evict("model-a", "r1")
        assert node._get_from_map(node.cache, "model-a", "r1") is not None

        # Re-evaluate (should be CACHED_AND_FREE)
        candidate2 = node.evaluate("model-a", "r1", 5 * GIB)
        assert candidate2.candidate_level == CandidateLevel.CACHED_AND_FREE

        # Re-deploy from cache
        cpu_before = node.resources.available_cpu_memory_bytes
        node.deploy("model-a", "r1", candidate2, size_bytes=5 * GIB)
        assert node._get_from_map(node.cache, "model-a", "r1") is None
        assert node._get_from_map(node.deployments, "model-a", "r1") is not None
        # CPU memory should be returned
        assert node.resources.available_cpu_memory_bytes == cpu_before + 5 * GIB

    def test_multiple_fractional_deployments_on_same_gpu(self, make_node):
        """Multiple small models can share a single GPU via fractional allocation."""
        node = make_node(total_gpus=1, gpu_memory_bytes=80 * GIB)

        # Deploy 3 small models (5 GiB each, requiring 15 GiB each via 3x factor)
        for i in range(3):
            candidate = node.evaluate(f"model-{i}", f"r{i}", 5 * GIB)
            assert candidate.candidate_level == CandidateLevel.FREE
            assert candidate.gpus_required == 1
            node.deploy(f"model-{i}", f"r{i}", candidate, size_bytes=5 * GIB)

        # All 3 should be deployed
        for i in range(3):
            assert node._get_from_map(node.deployments, f"model-{i}", f"r{i}") is not None

    def test_deploy_dedicated_flag_propagates(self, make_node):
        """The dedicated flag should be set on the deployment."""
        node = make_node(total_gpus=2, gpu_memory_bytes=80 * GIB)
        candidate = Candidate(
            candidate_level=CandidateLevel.FREE,
            gpus_required=1,
            gpu_ids=[0],
            gpu_mem_bytes_by_id={0: 20 * GIB},
            gpu_memory_required_bytes=20 * GIB,
        )
        node.deploy("model-a", "r1", candidate, size_bytes=5 * GIB, dedicated=True)

        dep = node._get_from_map(node.deployments, "model-a", "r1")
        assert dep.dedicated is True
