"""Contract tests for the API ↔ Ray Controller boundary.

These tests verify that the message formats exchanged between
the API service (Processor/Dispatcher) and the Ray Controller
stay consistent. They test both the producer and consumer sides
of each contract.
"""

import inspect
from unittest.mock import MagicMock, patch

import pytest


class TestDeployContract:
    """Verify deploy() call signature and return format."""

    def test_controller_deploy_returns_expected_format(self, make_node, make_resources):
        """Cluster.deploy() must return {result: {(model_key, replica_id): status}, evictions: set}."""
        from src.services.ray.src.ray.deployments.controller.cluster.cluster import Cluster

        cluster = Cluster()
        node = make_node(node_id="n1", name="node-1", total_gpus=4)
        cluster.nodes["n1"] = node

        # Mock evaluator to return a small model size
        cluster.evaluator = MagicMock(return_value=1 * 1024**3)  # 1 GiB

        with patch(
            "src.services.ray.src.ray.deployments.controller.cluster.deployment.Deployment.delete"
        ):
            results, change = cluster.deploy(["model-a"], replicas=1)

        # Verify structure
        assert "result" in results
        assert "evictions" in results
        assert isinstance(results["result"], dict)
        assert isinstance(results["evictions"], set)

        # Verify result keys are (model_key, replica_id) tuples
        for key, status in results["result"].items():
            assert isinstance(key, tuple)
            assert len(key) == 2
            model_key, replica_id = key
            assert isinstance(model_key, str)
            assert isinstance(replica_id, str)
            # Status should be a string (CandidateLevel name)
            assert isinstance(status, str)

    def test_processor_parses_deploy_result(self):
        """Processor.provision() correctly parses the deploy result format."""
        # This is the format that Controller returns and Processor expects:
        deploy_result = {
            "result": {
                ("model-a", "replica-1"): "free",
                ("model-a", "replica-2"): "cached_and_free",
            },
            "evictions": [("model-b", "old-replica")],
        }

        # Verify the Processor can parse deployment statuses
        from src.services.api.src.queue.processor import DeploymentStatus

        for key, status_str in deploy_result["result"].items():
            status = DeploymentStatus(status_str.lower())
            assert status is not None

    def test_deploy_evictions_format(self, make_node):
        """Evictions in deploy result are (model_key, replica_id) tuples."""
        from src.services.ray.src.ray.deployments.controller.cluster.cluster import Cluster
        from src.services.ray.src.ray.deployments.controller.cluster.node import (
            Candidate,
            CandidateLevel,
        )

        cluster = Cluster()

        # Setup: deploy model-a on a node, then deploy model-b which causes eviction
        node = make_node(node_id="n1", name="node-1", total_gpus=2, gpu_memory_bytes=40 * 1024**3)
        cluster.nodes["n1"] = node

        # First deploy model-a
        cluster.evaluator = MagicMock(return_value=30 * 1024**3)
        with patch(
            "src.services.ray.src.ray.deployments.controller.cluster.deployment.Deployment.delete"
        ):
            results_a, _ = cluster.deploy(["model-a"], replicas=1)

        # Now deploy model-b which may need to evict
        cluster.evaluator = MagicMock(return_value=30 * 1024**3)
        with patch(
            "src.services.ray.src.ray.deployments.controller.cluster.deployment.Deployment.delete"
        ):
            results_b, _ = cluster.deploy(["model-b"], replicas=1)

        # Evictions should be (model_key, replica_id) tuples
        for eviction in results_b["evictions"]:
            assert isinstance(eviction, tuple)
            assert len(eviction) == 2


class TestEvictContract:
    """Verify evict() call signature and return format."""

    def test_evict_by_model_key_format(self, make_node):
        """Cluster.evict(model_keys) returns {(model_key, replica_id): {status, ...}}."""
        from src.services.ray.src.ray.deployments.controller.cluster.cluster import Cluster

        cluster = Cluster()
        node = make_node(node_id="n1", name="node-1", total_gpus=4)
        cluster.nodes["n1"] = node

        cluster.evaluator = MagicMock(return_value=1 * 1024**3)
        with patch(
            "src.services.ray.src.ray.deployments.controller.cluster.deployment.Deployment.delete"
        ):
            cluster.deploy(["model-a"], replicas=1)
            results, change = cluster.evict(["model-a"])

        assert isinstance(results, dict)
        for key, value in results.items():
            assert isinstance(key, tuple)
            assert "status" in value

    def test_evict_not_found_format(self, make_node):
        """Evicting a non-deployed model returns not_found status."""
        from src.services.ray.src.ray.deployments.controller.cluster.cluster import Cluster

        cluster = Cluster()
        node = make_node(node_id="n1", name="node-1")
        cluster.nodes["n1"] = node

        results, change = cluster.evict(["nonexistent-model"])
        assert change is False
        for key, value in results.items():
            assert value["status"] == "not_found"


class TestGetDeploymentContract:
    """Verify get_deployment return format matches Processor.check_dedicated parsing."""

    @pytest.mark.asyncio
    async def test_check_dedicated_parses_none(self):
        """check_dedicated returns False when controller returns None."""
        from src.services.api.src.queue.processor import Processor

        processor = Processor.__new__(Processor)
        processor.model_key = "test-model"

        async def mock_submit(*args, **kwargs):
            return None

        with patch("src.services.api.src.queue.processor.submit", side_effect=mock_submit):
            result = await processor.check_dedicated(MagicMock())
            assert result is False

    @pytest.mark.asyncio
    async def test_check_dedicated_parses_not_found(self):
        """check_dedicated returns False when deployments_state is not_found."""
        from src.services.api.src.queue.processor import Processor

        processor = Processor.__new__(Processor)
        processor.model_key = "test-model"

        async def mock_submit(*args, **kwargs):
            return {"deployments_state": "not_found"}

        with patch("src.services.api.src.queue.processor.submit", side_effect=mock_submit):
            result = await processor.check_dedicated(MagicMock())
            assert result is False

    @pytest.mark.asyncio
    async def test_check_dedicated_parses_dedicated_true(self):
        """check_dedicated returns True when a deployment has dedicated=True."""
        from src.services.api.src.queue.processor import Processor

        processor = Processor.__new__(Processor)
        processor.model_key = "test-model"

        async def mock_submit(*args, **kwargs):
            return {
                "replica-1": {"dedicated": True, "status": "hot"},
            }

        with patch("src.services.api.src.queue.processor.submit", side_effect=mock_submit):
            result = await processor.check_dedicated(MagicMock())
            assert result is True


class TestDeploymentStatusContract:
    """Verify DeploymentStatus enum values match CandidateLevel names."""

    def test_all_candidate_levels_have_deployment_status(self):
        """Every CandidateLevel name (lowered) should be a valid DeploymentStatus."""
        from src.services.api.src.queue.processor import DeploymentStatus
        from src.services.ray.src.ray.deployments.controller.cluster.node import CandidateLevel

        for level in CandidateLevel:
            status_str = level.name.lower()
            try:
                ds = DeploymentStatus(status_str)
                assert ds.value == status_str
            except ValueError:
                pytest.fail(
                    f"CandidateLevel.{level.name} has no matching DeploymentStatus "
                    f"(tried '{status_str}')"
                )
