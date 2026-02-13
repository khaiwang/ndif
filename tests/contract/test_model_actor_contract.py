"""Contract tests for the ModelActor interface.

Verifies that the ModelActor class exposes the methods expected
by the API service (Processor/Replicas).
"""

import inspect
from unittest.mock import MagicMock, patch

import pytest


class TestModelActorInterface:
    """Verify ModelActor exposes the methods the API side expects."""

    def test_model_actor_has_call_method(self):
        """ModelActor must have a __call__ method (used by Processor._execute_on_replica)."""
        from src.services.ray.src.ray.deployments.modeling.base import BaseModelDeployment

        assert hasattr(BaseModelDeployment, "__call__")

    def test_model_actor_has_cancel_method(self):
        """ModelActor must have a cancel method (used by Processor.kill_request)."""
        from src.services.ray.src.ray.deployments.modeling.base import BaseModelDeployment

        assert hasattr(BaseModelDeployment, "cancel")

    def test_model_actor_has_ray_ready(self):
        """ModelActor must have __ray_ready__ (used by Replica.wait_until_ready)."""
        # Ray actors support __ray_ready__ by default through Ray's actor protocol.
        # BaseModelDeployment inherits this from Ray actor base.
        # We just verify that our code expects this method name.
        from src.services.api.src.queue.replicas import Replica

        source = inspect.getsource(Replica.wait_until_ready)
        assert "__ray_ready__" in source

    def test_model_actor_has_to_cache_method(self):
        """ModelActor must have to_cache (used by Deployment.cache)."""
        from src.services.ray.src.ray.deployments.modeling.base import BaseModelDeployment

        assert hasattr(BaseModelDeployment, "to_cache")

    def test_model_actor_has_from_cache_method(self):
        """ModelActor must have from_cache (used by Deployment.from_cache)."""
        from src.services.ray.src.ray.deployments.modeling.base import BaseModelDeployment

        assert hasattr(BaseModelDeployment, "from_cache")


class TestReplicaActorNaming:
    """Verify that the actor naming convention is consistent between API and Ray."""

    def test_deployment_name_matches_api_convention(self):
        """Deployment.name must follow 'ModelActor:{model_key}:{replica_id}' pattern."""
        from src.services.ray.src.ray.deployments.controller.cluster.deployment import (
            Deployment,
            DeploymentLevel,
        )

        deployment = Deployment(
            model_key="meta-llama/Llama-2-7b",
            replica_id="abc123",
            deployment_level=DeploymentLevel.HOT,
            gpu_mem_bytes_by_id={0: 80 * 1024**3},
            gpu_memory_fraction=0.5,
            size_bytes=10 * 1024**3,
        )
        assert deployment.name == "ModelActor:meta-llama/Llama-2-7b:abc123"

    def test_api_actor_name_matches_deployment(self):
        """model_actor_name() in API util must produce the same format."""
        from src.services.api.src.queue.util import model_actor_name

        name = model_actor_name("meta-llama/Llama-2-7b", "abc123")
        assert name == "ModelActor:meta-llama/Llama-2-7b:abc123"
