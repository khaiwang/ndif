"""Unit tests for Deployment actor lifecycle management.

Tests the Deployment class which wraps Ray actor operations:
construction, properties, state serialization, and lifecycle
methods (create, delete, restart, cache, from_cache).
"""

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import ray

from src.services.ray.src.ray.deployments.controller.cluster.deployment import (
    Deployment,
    DeploymentLevel,
)

# Module path for patching
_DEPLOYMENT_MODULE = (
    "src.services.ray.src.ray.deployments.controller.cluster.deployment"
)


@pytest.fixture(autouse=True)
def _reset_ray_mocks():
    """Reset global ray mock state between tests."""
    ray.get_actor.reset_mock()
    ray.get_actor.side_effect = None
    ray.kill.reset_mock()
    ray.kill.side_effect = None
    yield
    ray.get_actor.side_effect = None
    ray.kill.side_effect = None


# =========================================================================
# DeploymentLevel enum
# =========================================================================


class TestDeploymentLevel:
    """Tests for the DeploymentLevel enum."""

    def test_values(self):
        """HOT, WARM, COLD have expected string values."""
        assert DeploymentLevel.HOT.value == "hot"
        assert DeploymentLevel.WARM.value == "warm"
        assert DeploymentLevel.COLD.value == "cold"

    def test_exactly_three_members(self):
        """Enum has exactly three members."""
        assert len(DeploymentLevel) == 3


# =========================================================================
# Construction & Properties
# =========================================================================


class TestConstruction:
    """Tests for Deployment construction and property accessors."""

    def test_init_stores_all_fields(self, make_deployment):
        """All constructor arguments are stored as instance attributes."""
        dep = make_deployment(
            model_key="llama-7b",
            replica_id="r1",
            deployment_level=DeploymentLevel.WARM,
            gpu_mem_bytes_by_id={0: 40 * 1024**3, 1: 40 * 1024**3},
            gpu_memory_fraction=0.5,
            size_bytes=14 * 1024**3,
            dedicated=True,
            node_id="node-42",
        )
        assert dep.model_key == "llama-7b"
        assert dep.replica_id == "r1"
        assert dep.deployment_level == DeploymentLevel.WARM
        assert dep.gpu_mem_bytes_by_id == {0: 40 * 1024**3, 1: 40 * 1024**3}
        assert dep.gpu_memory_fraction == 0.5
        assert dep.size_bytes == 14 * 1024**3
        assert dep.dedicated is True
        assert dep.node_id == "node-42"

    def test_deployed_timestamp_set(self, make_deployment):
        """deployed attribute is set to current time at construction."""
        before = time.time()
        dep = make_deployment()
        after = time.time()
        assert before <= dep.deployed <= after

    def test_name_format(self, make_deployment):
        """name property returns 'ModelActor:{model_key}:{replica_id}'."""
        dep = make_deployment(model_key="meta-llama/Llama-2-7b", replica_id="rep-3")
        assert dep.name == "ModelActor:meta-llama/Llama-2-7b:rep-3"

    def test_actor_calls_ray_get_actor(self, make_deployment):
        """actor property calls ray.get_actor with correct name and namespace."""
        dep = make_deployment(model_key="gpt2", replica_id="r1")
        mock_actor = MagicMock()
        ray.get_actor.return_value = mock_actor

        result = dep.actor

        ray.get_actor.assert_called_with("ModelActor:gpt2:r1", namespace="NDIF")
        assert result is mock_actor

    def test_gpus_returns_keys(self, make_deployment):
        """gpus property returns GPU IDs from gpu_mem_bytes_by_id keys."""
        dep = make_deployment(gpu_mem_bytes_by_id={0: 80e9, 2: 80e9, 5: 40e9})
        assert dep.gpus == [0, 2, 5]


# =========================================================================
# get_state()
# =========================================================================


class TestGetState:
    """Tests for Deployment.get_state() serialization."""

    def test_returns_all_fields(self, make_deployment):
        """State dict contains all 9 expected keys with correct values."""
        dep = make_deployment(
            model_key="llama-7b",
            replica_id="r1",
            deployment_level=DeploymentLevel.HOT,
            gpu_mem_bytes_by_id={0: 80 * 1024**3},
            gpu_memory_fraction=None,
            size_bytes=10 * 1024**3,
            dedicated=False,
            node_id="node-1",
        )
        state = dep.get_state()

        assert state["model_key"] == "llama-7b"
        assert state["replica_id"] == "r1"
        assert state["gpu_mem_bytes_by_id"] == {0: 80 * 1024**3}
        assert state["gpu_memory_fraction"] is None
        assert state["size_bytes"] == 10 * 1024**3
        assert state["dedicated"] is False
        assert state["node_id"] == "node-1"
        assert "deployed" in state

    def test_deployment_level_serialized_as_value(self, make_deployment):
        """deployment_level is the string value, not the enum object."""
        dep = make_deployment(deployment_level=DeploymentLevel.WARM)
        state = dep.get_state()
        assert state["deployment_level"] == "warm"
        assert isinstance(state["deployment_level"], str)

    def test_deployed_timestamp_matches(self, make_deployment):
        """deployed in state matches the instance attribute."""
        dep = make_deployment()
        state = dep.get_state()
        assert state["deployed"] == dep.deployed


# =========================================================================
# end_time()
# =========================================================================


class TestEndTime:
    """Tests for Deployment.end_time() calculation."""

    def test_calculates_correctly(self, make_deployment):
        """Returns deployed + seconds as a datetime."""
        dep = make_deployment()
        dep.deployed = 1700000000.0  # Fixed timestamp
        result = dep.end_time(3600)
        expected = datetime.fromtimestamp(1700003600.0, tz=timezone.utc)
        assert result == expected

    def test_is_timezone_aware(self, make_deployment):
        """Result has UTC timezone info."""
        dep = make_deployment()
        result = dep.end_time(60)
        assert result.tzinfo is timezone.utc


# =========================================================================
# delete()
# =========================================================================


class TestDelete:
    """Tests for Deployment.delete() — permanent actor termination."""

    def test_kills_actor_no_restart(self, make_deployment):
        """Calls ray.kill(actor, no_restart=True)."""
        dep = make_deployment()
        mock_actor = MagicMock()
        ray.get_actor.return_value = mock_actor

        dep.delete()

        ray.kill.assert_called_once_with(mock_actor, no_restart=True)

    def test_logs_on_get_actor_failure(self, make_deployment):
        """When ray.get_actor raises, exception is logged, not re-raised."""
        dep = make_deployment()
        ray.get_actor.side_effect = Exception("actor not found")

        dep.delete()  # Should not raise

        ray.kill.assert_not_called()
    def test_logs_on_kill_failure(self, make_deployment):
        """When ray.kill raises, exception is logged, not re-raised."""
        dep = make_deployment()
        mock_actor = MagicMock()
        ray.get_actor.return_value = mock_actor
        ray.kill.side_effect = Exception("kill failed")

        dep.delete()  # Should not raise


# =========================================================================
# restart()
# =========================================================================


class TestRestart:
    """Tests for Deployment.restart() — actor restart (no_restart=False)."""

    def test_kills_actor_with_restart(self, make_deployment):
        """Calls ray.kill(actor, no_restart=False)."""
        dep = make_deployment()
        mock_actor = MagicMock()
        ray.get_actor.return_value = mock_actor

        dep.restart()

        ray.kill.assert_called_once_with(mock_actor, no_restart=False)

    def test_logs_on_get_actor_failure(self, make_deployment):
        """When ray.get_actor raises, exception is logged, not re-raised."""
        dep = make_deployment()
        ray.get_actor.side_effect = Exception("actor not found")

        dep.restart()  # Should not raise

    def test_logs_on_kill_failure(self, make_deployment):
        """When ray.kill raises, exception is logged, not re-raised."""
        dep = make_deployment()
        mock_actor = MagicMock()
        ray.get_actor.return_value = mock_actor
        ray.kill.side_effect = Exception("kill failed")

        dep.restart()  # Should not raise


# =========================================================================
# cache()
# =========================================================================


class TestCache:
    """Tests for Deployment.cache() — move model to CPU cache."""

    def test_calls_to_cache_remote(self, make_deployment):
        """Calls actor.to_cache.remote() and returns the future."""
        dep = make_deployment()
        mock_actor = MagicMock()
        mock_future = MagicMock()
        mock_actor.to_cache.remote.return_value = mock_future
        ray.get_actor.return_value = mock_actor

        result = dep.cache()

        mock_actor.to_cache.remote.assert_called_once()
        assert result is mock_future

    def test_returns_none_on_failure(self, make_deployment):
        """Returns None when actor lookup or remote call fails."""
        dep = make_deployment()
        ray.get_actor.side_effect = Exception("actor gone")

        result = dep.cache()

        assert result is None

    def test_returns_none_on_remote_failure(self, make_deployment):
        """Returns None when to_cache.remote() raises."""
        dep = make_deployment()
        mock_actor = MagicMock()
        mock_actor.to_cache.remote.side_effect = Exception("cache failed")
        ray.get_actor.return_value = mock_actor

        result = dep.cache()

        assert result is None


# =========================================================================
# from_cache()
# =========================================================================


class TestFromCache:
    """Tests for Deployment.from_cache() — restore model from CPU cache."""

    def test_calls_remote_with_gpu_map(self, make_deployment):
        """Calls actor.from_cache.remote(gpu_mem_bytes_by_id)."""
        gpu_map = {0: 80 * 1024**3, 1: 80 * 1024**3}
        dep = make_deployment(gpu_mem_bytes_by_id=gpu_map)
        mock_actor = MagicMock()
        mock_future = MagicMock()
        mock_actor.from_cache.remote.return_value = mock_future
        ray.get_actor.return_value = mock_actor

        result = dep.from_cache()

        mock_actor.from_cache.remote.assert_called_once_with(gpu_map)
        assert result is mock_future

    def test_returns_none_on_failure(self, make_deployment):
        """Returns None when actor lookup fails."""
        dep = make_deployment()
        ray.get_actor.side_effect = Exception("actor gone")

        result = dep.from_cache()

        assert result is None

    def test_returns_none_on_remote_failure(self, make_deployment):
        """Returns None when from_cache.remote() raises."""
        dep = make_deployment()
        mock_actor = MagicMock()
        mock_actor.from_cache.remote.side_effect = Exception("restore failed")
        ray.get_actor.return_value = mock_actor

        result = dep.from_cache()

        assert result is None


# =========================================================================
# create()
# =========================================================================


class TestCreate:
    """Tests for Deployment.create() — Ray actor creation."""

    def _make_deployment_args(self):
        """Create a mock BaseModelDeploymentArgs."""
        args = MagicMock()
        args.model_dump.return_value = {
            "model_key": "llama-7b",
            "gpu_mem_bytes_by_id": {0: 80 * 1024**3},
        }
        return args

    def _setup_model_actor_mock(self):
        """Set up ModelActor.options().remote() chain mock."""
        mock_options = MagicMock()
        mock_remote = MagicMock()
        mock_options.remote = mock_remote
        return mock_options, mock_remote

    @patch(f"{_DEPLOYMENT_MODULE}.ModelActor")
    @patch(f"{_DEPLOYMENT_MODULE}.SioProvider")
    @patch(f"{_DEPLOYMENT_MODULE}.ObjectStoreProvider")
    @patch(f"{_DEPLOYMENT_MODULE}.MailgunProvider")
    def test_calls_model_actor_options(
        self, mock_mailgun, mock_obj, mock_sio, mock_model_actor, make_deployment
    ):
        """Verifies actor options: name, namespace, lifetime, resources."""
        mock_options, _ = self._setup_model_actor_mock()
        mock_model_actor.options.return_value = mock_options
        mock_sio.to_env.return_value = {}
        mock_obj.to_env.return_value = {}
        mock_mailgun.to_env.return_value = {}

        dep = make_deployment(model_key="llama-7b", replica_id="r1")
        args = self._make_deployment_args()

        dep.create("gpu-node-1", args)

        mock_model_actor.options.assert_called_once()
        call_kwargs = mock_model_actor.options.call_args[1]
        assert call_kwargs["name"] == "ModelActor:llama-7b:r1"
        assert call_kwargs["namespace"] == "NDIF"
        assert call_kwargs["lifetime"] == "detached"
        assert call_kwargs["resources"] == {"node:gpu-node-1": 0.01}

    @patch(f"{_DEPLOYMENT_MODULE}.ModelActor")
    @patch(f"{_DEPLOYMENT_MODULE}.SioProvider")
    @patch(f"{_DEPLOYMENT_MODULE}.ObjectStoreProvider")
    @patch(f"{_DEPLOYMENT_MODULE}.MailgunProvider")
    def test_passes_deployment_args(
        self, mock_mailgun, mock_obj, mock_sio, mock_model_actor, make_deployment
    ):
        """Calls .remote() with model_dump() of deployment_args."""
        mock_options, mock_remote = self._setup_model_actor_mock()
        mock_model_actor.options.return_value = mock_options
        mock_sio.to_env.return_value = {}
        mock_obj.to_env.return_value = {}
        mock_mailgun.to_env.return_value = {}

        dep = make_deployment()
        args = self._make_deployment_args()
        dep.create("node-1", args)

        mock_remote.assert_called_once_with(**args.model_dump())

    @patch(f"{_DEPLOYMENT_MODULE}.ModelActor")
    @patch(f"{_DEPLOYMENT_MODULE}.SioProvider")
    @patch(f"{_DEPLOYMENT_MODULE}.ObjectStoreProvider")
    @patch(f"{_DEPLOYMENT_MODULE}.MailgunProvider")
    def test_env_vars_include_providers(
        self, mock_mailgun, mock_obj, mock_sio, mock_model_actor, make_deployment
    ):
        """Runtime env vars include provider outputs and CUDA override."""
        mock_options, _ = self._setup_model_actor_mock()
        mock_model_actor.options.return_value = mock_options
        mock_sio.to_env.return_value = {"SIO_URL": "http://sio"}
        mock_obj.to_env.return_value = {"S3_ENDPOINT": "http://minio"}
        mock_mailgun.to_env.return_value = {"MAILGUN_KEY": "mg-key"}

        dep = make_deployment()
        dep.create("node-1", self._make_deployment_args())

        call_kwargs = mock_model_actor.options.call_args[1]
        env_vars = call_kwargs["runtime_env"]["env_vars"]
        assert env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "1"
        assert env_vars["SIO_URL"] == "http://sio"
        assert env_vars["S3_ENDPOINT"] == "http://minio"
        assert env_vars["MAILGUN_KEY"] == "mg-key"

    @patch(f"{_DEPLOYMENT_MODULE}.ModelActor")
    @patch(f"{_DEPLOYMENT_MODULE}.SioProvider")
    @patch(f"{_DEPLOYMENT_MODULE}.ObjectStoreProvider")
    @patch(f"{_DEPLOYMENT_MODULE}.MailgunProvider")
    def test_filters_none_env_vars(
        self, mock_mailgun, mock_obj, mock_sio, mock_model_actor, make_deployment
    ):
        """Env vars with None values are excluded."""
        mock_options, _ = self._setup_model_actor_mock()
        mock_model_actor.options.return_value = mock_options
        mock_sio.to_env.return_value = {"SIO_URL": "http://sio", "SIO_TOKEN": None}
        mock_obj.to_env.return_value = {}
        mock_mailgun.to_env.return_value = {}

        dep = make_deployment()
        dep.create("node-1", self._make_deployment_args())

        call_kwargs = mock_model_actor.options.call_args[1]
        env_vars = call_kwargs["runtime_env"]["env_vars"]
        assert "SIO_TOKEN" not in env_vars

    @patch(f"{_DEPLOYMENT_MODULE}.ModelActor")
    @patch(f"{_DEPLOYMENT_MODULE}.SioProvider")
    @patch(f"{_DEPLOYMENT_MODULE}.ObjectStoreProvider")
    @patch(f"{_DEPLOYMENT_MODULE}.MailgunProvider")
    def test_logs_on_failure(
        self, mock_mailgun, mock_obj, mock_sio, mock_model_actor, make_deployment
    ):
        """Exception during create is caught and logged, not re-raised."""
        mock_model_actor.options.side_effect = Exception("actor creation failed")
        mock_sio.to_env.return_value = {}
        mock_obj.to_env.return_value = {}
        mock_mailgun.to_env.return_value = {}

        dep = make_deployment()
        dep.create("node-1", self._make_deployment_args())  # Should not raise
