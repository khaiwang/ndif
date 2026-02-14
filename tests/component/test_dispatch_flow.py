"""Component tests for the Dispatcher + Processor request routing flow.

Tests the coordination between Dispatcher and Processor with mocked
Ray and Redis dependencies.
"""

import asyncio
import pickle
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest


def _make_dispatcher():
    """Create a Dispatcher with all external dependencies mocked."""
    with patch("src.services.api.src.queue.dispatcher.RayProvider") as mock_ray, \
         patch("src.services.api.src.queue.dispatcher.RedisProvider") as mock_redis, \
         patch("src.services.api.src.queue.dispatcher.ObjectStoreProvider") as mock_obj, \
         patch("src.services.api.src.queue.dispatcher.set_logger") as mock_log, \
         patch("src.services.api.src.queue.dispatcher.patch") as mock_patch:
        mock_ray.connected.return_value = True
        mock_redis.sync_client = MagicMock()
        mock_redis.async_client = AsyncMock()
        mock_log.return_value = MagicMock()
        mock_obj.connect.return_value = None

        from src.services.api.src.queue.dispatcher import Dispatcher
        dispatcher = Dispatcher()
        dispatcher._mock_ray = mock_ray
        dispatcher._mock_redis_cls = mock_redis
        return dispatcher


def _make_mock_request(model_key="test-model", request_id="req-001", hotswapping=True):
    """Create a mock BackendRequestModel."""
    req = MagicMock()
    req.id = request_id
    req.model_key = model_key
    req.hotswapping = hotswapping
    req.session_id = "sess-1"
    req.create_response.return_value = MagicMock()
    req.create_response.return_value.respond.return_value = None
    return req


class TestDispatchFlow:
    """Test request routing through Dispatcher to Processors."""

    def test_dispatch_creates_processor_on_first_request(self):
        """Verify that dispatching a request for a new model creates a Processor and starts its task."""
        dispatcher = _make_dispatcher()
        request = _make_mock_request()

        with patch("src.services.api.src.queue.dispatcher.asyncio") as mock_asyncio:
            mock_asyncio.create_task = MagicMock()
            dispatcher.dispatch(request)

        assert "test-model" in dispatcher.processors
        mock_asyncio.create_task.assert_called_once()

    def test_dispatch_reuses_processor_for_same_model(self):
        """Verify that multiple requests for the same model reuse a single Processor."""
        dispatcher = _make_dispatcher()
        req1 = _make_mock_request(request_id="req-1")
        req2 = _make_mock_request(request_id="req-2")

        with patch("src.services.api.src.queue.dispatcher.asyncio") as mock_asyncio:
            mock_asyncio.create_task = MagicMock()
            dispatcher.dispatch(req1)
            dispatcher.dispatch(req2)

        # Only one processor should be created
        assert len(dispatcher.processors) == 1
        # create_task called only once (for the first dispatch)
        mock_asyncio.create_task.assert_called_once()

    def test_dispatch_creates_separate_processors_per_model(self):
        """Verify that requests for different models each get their own Processor."""
        dispatcher = _make_dispatcher()
        req1 = _make_mock_request(model_key="model-a", request_id="req-1")
        req2 = _make_mock_request(model_key="model-b", request_id="req-2")

        with patch("src.services.api.src.queue.dispatcher.asyncio") as mock_asyncio:
            mock_asyncio.create_task = MagicMock()
            dispatcher.dispatch(req1)
            dispatcher.dispatch(req2)

        assert "model-a" in dispatcher.processors
        assert "model-b" in dispatcher.processors
        assert mock_asyncio.create_task.call_count == 2


class TestEvictionFlow:
    """Test eviction event handling in Dispatcher."""

    def test_handle_evictions_removes_processor(self):
        """Verify that a whole-model eviction event removes the Processor and calls purge."""
        dispatcher = _make_dispatcher()

        # Manually add a processor
        mock_processor = MagicMock()
        mock_processor.status = MagicMock()
        dispatcher.processors["model-a"] = mock_processor

        # Queue an eviction event (replica_id=None means remove whole model)
        dispatcher.eviction_queue.put_nowait(("model-a", "Model evicted", None))
        dispatcher.handle_evictions()

        assert "model-a" not in dispatcher.processors
        mock_processor.purge.assert_called_once_with("Model evicted")

    def test_handle_evictions_removes_specific_replica(self):
        """Verify that a replica-specific eviction removes the replica but keeps the Processor."""
        dispatcher = _make_dispatcher()
        from src.services.api.src.queue.processor import ProcessorStatus

        mock_processor = MagicMock()
        mock_processor.status = ProcessorStatus.READY
        dispatcher.processors["model-a"] = mock_processor

        dispatcher.eviction_queue.put_nowait(("model-a", "Replica evicted", "replica-1"))
        dispatcher.handle_evictions()

        mock_processor.remove_replica.assert_called_once_with("replica-1", "Replica evicted")
        # Processor should still exist if not cancelled
        assert "model-a" in dispatcher.processors

    def test_handle_evictions_removes_processor_when_last_replica(self):
        """Verify that evicting the last replica removes the Processor when status is CANCELLED."""
        dispatcher = _make_dispatcher()
        from src.services.api.src.queue.processor import ProcessorStatus

        mock_processor = MagicMock()
        mock_processor.status = ProcessorStatus.CANCELLED
        dispatcher.processors["model-a"] = mock_processor

        dispatcher.eviction_queue.put_nowait(("model-a", "Replica evicted", "replica-1"))
        dispatcher.handle_evictions()

        assert "model-a" not in dispatcher.processors


class TestErrorFlow:
    """Test error handling in Dispatcher."""

    @pytest.mark.asyncio
    async def test_handle_errors_connection_error_triggers_reconnect(self):
        """Verify that a Ray connection error purges all processors and triggers reconnect."""
        dispatcher = _make_dispatcher()
        from src.services.api.src.queue.processor import ProcessorStatus

        mock_processor = MagicMock()
        dispatcher.processors["model-a"] = mock_processor

        error = RuntimeError("Ray client has already been disconnected")
        dispatcher.error_queue.put_nowait(("model-a", error))

        with patch("src.services.api.src.queue.dispatcher.RayProvider") as mock_ray:
            mock_ray.is_connection_error.return_value = True
            mock_ray.connected.return_value = True
            with patch("src.services.api.src.queue.dispatcher.RedisProvider") as mock_redis:
                mock_redis.sync_client = MagicMock()
                mock_redis.async_client = AsyncMock()
                dispatcher.connect = MagicMock()
                await dispatcher.handle_errors()

        # Should have purged all processors
        assert len(dispatcher.processors) == 0
        dispatcher.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_errors_non_connection_resets_to_ready(self):
        """Verify that a non-connection error resets the Processor status to READY."""
        dispatcher = _make_dispatcher()
        from src.services.api.src.queue.processor import ProcessorStatus

        mock_processor = MagicMock()
        mock_processor.status = ProcessorStatus.BUSY
        dispatcher.processors["model-a"] = mock_processor

        error = RuntimeError("Some transient error")
        dispatcher.error_queue.put_nowait(("model-a", error))

        with patch("src.services.api.src.queue.dispatcher.RayProvider") as mock_ray:
            mock_ray.is_connection_error.return_value = False
            mock_ray.connected.return_value = True
            with patch("src.services.api.src.queue.dispatcher.RedisProvider") as mock_redis:
                mock_redis.async_client = AsyncMock()
                await dispatcher.handle_errors()

        # Processor should be reset to READY
        assert mock_processor.status == ProcessorStatus.READY
        assert "model-a" in dispatcher.processors


class TestEventHandlers:
    """Test Dispatcher event handlers."""

    @pytest.mark.asyncio
    async def test_handle_deploy_event_updates_replica_count(self):
        """Verify that a deploy event updates the Processor's requested replica count."""
        dispatcher = _make_dispatcher()
        mock_processor = MagicMock()
        mock_processor.requested_replica_count = 1
        dispatcher.processors["model-a"] = mock_processor

        event_data = {
            b"event_type": b"deploy",
            b"model_key": b"model-a",
            b"replicas": b"3",
        }
        await dispatcher._handle_deploy_event(event_data)
        assert mock_processor.requested_replica_count == 3

    @pytest.mark.asyncio
    async def test_handle_deploy_event_ignores_unknown_model(self):
        """Verify that a deploy event for an unknown model does not raise an error."""
        dispatcher = _make_dispatcher()
        event_data = {
            b"event_type": b"deploy",
            b"model_key": b"unknown-model",
            b"replicas": b"2",
        }
        # Should not raise
        await dispatcher._handle_deploy_event(event_data)

    @pytest.mark.asyncio
    async def test_handle_evict_event_removes_processor(self):
        """Verify that an evict event removes the Processor from the dispatcher."""
        dispatcher = _make_dispatcher()
        mock_processor = MagicMock()
        mock_processor.status = MagicMock()
        dispatcher.processors["model-a"] = mock_processor

        event_data = {
            b"event_type": b"evict",
            b"model_key": b"model-a",
            b"replica_id": b"",
        }
        await dispatcher._handle_evict_event(event_data)
        assert "model-a" not in dispatcher.processors

    @pytest.mark.asyncio
    async def test_handle_queue_state_request(self):
        """Verify that a queue state request returns pickled processor state via Redis."""
        dispatcher = _make_dispatcher()

        event_data = {
            b"event_type": b"queue_state_request",
            b"response_key": b"response:queue:123",
        }

        mock_client = AsyncMock()
        with patch("src.services.api.src.queue.dispatcher.RedisProvider") as mock_redis:
            mock_redis.async_client = mock_client
            await dispatcher._handle_queue_state_request(event_data)

        mock_client.lpush.assert_called_once()
        call_args = mock_client.lpush.call_args
        assert call_args[0][0] == "response:queue:123"
        # Verify it's a valid pickled state
        state = pickle.loads(call_args[0][1])
        assert "processors" in state

    @pytest.mark.asyncio
    async def test_handle_kill_request_found(self):
        """Verify that a kill request for an existing request returns a success status."""
        dispatcher = _make_dispatcher()
        mock_processor = MagicMock()
        mock_processor.kill_request = AsyncMock(return_value={
            "status": "removed_from_queue",
            "message": "Removed",
        })
        dispatcher.processors["model-a"] = mock_processor

        event_data = {
            b"event_type": b"kill_request",
            b"request_id": b"req-001",
            b"response_key": b"response:kill:123",
        }

        mock_client = AsyncMock()
        with patch("src.services.api.src.queue.dispatcher.RedisProvider") as mock_redis:
            mock_redis.async_client = mock_client
            await dispatcher._handle_kill_request(event_data)

        mock_client.lpush.assert_called_once()
        result = pickle.loads(mock_client.lpush.call_args[0][1])
        assert result["status"] == "removed_from_queue"

    @pytest.mark.asyncio
    async def test_handle_kill_request_not_found(self):
        """Verify that a kill request for a nonexistent request returns a not_found status."""
        dispatcher = _make_dispatcher()
        mock_processor = MagicMock()
        mock_processor.kill_request = AsyncMock(return_value={
            "status": "not_found",
            "message": "Not found",
        })
        dispatcher.processors["model-a"] = mock_processor

        event_data = {
            b"event_type": b"kill_request",
            b"request_id": b"req-999",
            b"response_key": b"response:kill:456",
        }

        mock_client = AsyncMock()
        with patch("src.services.api.src.queue.dispatcher.RedisProvider") as mock_redis:
            mock_redis.async_client = mock_client
            await dispatcher._handle_kill_request(event_data)

        result = pickle.loads(mock_client.lpush.call_args[0][1])
        assert result["status"] == "not_found"
