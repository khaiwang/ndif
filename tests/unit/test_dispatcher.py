"""Unit tests for the Dispatcher class.

Tests cover construction, dispatch, removal, purging, eviction handling,
error handling, state aggregation, and all Redis-stream event handlers.
"""

import asyncio
import pickle
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from src.services.api.src.queue.processor import ProcessorStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Shared patch targets for the Dispatcher constructor dependencies.
_DISPATCHER_MODULE = "src.services.api.src.queue.dispatcher"

PATCH_RAY = f"{_DISPATCHER_MODULE}.RayProvider"
PATCH_REDIS = f"{_DISPATCHER_MODULE}.RedisProvider"
PATCH_OBJ = f"{_DISPATCHER_MODULE}.ObjectStoreProvider"
PATCH_LOGGER = f"{_DISPATCHER_MODULE}.set_logger"
PATCH_PATCH = f"{_DISPATCHER_MODULE}.patch"
PATCH_PROCESSOR = f"{_DISPATCHER_MODULE}.Processor"
PATCH_CONTROLLER = f"{_DISPATCHER_MODULE}.controller_handle"
PATCH_SUBMIT = f"{_DISPATCHER_MODULE}.submit"


def _make_event(**overrides) -> dict:
    """Build a Redis-stream event_data dict with bytes keys/values."""
    defaults = {
        "event_type": "",
        "model_key": "meta-llama/Llama-2-7b",
        "replicas": "",
        "response_key": "response:123",
        "request_id": "req-001",
        "replica_id": "",
    }
    defaults.update(overrides)
    return {k.encode(): v.encode() for k, v in defaults.items()}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dispatcher_deps():
    """Patch all external dependencies and yield a dict of mocks.

    The returned dict contains:
        ray, redis, obj, logger, patch_fn
    and the ready-to-use ``dispatcher`` instance.
    """
    with (
        patch(PATCH_RAY) as mock_ray,
        patch(PATCH_REDIS) as mock_redis,
        patch(PATCH_OBJ) as mock_obj,
        patch(PATCH_LOGGER) as mock_log,
        patch(PATCH_PATCH) as mock_patch_fn,
    ):
        # Make connected() return True on first call so the while-loop
        # in connect() exits immediately.
        mock_ray.connected.return_value = True
        mock_log.return_value = MagicMock()

        # Provide async_client and sync_client stubs.
        mock_redis.sync_client = MagicMock()
        mock_redis.async_client = AsyncMock()

        from src.services.api.src.queue.dispatcher import Dispatcher

        dispatcher = Dispatcher()

        yield {
            "dispatcher": dispatcher,
            "ray": mock_ray,
            "redis": mock_redis,
            "obj": mock_obj,
            "logger": mock_log,
            "patch_fn": mock_patch_fn,
        }


@pytest.fixture
def dispatcher(dispatcher_deps):
    """Shorthand: return just the dispatcher instance."""
    return dispatcher_deps["dispatcher"]


@pytest.fixture
def mock_ray(dispatcher_deps):
    return dispatcher_deps["ray"]


@pytest.fixture
def mock_redis(dispatcher_deps):
    return dispatcher_deps["redis"]


# ---------------------------------------------------------------------------
# Construction / connect
# ---------------------------------------------------------------------------

class TestDispatcherInit:
    """Tests for Dispatcher.__init__ and connect()."""

    def test_init_creates_empty_processors(self, dispatcher):
        assert dispatcher.processors == {}

    def test_init_creates_queues(self, dispatcher):
        assert isinstance(dispatcher.error_queue, asyncio.Queue)
        assert isinstance(dispatcher.eviction_queue, asyncio.Queue)

    def test_connect_calls_ray_sequence(self, dispatcher_deps):
        """connect() should call RayProvider.connected, reset, connect and
        set the Redis 'ray:connected' key."""
        mock_ray = dispatcher_deps["ray"]
        mock_redis = dispatcher_deps["redis"]

        # connected() is True from the start, so reset/connect are NOT called
        # (while-loop body only runs when connected() returns False).
        # But the delete and set on sync_client should happen.
        mock_redis.sync_client.delete.assert_called_with("ray:connected")
        mock_redis.sync_client.set.assert_called_with("ray:connected", "1")

    def test_connect_retries_on_failure(self):
        """When connected() returns False then True, reset and connect
        are called at least once."""
        with (
            patch(PATCH_RAY) as mock_ray,
            patch(PATCH_REDIS) as mock_redis,
            patch(PATCH_OBJ),
            patch(PATCH_LOGGER) as mock_log,
            patch(PATCH_PATCH),
        ):
            # First call: not connected, second: still not after reset/connect,
            # third: connected.
            mock_ray.connected.side_effect = [False, False, True]
            mock_ray.connect.side_effect = [Exception("fail"), None]
            mock_log.return_value = MagicMock()
            mock_redis.sync_client = MagicMock()

            from src.services.api.src.queue.dispatcher import Dispatcher

            with patch("time.sleep"):  # avoid real sleep
                d = Dispatcher()

            assert mock_ray.reset.call_count >= 1
            assert mock_ray.connect.call_count >= 1

    def test_objectstore_connect_called(self, dispatcher_deps):
        dispatcher_deps["obj"].connect.assert_called_once()

    def test_patch_called(self, dispatcher_deps):
        dispatcher_deps["patch_fn"].assert_called_once()


# ---------------------------------------------------------------------------
# dispatch()
# ---------------------------------------------------------------------------

class TestDispatch:
    """Tests for Dispatcher.dispatch()."""

    @pytest.mark.asyncio
    async def test_dispatch_creates_processor_for_new_model(self, dispatcher):
        """First request for a model_key should create a Processor and
        start its worker task."""
        with patch(PATCH_PROCESSOR) as MockProcessor:
            mock_proc = MagicMock()
            mock_proc.processor_worker = AsyncMock()
            MockProcessor.return_value = mock_proc

            request = MagicMock()
            request.model_key = "model-A"

            dispatcher.dispatch(request)

            MockProcessor.assert_called_once_with(
                "model-A", dispatcher.eviction_queue, dispatcher.error_queue
            )
            assert "model-A" in dispatcher.processors
            mock_proc.enqueue.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_dispatch_reuses_existing_processor(self, dispatcher):
        """Second request for the same model_key should NOT create a new
        Processor; it should enqueue on the existing one."""
        mock_proc = MagicMock()
        dispatcher.processors["model-A"] = mock_proc

        request = MagicMock()
        request.model_key = "model-A"

        dispatcher.dispatch(request)

        # No new Processor was constructed; existing one received enqueue.
        mock_proc.enqueue.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_dispatch_multiple_models(self, dispatcher):
        """Dispatching to different model_keys creates separate Processors."""
        with patch(PATCH_PROCESSOR) as MockProcessor:
            procs = [MagicMock(), MagicMock()]
            procs[0].processor_worker = AsyncMock()
            procs[1].processor_worker = AsyncMock()
            MockProcessor.side_effect = procs

            req_a = MagicMock()
            req_a.model_key = "model-A"
            req_b = MagicMock()
            req_b.model_key = "model-B"

            dispatcher.dispatch(req_a)
            dispatcher.dispatch(req_b)

            assert len(dispatcher.processors) == 2
            assert "model-A" in dispatcher.processors
            assert "model-B" in dispatcher.processors


# ---------------------------------------------------------------------------
# remove() / purge()
# ---------------------------------------------------------------------------

class TestRemoveAndPurge:
    """Tests for Dispatcher.remove() and Dispatcher.purge()."""

    def test_remove_pops_processor_and_sets_cancelled(self, dispatcher):
        mock_proc = MagicMock()
        mock_proc.status = ProcessorStatus.READY
        dispatcher.processors["model-A"] = mock_proc

        dispatcher.remove("model-A", "gone")

        assert "model-A" not in dispatcher.processors
        assert mock_proc.status == ProcessorStatus.CANCELLED
        mock_proc.purge.assert_called_once_with("gone")

    def test_remove_raises_on_missing_key(self, dispatcher):
        with pytest.raises(KeyError):
            dispatcher.remove("nonexistent", "nope")

    def test_purge_removes_all_processors(self, dispatcher):
        for name in ("A", "B", "C"):
            p = MagicMock()
            p.status = ProcessorStatus.READY
            dispatcher.processors[name] = p

        dispatcher.purge("boom")

        assert dispatcher.processors == {}

    def test_purge_on_empty_is_noop(self, dispatcher):
        dispatcher.purge("nothing")
        assert dispatcher.processors == {}


# ---------------------------------------------------------------------------
# handle_evictions()
# ---------------------------------------------------------------------------

class TestHandleEvictions:
    """Tests for Dispatcher.handle_evictions()."""

    def test_full_removal_when_replica_id_is_none(self, dispatcher):
        mock_proc = MagicMock()
        mock_proc.status = ProcessorStatus.READY
        dispatcher.processors["model-A"] = mock_proc

        dispatcher.eviction_queue.put_nowait(("model-A", "evicted", None))
        dispatcher.handle_evictions()

        assert "model-A" not in dispatcher.processors
        assert mock_proc.status == ProcessorStatus.CANCELLED

    def test_replica_removal_keeps_processor_if_not_cancelled(self, dispatcher):
        mock_proc = MagicMock()
        mock_proc.status = ProcessorStatus.READY  # stays non-CANCELLED
        dispatcher.processors["model-A"] = mock_proc

        dispatcher.eviction_queue.put_nowait(("model-A", "replica gone", "replica-1"))
        dispatcher.handle_evictions()

        mock_proc.remove_replica.assert_called_once_with("replica-1", "replica gone")
        assert "model-A" in dispatcher.processors

    def test_replica_removal_pops_processor_if_cancelled(self, dispatcher):
        mock_proc = MagicMock()
        mock_proc.status = ProcessorStatus.CANCELLED
        dispatcher.processors["model-A"] = mock_proc

        dispatcher.eviction_queue.put_nowait(("model-A", "last replica", "replica-1"))
        dispatcher.handle_evictions()

        mock_proc.remove_replica.assert_called_once_with("replica-1", "last replica")
        assert "model-A" not in dispatcher.processors

    def test_handles_multiple_evictions(self, dispatcher):
        for name in ("A", "B"):
            p = MagicMock()
            p.status = ProcessorStatus.READY
            dispatcher.processors[name] = p

        dispatcher.eviction_queue.put_nowait(("A", "reason", None))
        dispatcher.eviction_queue.put_nowait(("B", "reason", None))
        dispatcher.handle_evictions()

        assert dispatcher.processors == {}

    def test_missing_model_key_does_not_crash(self, dispatcher):
        """If the model_key is already gone, the exception is logged but
        processing continues."""
        dispatcher.eviction_queue.put_nowait(("missing-model", "reason", None))
        # Should not raise
        dispatcher.handle_evictions()

    def test_replica_eviction_for_missing_processor_is_noop(self, dispatcher):
        """Replica eviction for a non-existent model_key is silently skipped."""
        dispatcher.eviction_queue.put_nowait(("ghost", "reason", "r-1"))
        dispatcher.handle_evictions()
        # No error, no crash

    def test_empty_eviction_queue_is_noop(self, dispatcher):
        dispatcher.handle_evictions()
        assert dispatcher.processors == {}


# ---------------------------------------------------------------------------
# handle_errors()
# ---------------------------------------------------------------------------

class TestHandleErrors:
    """Tests for Dispatcher.handle_errors()."""

    @pytest.mark.asyncio
    async def test_connection_error_purges_and_reconnects(
        self, dispatcher, mock_ray, mock_redis
    ):
        mock_proc = MagicMock()
        mock_proc.status = ProcessorStatus.READY
        dispatcher.processors["model-A"] = mock_proc

        error = Exception("Unrecoverable error in data channel")
        dispatcher.error_queue.put_nowait(("model-A", error))

        mock_ray.is_connection_error.return_value = True
        mock_ray.connected.return_value = True

        await dispatcher.handle_errors()

        # Processors should be purged
        assert dispatcher.processors == {}
        # env cache should be cleared
        mock_redis.async_client.delete.assert_awaited_with("env")

    @pytest.mark.asyncio
    async def test_non_connection_error_resets_to_ready(
        self, dispatcher, mock_ray
    ):
        mock_proc = MagicMock()
        mock_proc.status = ProcessorStatus.BUSY
        dispatcher.processors["model-A"] = mock_proc

        error = Exception("some transient bug")
        dispatcher.error_queue.put_nowait(("model-A", error))

        mock_ray.is_connection_error.return_value = False
        mock_ray.connected.return_value = True

        await dispatcher.handle_errors()

        # Processor should be reset to READY
        assert mock_proc.status == ProcessorStatus.READY
        assert "model-A" in dispatcher.processors

    @pytest.mark.asyncio
    async def test_disconnected_ray_triggers_reconnect(
        self, dispatcher, mock_ray, mock_redis
    ):
        """Even if the error itself is not a connection error, if
        RayProvider.connected() returns False, we still purge and reconnect."""
        mock_proc = MagicMock()
        mock_proc.status = ProcessorStatus.READY
        dispatcher.processors["model-A"] = mock_proc

        error = Exception("ordinary error")
        dispatcher.error_queue.put_nowait(("model-A", error))

        mock_ray.is_connection_error.return_value = False
        # connected() False means the Ray cluster is down
        mock_ray.connected.return_value = False
        # After reconnect, connected returns True
        mock_ray.connected.side_effect = [False, True]

        await dispatcher.handle_errors()

        assert dispatcher.processors == {}

    @pytest.mark.asyncio
    async def test_empty_error_queue_is_noop(self, dispatcher, mock_ray):
        await dispatcher.handle_errors()
        mock_ray.is_connection_error.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_errors_handled(self, dispatcher, mock_ray):
        for name in ("A", "B"):
            p = MagicMock()
            p.status = ProcessorStatus.BUSY
            dispatcher.processors[name] = p

        dispatcher.error_queue.put_nowait(("A", Exception("err A")))
        dispatcher.error_queue.put_nowait(("B", Exception("err B")))

        mock_ray.is_connection_error.return_value = False
        mock_ray.connected.return_value = True

        await dispatcher.handle_errors()

        for name in ("A", "B"):
            assert dispatcher.processors[name].status == ProcessorStatus.READY

    @pytest.mark.asyncio
    async def test_error_for_already_removed_processor(
        self, dispatcher, mock_ray
    ):
        """If the processor was already removed (e.g. by an eviction) before
        handle_errors runs, it should not crash."""
        dispatcher.error_queue.put_nowait(("ghost", Exception("oops")))

        mock_ray.is_connection_error.return_value = False
        mock_ray.connected.return_value = True

        await dispatcher.handle_errors()
        # No KeyError, no crash


# ---------------------------------------------------------------------------
# get_state()
# ---------------------------------------------------------------------------

class TestGetState:
    """Tests for Dispatcher.get_state()."""

    def test_get_state_aggregates_processors(self, dispatcher):
        for name in ("A", "B"):
            p = MagicMock()
            p.get_state.return_value = {"model_key": name, "status": "ready"}
            dispatcher.processors[name] = p

        state = dispatcher.get_state()

        assert "processors" in state
        assert len(state["processors"]) == 2
        assert state["processors"]["A"]["model_key"] == "A"
        assert state["processors"]["B"]["model_key"] == "B"

    def test_get_state_empty(self, dispatcher):
        state = dispatcher.get_state()
        assert state == {"processors": {}}


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

class TestHandleDeployEvent:
    """Tests for Dispatcher._handle_deploy_event()."""

    @pytest.mark.asyncio
    async def test_updates_replica_count(self, dispatcher):
        mock_proc = MagicMock()
        mock_proc.requested_replica_count = 1
        dispatcher.processors["meta-llama/Llama-2-7b"] = mock_proc

        event = _make_event(event_type="deploy", replicas="3")
        await dispatcher._handle_deploy_event(event)

        assert mock_proc.requested_replica_count == 3

    @pytest.mark.asyncio
    async def test_clamps_replica_count_to_one(self, dispatcher):
        mock_proc = MagicMock()
        mock_proc.requested_replica_count = 2
        dispatcher.processors["meta-llama/Llama-2-7b"] = mock_proc

        event = _make_event(event_type="deploy", replicas="0")
        await dispatcher._handle_deploy_event(event)

        # max(1, 0) == 1
        assert mock_proc.requested_replica_count == 1

    @pytest.mark.asyncio
    async def test_ignores_deploy_for_unknown_model(self, dispatcher):
        """Deploy event for a model without an active Processor is ignored."""
        event = _make_event(event_type="deploy", model_key="unknown/model", replicas="2")
        # Should not raise
        await dispatcher._handle_deploy_event(event)

    @pytest.mark.asyncio
    async def test_no_replicas_field_does_not_update(self, dispatcher):
        """When replicas is empty string (no value), replicas is None and
        no update happens."""
        mock_proc = MagicMock()
        mock_proc.requested_replica_count = 2
        dispatcher.processors["meta-llama/Llama-2-7b"] = mock_proc

        event = _make_event(event_type="deploy", replicas="")
        await dispatcher._handle_deploy_event(event)

        # Should NOT have been changed since replicas is None
        assert mock_proc.requested_replica_count == 2


class TestHandleEvictEvent:
    """Tests for Dispatcher._handle_evict_event()."""

    @pytest.mark.asyncio
    async def test_full_eviction_removes_processor(self, dispatcher):
        mock_proc = MagicMock()
        mock_proc.status = ProcessorStatus.READY
        dispatcher.processors["meta-llama/Llama-2-7b"] = mock_proc

        event = _make_event(event_type="evict", replica_id="")
        await dispatcher._handle_evict_event(event)

        assert "meta-llama/Llama-2-7b" not in dispatcher.processors
        assert mock_proc.status == ProcessorStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_replica_eviction(self, dispatcher):
        mock_proc = MagicMock()
        mock_proc.status = ProcessorStatus.READY
        dispatcher.processors["meta-llama/Llama-2-7b"] = mock_proc

        event = _make_event(event_type="evict", replica_id="replica-42")
        await dispatcher._handle_evict_event(event)

        mock_proc.remove_replica.assert_called_once_with(
            "replica-42", "Model replica removed by external command"
        )

    @pytest.mark.asyncio
    async def test_replica_eviction_pops_cancelled_processor(self, dispatcher):
        mock_proc = MagicMock()
        mock_proc.status = ProcessorStatus.CANCELLED
        dispatcher.processors["meta-llama/Llama-2-7b"] = mock_proc

        event = _make_event(event_type="evict", replica_id="replica-42")
        await dispatcher._handle_evict_event(event)

        assert "meta-llama/Llama-2-7b" not in dispatcher.processors

    @pytest.mark.asyncio
    async def test_evict_for_unknown_model_is_noop(self, dispatcher):
        event = _make_event(event_type="evict", model_key="ghost/model")
        await dispatcher._handle_evict_event(event)
        # No error


class TestHandleKillRequest:
    """Tests for Dispatcher._handle_kill_request()."""

    @pytest.mark.asyncio
    async def test_kill_found_in_processor(self, dispatcher, mock_redis):
        mock_proc = MagicMock()
        mock_proc.kill_request = AsyncMock(return_value={
            "status": "removed_from_queue",
            "message": "Removed request req-001 from queue",
        })
        dispatcher.processors["model-A"] = mock_proc

        event = _make_event(event_type="kill_request", request_id="req-001")
        await dispatcher._handle_kill_request(event)

        mock_proc.kill_request.assert_awaited_once_with("req-001")
        # The result should be pushed to Redis
        mock_redis.async_client.lpush.assert_awaited_once()
        pushed_data = mock_redis.async_client.lpush.call_args[0]
        assert pushed_data[0] == "response:123"
        unpickled = pickle.loads(pushed_data[1])
        assert unpickled["status"] == "removed_from_queue"

    @pytest.mark.asyncio
    async def test_kill_not_found_in_any_processor(self, dispatcher, mock_redis):
        mock_proc = MagicMock()
        mock_proc.kill_request = AsyncMock(return_value={
            "status": "not_found",
            "message": "nope",
        })
        dispatcher.processors["model-A"] = mock_proc

        event = _make_event(event_type="kill_request", request_id="req-999")
        await dispatcher._handle_kill_request(event)

        pushed_data = mock_redis.async_client.lpush.call_args[0]
        unpickled = pickle.loads(pushed_data[1])
        assert unpickled["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_kill_with_no_processors(self, dispatcher, mock_redis):
        event = _make_event(event_type="kill_request", request_id="req-001")
        await dispatcher._handle_kill_request(event)

        pushed_data = mock_redis.async_client.lpush.call_args[0]
        unpickled = pickle.loads(pushed_data[1])
        assert unpickled["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_kill_stops_iterating_on_first_match(self, dispatcher, mock_redis):
        """When the first processor finds the request, the second processor
        should NOT be queried."""
        proc_a = MagicMock()
        proc_a.kill_request = AsyncMock(return_value={
            "status": "cancelled_execution",
            "message": "killed",
        })
        proc_b = MagicMock()
        proc_b.kill_request = AsyncMock(return_value={
            "status": "not_found",
            "message": "nope",
        })
        dispatcher.processors["A"] = proc_a
        dispatcher.processors["B"] = proc_b

        event = _make_event(event_type="kill_request", request_id="req-001")
        await dispatcher._handle_kill_request(event)

        proc_a.kill_request.assert_awaited_once()
        proc_b.kill_request.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_kill_exception_pushes_error(self, dispatcher, mock_redis):
        mock_proc = MagicMock()
        mock_proc.kill_request = AsyncMock(side_effect=RuntimeError("bang"))
        dispatcher.processors["model-A"] = mock_proc

        event = _make_event(event_type="kill_request", request_id="req-001")
        await dispatcher._handle_kill_request(event)

        pushed_data = mock_redis.async_client.lpush.call_args[0]
        unpickled = pickle.loads(pushed_data[1])
        assert unpickled["status"] == "error"
        assert "bang" in unpickled["message"]


class TestHandleQueueStateRequest:
    """Tests for Dispatcher._handle_queue_state_request()."""

    @pytest.mark.asyncio
    async def test_pushes_state_to_redis(self, dispatcher, mock_redis):
        mock_proc = MagicMock()
        mock_proc.get_state.return_value = {"status": "ready"}
        dispatcher.processors["model-A"] = mock_proc

        event = _make_event(event_type="queue_state_request")
        await dispatcher._handle_queue_state_request(event)

        mock_redis.async_client.lpush.assert_awaited_once()
        key, data = mock_redis.async_client.lpush.call_args[0]
        assert key == "response:123"
        unpickled = pickle.loads(data)
        assert "processors" in unpickled
        assert "model-A" in unpickled["processors"]

    @pytest.mark.asyncio
    async def test_pushes_empty_state(self, dispatcher, mock_redis):
        event = _make_event(event_type="queue_state_request")
        await dispatcher._handle_queue_state_request(event)

        key, data = mock_redis.async_client.lpush.call_args[0]
        unpickled = pickle.loads(data)
        assert unpickled == {"processors": {}}

    @pytest.mark.asyncio
    async def test_pushes_error_on_exception(self, dispatcher, mock_redis):
        """If get_state raises, the error is serialised and pushed."""
        mock_proc = MagicMock()
        mock_proc.get_state.side_effect = RuntimeError("state boom")
        dispatcher.processors["model-A"] = mock_proc

        event = _make_event(event_type="queue_state_request")
        await dispatcher._handle_queue_state_request(event)

        key, data = mock_redis.async_client.lpush.call_args[0]
        unpickled = pickle.loads(data)
        assert "error" in unpickled


class TestHandleEnvEvent:
    """Tests for Dispatcher._handle_env_event()."""

    @pytest.mark.asyncio
    async def test_returns_cached_env(self, dispatcher, mock_redis):
        cached = pickle.dumps({"python": "3.10"})
        mock_redis.async_client.get = AsyncMock(return_value=cached)

        event = _make_event(event_type="env")
        await dispatcher._handle_env_event(event)

        # Should push cached value directly
        mock_redis.async_client.lpush.assert_awaited_once_with(
            "response:123", cached
        )

    @pytest.mark.asyncio
    async def test_fetches_env_from_controller(self, dispatcher, mock_redis):
        mock_redis.async_client.get = AsyncMock(return_value=None)

        env_info = {"python": "3.10", "packages": []}

        async def _fake_submit(*args, **kwargs):
            return env_info

        with (
            patch(PATCH_CONTROLLER) as mock_ctrl,
            patch(PATCH_SUBMIT, side_effect=_fake_submit) as mock_submit,
        ):
            event = _make_event(event_type="env")
            await dispatcher._handle_env_event(event)

        # Should have cached the result
        mock_redis.async_client.set.assert_awaited()
        set_call = mock_redis.async_client.set.call_args
        assert set_call[0][0] == "env"

        # Should have pushed the result
        mock_redis.async_client.lpush.assert_awaited()

    @pytest.mark.asyncio
    async def test_env_error_pushes_error_result(self, dispatcher, mock_redis):
        mock_redis.async_client.get = AsyncMock(return_value=None)

        with (
            patch(PATCH_CONTROLLER) as mock_ctrl,
            patch(PATCH_SUBMIT) as mock_submit,
        ):
            mock_submit.side_effect = RuntimeError("controller down")

            event = _make_event(event_type="env")
            await dispatcher._handle_env_event(event)

        pushed_data = mock_redis.async_client.lpush.call_args[0]
        unpickled = pickle.loads(pushed_data[1])
        assert "error" in unpickled
        assert "controller down" in unpickled["error"]
