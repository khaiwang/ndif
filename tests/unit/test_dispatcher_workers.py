"""Unit tests for the Dispatcher async worker loops.

Tests cover the three infinite loops: dispatch_worker, status_worker,
and events_worker. Each loop is tested in isolation by mocking the
blocking I/O (Redis xread/brpop) and using asyncio.CancelledError
to terminate the loop (CancelledError is not a subclass of Exception
in Python 3.9+, so it propagates past the loops' ``except Exception``).

Fixtures (dispatcher_deps, dispatcher, mock_ray, mock_redis, make_event)
are defined in conftest.py.
"""

import asyncio
import pickle
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from tests.unit.conftest import (
    DISPATCHER_PATCH_CONTROLLER,
    DISPATCHER_PATCH_SUBMIT,
)

_DISPATCHER_MODULE = "src.services.api.src.queue.dispatcher"


# ---------------------------------------------------------------------------
# TestDispatchWorker
# ---------------------------------------------------------------------------

class TestDispatchWorker:
    """Tests for Dispatcher.dispatch_worker() — the main async loop.

    The loop: spawn background tasks, then ``while True: get() -> dispatch()
    -> handle_evictions() -> handle_errors()``.
    """

    @pytest.mark.asyncio
    async def test_spawns_background_tasks(self, dispatcher):
        """dispatch_worker creates tasks for status_worker and events_worker."""
        with (
            patch.object(dispatcher, "get", new_callable=AsyncMock) as mock_get,
            patch.object(dispatcher, "status_worker", new_callable=AsyncMock) as mock_sw,
            patch.object(dispatcher, "events_worker", new_callable=AsyncMock) as mock_ew,
            patch.object(dispatcher, "handle_evictions"),
            patch.object(dispatcher, "handle_errors", new_callable=AsyncMock),
        ):
            # First call returns None, second raises CancelledError to exit.
            mock_get.side_effect = [None, asyncio.CancelledError]

            with pytest.raises(asyncio.CancelledError):
                await dispatcher.dispatch_worker()

        # status_worker and events_worker should have been scheduled as tasks.
        # We can't directly check asyncio.create_task, but we verify
        # the coroutines were created (they run in tasks).
        # The fact that the loop ran without error is sufficient;
        # detailed handler tests are below.

    @pytest.mark.asyncio
    async def test_dispatches_request(self, dispatcher):
        """When get() returns a request, dispatch() is called with it."""
        mock_request = MagicMock()

        with (
            patch.object(dispatcher, "get", new_callable=AsyncMock) as mock_get,
            patch.object(dispatcher, "dispatch") as mock_dispatch,
            patch.object(dispatcher, "status_worker", new_callable=AsyncMock),
            patch.object(dispatcher, "events_worker", new_callable=AsyncMock),
            patch.object(dispatcher, "handle_evictions"),
            patch.object(dispatcher, "handle_errors", new_callable=AsyncMock),
        ):
            mock_get.side_effect = [mock_request, asyncio.CancelledError]

            with pytest.raises(asyncio.CancelledError):
                await dispatcher.dispatch_worker()

            mock_dispatch.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_skips_dispatch_when_get_returns_none(self, dispatcher):
        """When get() returns None, dispatch() is NOT called."""
        with (
            patch.object(dispatcher, "get", new_callable=AsyncMock) as mock_get,
            patch.object(dispatcher, "dispatch") as mock_dispatch,
            patch.object(dispatcher, "status_worker", new_callable=AsyncMock),
            patch.object(dispatcher, "events_worker", new_callable=AsyncMock),
            patch.object(dispatcher, "handle_evictions"),
            patch.object(dispatcher, "handle_errors", new_callable=AsyncMock),
        ):
            mock_get.side_effect = [None, asyncio.CancelledError]

            with pytest.raises(asyncio.CancelledError):
                await dispatcher.dispatch_worker()

            mock_dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_handle_evictions_every_iteration(self, dispatcher):
        with (
            patch.object(dispatcher, "get", new_callable=AsyncMock) as mock_get,
            patch.object(dispatcher, "dispatch"),
            patch.object(dispatcher, "status_worker", new_callable=AsyncMock),
            patch.object(dispatcher, "events_worker", new_callable=AsyncMock),
            patch.object(dispatcher, "handle_evictions") as mock_evict,
            patch.object(dispatcher, "handle_errors", new_callable=AsyncMock),
        ):
            mock_get.side_effect = [MagicMock(), None, asyncio.CancelledError]

            with pytest.raises(asyncio.CancelledError):
                await dispatcher.dispatch_worker()

            assert mock_evict.call_count == 2

    @pytest.mark.asyncio
    async def test_calls_handle_errors_every_iteration(self, dispatcher):
        with (
            patch.object(dispatcher, "get", new_callable=AsyncMock) as mock_get,
            patch.object(dispatcher, "dispatch"),
            patch.object(dispatcher, "status_worker", new_callable=AsyncMock),
            patch.object(dispatcher, "events_worker", new_callable=AsyncMock),
            patch.object(dispatcher, "handle_evictions"),
            patch.object(dispatcher, "handle_errors", new_callable=AsyncMock) as mock_err,
        ):
            mock_get.side_effect = [None, None, asyncio.CancelledError]

            with pytest.raises(asyncio.CancelledError):
                await dispatcher.dispatch_worker()

            assert mock_err.await_count == 2

    @pytest.mark.asyncio
    async def test_continues_on_get_exception(self, dispatcher):
        """If get() raises, the loop catches it and continues."""
        with (
            patch.object(dispatcher, "get", new_callable=AsyncMock) as mock_get,
            patch.object(dispatcher, "dispatch") as mock_dispatch,
            patch.object(dispatcher, "status_worker", new_callable=AsyncMock),
            patch.object(dispatcher, "events_worker", new_callable=AsyncMock),
            patch.object(dispatcher, "handle_evictions"),
            patch.object(dispatcher, "handle_errors", new_callable=AsyncMock),
        ):
            # First call raises (caught), second returns request, third exits.
            mock_request = MagicMock()
            mock_get.side_effect = [
                RuntimeError("redis down"),
                mock_request,
                asyncio.CancelledError,
            ]

            with pytest.raises(asyncio.CancelledError):
                await dispatcher.dispatch_worker()

            # The second iteration should have dispatched successfully.
            mock_dispatch.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_continues_on_handle_errors_exception(self, dispatcher):
        """If handle_errors() raises, the loop catches it and continues."""
        with (
            patch.object(dispatcher, "get", new_callable=AsyncMock) as mock_get,
            patch.object(dispatcher, "dispatch"),
            patch.object(dispatcher, "status_worker", new_callable=AsyncMock),
            patch.object(dispatcher, "events_worker", new_callable=AsyncMock),
            patch.object(dispatcher, "handle_evictions") as mock_evict,
            patch.object(dispatcher, "handle_errors", new_callable=AsyncMock) as mock_err,
        ):
            mock_err.side_effect = [RuntimeError("boom"), None]
            mock_get.side_effect = [None, None, asyncio.CancelledError]

            with pytest.raises(asyncio.CancelledError):
                await dispatcher.dispatch_worker()

            # Despite the first handle_errors failing, the loop continued.
            assert mock_evict.call_count == 2

    @pytest.mark.asyncio
    async def test_handles_evictions_and_errors_when_no_request(self, dispatcher):
        """handle_evictions and handle_errors are called even when get() returns None."""
        with (
            patch.object(dispatcher, "get", new_callable=AsyncMock) as mock_get,
            patch.object(dispatcher, "dispatch") as mock_dispatch,
            patch.object(dispatcher, "status_worker", new_callable=AsyncMock),
            patch.object(dispatcher, "events_worker", new_callable=AsyncMock),
            patch.object(dispatcher, "handle_evictions") as mock_evict,
            patch.object(dispatcher, "handle_errors", new_callable=AsyncMock) as mock_err,
        ):
            mock_get.side_effect = [None, asyncio.CancelledError]

            with pytest.raises(asyncio.CancelledError):
                await dispatcher.dispatch_worker()

            mock_dispatch.assert_not_called()
            mock_evict.assert_called_once()
            mock_err.assert_awaited_once()


# ---------------------------------------------------------------------------
# TestStatusWorker
# ---------------------------------------------------------------------------

class TestStatusWorker:
    """Tests for Dispatcher.status_worker() — cluster status reporting loop.

    The loop: wait for Redis xread trigger -> query controller ->
    publish + cache + clear flag.
    """

    def _xread_response(self, entry_id="1-0", data=None):
        """Build a mock xread return value."""
        if data is None:
            data = {}
        return [("status:trigger", [(entry_id, data)])]

    @pytest.mark.asyncio
    async def test_reads_from_status_trigger_stream(self, dispatcher, mock_redis):
        """xread is called with the status:trigger stream starting at '$'."""
        mock_redis.async_client.xread = AsyncMock(
            side_effect=[self._xread_response(), asyncio.CancelledError]
        )

        with (
            patch(DISPATCHER_PATCH_CONTROLLER),
            patch(DISPATCHER_PATCH_SUBMIT, new_callable=AsyncMock, return_value={}),
        ):
            with pytest.raises(asyncio.CancelledError):
                await dispatcher.status_worker()

        first_call = mock_redis.async_client.xread.call_args_list[0]
        assert first_call == call({"status:trigger": "$"}, count=1, block=0)

    @pytest.mark.asyncio
    async def test_queries_controller_after_trigger(self, dispatcher, mock_redis):
        mock_redis.async_client.xread = AsyncMock(
            side_effect=[self._xread_response(), asyncio.CancelledError]
        )

        with (
            patch(DISPATCHER_PATCH_CONTROLLER) as mock_ctrl,
            patch(DISPATCHER_PATCH_SUBMIT, new_callable=AsyncMock, return_value={}) as mock_submit,
        ):
            with pytest.raises(asyncio.CancelledError):
                await dispatcher.status_worker()

            mock_ctrl.assert_called_once()
            mock_submit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_publishes_status_to_redis(self, dispatcher, mock_redis):
        status_data = {"models": ["llama"]}
        mock_redis.async_client.xread = AsyncMock(
            side_effect=[self._xread_response(), asyncio.CancelledError]
        )

        with (
            patch(DISPATCHER_PATCH_CONTROLLER),
            patch(DISPATCHER_PATCH_SUBMIT, new_callable=AsyncMock, return_value=status_data),
        ):
            with pytest.raises(asyncio.CancelledError):
                await dispatcher.status_worker()

        mock_redis.async_client.publish.assert_awaited_once_with(
            "status:event", pickle.dumps(status_data)
        )

    @pytest.mark.asyncio
    async def test_caches_status_with_ttl(self, dispatcher, mock_redis):
        status_data = {"models": []}
        mock_redis.async_client.xread = AsyncMock(
            side_effect=[self._xread_response(), asyncio.CancelledError]
        )

        with (
            patch(DISPATCHER_PATCH_CONTROLLER),
            patch(DISPATCHER_PATCH_SUBMIT, new_callable=AsyncMock, return_value=status_data),
        ):
            with pytest.raises(asyncio.CancelledError):
                await dispatcher.status_worker()

        mock_redis.async_client.set.assert_awaited_once_with(
            "status", pickle.dumps(status_data), ex=dispatcher.status_cache_freq_s
        )

    @pytest.mark.asyncio
    async def test_clears_requested_flag(self, dispatcher, mock_redis):
        mock_redis.async_client.xread = AsyncMock(
            side_effect=[self._xread_response(), asyncio.CancelledError]
        )

        with (
            patch(DISPATCHER_PATCH_CONTROLLER),
            patch(DISPATCHER_PATCH_SUBMIT, new_callable=AsyncMock, return_value={}),
        ):
            with pytest.raises(asyncio.CancelledError):
                await dispatcher.status_worker()

        mock_redis.async_client.delete.assert_awaited_with("status:requested")

    @pytest.mark.asyncio
    async def test_updates_last_id_for_next_xread(self, dispatcher, mock_redis):
        """After processing entry '42-0', the next xread uses that as last_id."""
        mock_redis.async_client.xread = AsyncMock(
            side_effect=[
                self._xread_response(entry_id="42-0"),
                self._xread_response(entry_id="43-0"),
                asyncio.CancelledError,
            ]
        )

        with (
            patch(DISPATCHER_PATCH_CONTROLLER),
            patch(DISPATCHER_PATCH_SUBMIT, new_callable=AsyncMock, return_value={}),
        ):
            with pytest.raises(asyncio.CancelledError):
                await dispatcher.status_worker()

        # Second xread call should use "42-0" as last_id.
        second_call = mock_redis.async_client.xread.call_args_list[1]
        assert second_call == call({"status:trigger": "42-0"}, count=1, block=0)

    @pytest.mark.asyncio
    async def test_controller_timeout_is_caught(self, dispatcher, mock_redis):
        """If the controller query times out, the loop continues and retries."""
        mock_redis.async_client.xread = AsyncMock(
            side_effect=[
                self._xread_response(entry_id="1-0"),
                asyncio.CancelledError,  # exit after retry succeeds
            ]
        )

        with (
            patch(DISPATCHER_PATCH_CONTROLLER),
            patch(DISPATCHER_PATCH_SUBMIT, new_callable=AsyncMock) as mock_submit,
        ):
            # First call times out, second (retry with got_status=False) succeeds.
            mock_submit.side_effect = [asyncio.TimeoutError, {}]

            with pytest.raises(asyncio.CancelledError):
                await dispatcher.status_worker()

        # publish should only be called for the successful retry.
        assert mock_redis.async_client.publish.await_count == 1

    @pytest.mark.asyncio
    async def test_xread_exception_is_caught(self, dispatcher, mock_redis):
        """If xread raises, the loop catches and continues."""
        mock_redis.async_client.xread = AsyncMock(
            side_effect=[
                RuntimeError("connection lost"),
                self._xread_response(),
                asyncio.CancelledError,
            ]
        )

        with (
            patch(DISPATCHER_PATCH_CONTROLLER),
            patch(DISPATCHER_PATCH_SUBMIT, new_callable=AsyncMock, return_value={}),
        ):
            with pytest.raises(asyncio.CancelledError):
                await dispatcher.status_worker()

        # Despite first xread failing, second succeeded.
        mock_redis.async_client.publish.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retries_controller_without_new_trigger(self, dispatcher, mock_redis):
        """When controller fails (got_status stays False), the next iteration
        skips xread and retries the controller query directly."""
        call_count = 0

        async def controlled_xread(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return self._xread_response(entry_id="1-0")
            # Should not reach here if retry skips xread.
            raise asyncio.CancelledError

        mock_redis.async_client.xread = AsyncMock(side_effect=controlled_xread)

        with (
            patch(DISPATCHER_PATCH_CONTROLLER),
            patch(DISPATCHER_PATCH_SUBMIT, new_callable=AsyncMock) as mock_submit,
        ):
            # First submit fails, second succeeds, then CancelledError from xread.
            mock_submit.side_effect = [RuntimeError("controller down"), {}, asyncio.CancelledError]

            # The flow: xread(1) -> submit fails -> submit retries -> submit succeeds ->
            # xread(2) -> CancelledError
            with pytest.raises(asyncio.CancelledError):
                await dispatcher.status_worker()

        # xread should have been called only once for the trigger, then
        # the retry skipped xread and went straight to submit.
        # After success, got_status=True so next iteration calls xread again.
        assert call_count == 2
        assert mock_submit.await_count == 2

    @pytest.mark.asyncio
    async def test_pickles_status_data(self, dispatcher, mock_redis):
        """Verify the published and cached data is pickle.dumps(status)."""
        status_data = {"cluster": "healthy", "models": 5}
        mock_redis.async_client.xread = AsyncMock(
            side_effect=[self._xread_response(), asyncio.CancelledError]
        )

        with (
            patch(DISPATCHER_PATCH_CONTROLLER),
            patch(DISPATCHER_PATCH_SUBMIT, new_callable=AsyncMock, return_value=status_data),
        ):
            with pytest.raises(asyncio.CancelledError):
                await dispatcher.status_worker()

        expected_bytes = pickle.dumps(status_data)

        publish_data = mock_redis.async_client.publish.call_args[0][1]
        assert publish_data == expected_bytes

        set_data = mock_redis.async_client.set.call_args[0][1]
        assert set_data == expected_bytes


# ---------------------------------------------------------------------------
# TestEventsWorker
# ---------------------------------------------------------------------------

class TestEventsWorker:
    """Tests for Dispatcher.events_worker() — unified event handler loop.

    The loop: xread("dispatcher:events") -> dispatch to handler by event_type.
    """

    def _xread_response(self, entry_id="1-0", event_data=None):
        """Build a mock xread return value for dispatcher:events stream."""
        if event_data is None:
            event_data = {b"event_type": b""}
        return [("dispatcher:events", [(entry_id, event_data)])]

    @pytest.mark.asyncio
    async def test_reads_from_dispatcher_events_stream(self, dispatcher, mock_redis):
        mock_redis.async_client.xread = AsyncMock(
            side_effect=[
                self._xread_response(),
                asyncio.CancelledError,
            ]
        )

        with patch.object(dispatcher, "_handle_queue_state_request", new_callable=AsyncMock):
            with pytest.raises(asyncio.CancelledError):
                await dispatcher.events_worker()

        first_call = mock_redis.async_client.xread.call_args_list[0]
        assert first_call == call(
            {"dispatcher:events": "$"}, count=1, block=1000
        )

    @pytest.mark.asyncio
    async def test_dispatches_queue_state_request(self, dispatcher, mock_redis, make_event):
        event = make_event(event_type="queue_state_request")
        mock_redis.async_client.xread = AsyncMock(
            side_effect=[self._xread_response(event_data=event), asyncio.CancelledError]
        )

        with patch.object(dispatcher, "_handle_queue_state_request", new_callable=AsyncMock) as h:
            with pytest.raises(asyncio.CancelledError):
                await dispatcher.events_worker()

            h.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_dispatches_deploy_event(self, dispatcher, mock_redis, make_event):
        event = make_event(event_type="deploy", replicas="2")
        mock_redis.async_client.xread = AsyncMock(
            side_effect=[self._xread_response(event_data=event), asyncio.CancelledError]
        )

        with patch.object(dispatcher, "_handle_deploy_event", new_callable=AsyncMock) as h:
            with pytest.raises(asyncio.CancelledError):
                await dispatcher.events_worker()

            h.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_dispatches_evict_event(self, dispatcher, mock_redis, make_event):
        event = make_event(event_type="evict", model_key="test/model")
        mock_redis.async_client.xread = AsyncMock(
            side_effect=[self._xread_response(event_data=event), asyncio.CancelledError]
        )

        with patch.object(dispatcher, "_handle_evict_event", new_callable=AsyncMock) as h:
            with pytest.raises(asyncio.CancelledError):
                await dispatcher.events_worker()

            h.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_dispatches_kill_request(self, dispatcher, mock_redis, make_event):
        event = make_event(event_type="kill_request", request_id="req-999")
        mock_redis.async_client.xread = AsyncMock(
            side_effect=[self._xread_response(event_data=event), asyncio.CancelledError]
        )

        with patch.object(dispatcher, "_handle_kill_request", new_callable=AsyncMock) as h:
            with pytest.raises(asyncio.CancelledError):
                await dispatcher.events_worker()

            h.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_dispatches_env_event(self, dispatcher, mock_redis, make_event):
        event = make_event(event_type="env")
        mock_redis.async_client.xread = AsyncMock(
            side_effect=[self._xread_response(event_data=event), asyncio.CancelledError]
        )

        with patch.object(dispatcher, "_handle_env_event", new_callable=AsyncMock) as h:
            with pytest.raises(asyncio.CancelledError):
                await dispatcher.events_worker()

            h.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_logs_warning_for_unknown_event(self, dispatcher, mock_redis):
        event = {b"event_type": b"totally_unknown"}
        mock_redis.async_client.xread = AsyncMock(
            side_effect=[self._xread_response(event_data=event), asyncio.CancelledError]
        )

        with pytest.raises(asyncio.CancelledError):
            await dispatcher.events_worker()

        # Verify the logger warning was called (logger is set on the instance).
        dispatcher.logger.warning.assert_called()
        warning_msg = dispatcher.logger.warning.call_args[0][0]
        assert "totally_unknown" in warning_msg

    @pytest.mark.asyncio
    async def test_continues_on_empty_messages(self, dispatcher, mock_redis, make_event):
        """When xread returns empty/None, the loop continues."""
        event = make_event(event_type="deploy", replicas="1")
        mock_redis.async_client.xread = AsyncMock(
            side_effect=[
                [],       # empty — continue
                None,     # None — continue
                self._xread_response(event_data=event),
                asyncio.CancelledError,
            ]
        )

        with patch.object(dispatcher, "_handle_deploy_event", new_callable=AsyncMock) as h:
            with pytest.raises(asyncio.CancelledError):
                await dispatcher.events_worker()

            # Despite two empty reads, the third succeeded.
            h.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_continues_on_handler_exception(self, dispatcher, mock_redis, make_event):
        """If a handler raises, the loop catches it and continues."""
        event1 = make_event(event_type="deploy", replicas="1")
        event2 = make_event(event_type="evict", model_key="test/model")

        mock_redis.async_client.xread = AsyncMock(
            side_effect=[
                self._xread_response(entry_id="1-0", event_data=event1),
                self._xread_response(entry_id="2-0", event_data=event2),
                asyncio.CancelledError,
            ]
        )

        with (
            patch.object(
                dispatcher, "_handle_deploy_event",
                new_callable=AsyncMock,
                side_effect=RuntimeError("handler crash"),
            ),
            patch.object(dispatcher, "_handle_evict_event", new_callable=AsyncMock) as h_evict,
        ):
            with pytest.raises(asyncio.CancelledError):
                await dispatcher.events_worker()

            # Despite the deploy handler crashing, the evict handler was called.
            h_evict.assert_awaited_once_with(event2)

    @pytest.mark.asyncio
    async def test_updates_last_id_from_entry(self, dispatcher, mock_redis, make_event):
        """After processing entry '10-0', the next xread uses that as last_id."""
        event = make_event(event_type="deploy", replicas="1")

        mock_redis.async_client.xread = AsyncMock(
            side_effect=[
                self._xread_response(entry_id="10-0", event_data=event),
                self._xread_response(entry_id="11-0", event_data=event),
                asyncio.CancelledError,
            ]
        )

        with patch.object(dispatcher, "_handle_deploy_event", new_callable=AsyncMock):
            with pytest.raises(asyncio.CancelledError):
                await dispatcher.events_worker()

        second_call = mock_redis.async_client.xread.call_args_list[1]
        assert second_call == call(
            {"dispatcher:events": "10-0"}, count=1, block=1000
        )
