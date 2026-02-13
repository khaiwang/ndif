"""Unit tests for the Processor async worker loops.

Tests cover _execute_on_replica, _replica_worker, _start_replica_worker,
_initialize_and_start_replica, initialize, reply_worker, and processor_worker.
All Ray and external dependencies are mocked.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.api.src.queue.processor import (
    Processor,
    ProcessorStatus,
)
from src.common.schema import BackendResponseModel


# ======================================================================
# Helpers
# ======================================================================

UTIL_PREFIX = "src.services.api.src.queue.processor"
REPLICAS_MOD = "src.services.api.src.queue.replicas"


# ======================================================================
# _execute_on_replica()
# ======================================================================


class TestExecuteOnReplica:
    """Test _execute_on_replica() single request execution on a Ray actor."""

    @pytest.mark.asyncio
    @patch(f"{REPLICAS_MOD}.submit", new_callable=AsyncMock)
    @patch(f"{REPLICAS_MOD}.get_model_actor_handle")
    async def test_sends_dispatched_response(
        self, mock_get_handle, mock_submit, make_processor, make_request
    ):
        mock_get_handle.return_value = MagicMock()
        mock_submit.return_value = None

        p = make_processor()
        p.replicas.add("r1")
        req = make_request()

        await p._execute_on_replica("r1", req)

        assert req.create_response.call_count == 1
        call_args = req.create_response.call_args
        assert call_args[0][0] == BackendResponseModel.JobStatus.DISPATCHED

    @pytest.mark.asyncio
    @patch(f"{REPLICAS_MOD}.submit", new_callable=AsyncMock)
    @patch(f"{REPLICAS_MOD}.get_model_actor_handle")
    async def test_calls_replica_submit_with_request(
        self, mock_get_handle, mock_submit, make_processor, make_request
    ):
        mock_get_handle.return_value = MagicMock()
        mock_submit.return_value = None

        p = make_processor()
        p.replicas.add("r1")
        req = make_request()

        await p._execute_on_replica("r1", req)

        mock_submit.assert_awaited_once()
        call_args = mock_submit.call_args
        assert call_args[0][1] == "__call__"
        assert call_args[0][2] is req

    @pytest.mark.asyncio
    @patch(f"{REPLICAS_MOD}.submit", new_callable=AsyncMock)
    @patch(f"{REPLICAS_MOD}.get_model_actor_handle")
    async def test_sets_and_clears_current_request(
        self, mock_get_handle, mock_submit, make_processor, make_request
    ):
        mock_get_handle.return_value = MagicMock()

        request_id_during_submit = []

        async def capture(*args, **kwargs):
            request_id_during_submit.append(
                p.replicas.get("r1").current_request_id
            )

        mock_submit.side_effect = capture

        p = make_processor()
        p.replicas.add("r1")
        req = make_request(request_id="req-123")

        await p._execute_on_replica("r1", req)

        assert request_id_during_submit == ["req-123"]
        assert p.replicas.get("r1").current_request_id is None

    @pytest.mark.asyncio
    @patch(f"{REPLICAS_MOD}.submit", new_callable=AsyncMock)
    @patch(f"{REPLICAS_MOD}.get_model_actor_handle")
    async def test_actor_not_found_triggers_eviction(
        self, mock_get_handle, mock_submit, make_processor, make_request
    ):
        mock_get_handle.return_value = MagicMock()
        mock_submit.side_effect = RuntimeError(
            "Failed to look up actor with name xyz"
        )

        p = make_processor()
        p.replicas.add("r1")
        p.status = ProcessorStatus.READY
        req = make_request()

        await p._execute_on_replica("r1", req)

        assert p.status == ProcessorStatus.CANCELLED
        eviction = p.eviction_queue.get_nowait()
        assert eviction[0] == p.model_key
        assert eviction[2] == "r1"

    @pytest.mark.asyncio
    @patch(f"{REPLICAS_MOD}.submit", new_callable=AsyncMock)
    @patch(f"{REPLICAS_MOD}.get_model_actor_handle")
    async def test_generic_error_sends_error_response(
        self, mock_get_handle, mock_submit, make_processor, make_request
    ):
        mock_get_handle.return_value = MagicMock()
        mock_submit.side_effect = RuntimeError("Something went wrong")

        p = make_processor()
        p.replicas.add("r1")
        req = make_request()

        await p._execute_on_replica("r1", req)

        assert req.create_response.call_count == 2
        error_call = req.create_response.call_args_list[1]
        assert error_call[0][0] == BackendResponseModel.JobStatus.ERROR

    @pytest.mark.asyncio
    @patch(f"{REPLICAS_MOD}.submit", new_callable=AsyncMock)
    @patch(f"{REPLICAS_MOD}.get_model_actor_handle")
    async def test_generic_error_puts_to_error_queue_not_eviction(
        self, mock_get_handle, mock_submit, make_processor, make_request
    ):
        mock_get_handle.return_value = MagicMock()
        err = RuntimeError("Something went wrong")
        mock_submit.side_effect = err

        p = make_processor()
        p.replicas.add("r1")
        req = make_request()

        await p._execute_on_replica("r1", req)

        error = p.error_queue.get_nowait()
        assert error[0] == p.model_key
        assert error[1] is err
        assert p.eviction_queue.empty()

    @pytest.mark.asyncio
    @patch(f"{REPLICAS_MOD}.submit", new_callable=AsyncMock)
    @patch(f"{REPLICAS_MOD}.get_model_actor_handle")
    async def test_clears_current_request_on_error(
        self, mock_get_handle, mock_submit, make_processor, make_request
    ):
        mock_get_handle.return_value = MagicMock()
        mock_submit.side_effect = RuntimeError("boom")

        p = make_processor()
        p.replicas.add("r1")
        req = make_request(request_id="req-err")

        await p._execute_on_replica("r1", req)

        assert p.replicas.get("r1").current_request_id is None


# ======================================================================
# _replica_worker()
# ======================================================================


class TestReplicaWorker:
    """Test _replica_worker() per-replica dequeue-execute loop."""

    @pytest.mark.asyncio
    async def test_dequeues_and_executes(self, make_processor, make_request):
        p = make_processor()
        p.replicas.add("r1")
        p.status = ProcessorStatus.READY

        req = make_request()
        p.queue.put_nowait(req)

        with patch.object(p, '_execute_on_replica', new_callable=AsyncMock) as mock_exec:
            async def stop(replica_id, request):
                p.status = ProcessorStatus.CANCELLED
            mock_exec.side_effect = stop

            await p._replica_worker("r1")

        mock_exec.assert_awaited_once_with("r1", req)

    @pytest.mark.asyncio
    async def test_sets_busy_during_execution(self, make_processor, make_request):
        p = make_processor()
        p.replicas.add("r1")
        p.status = ProcessorStatus.READY

        req = make_request()
        p.queue.put_nowait(req)

        statuses_during = []
        with patch.object(p, '_execute_on_replica', new_callable=AsyncMock) as mock_exec:
            async def capture(replica_id, request):
                statuses_during.append(p.status)
                p.status = ProcessorStatus.CANCELLED
            mock_exec.side_effect = capture

            await p._replica_worker("r1")

        assert statuses_during == [ProcessorStatus.BUSY]

    @pytest.mark.asyncio
    async def test_returns_to_ready_when_no_in_flight(self, make_processor, make_request):
        """After execute, if no other replicas busy, status goes to READY."""
        p = make_processor()
        p.replicas.add("r1")
        p.status = ProcessorStatus.READY

        req = make_request()
        p.queue.put_nowait(req)

        with patch.object(p, '_execute_on_replica', new_callable=AsyncMock):
            # Worker processes req, then blocks on empty queue
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(p._replica_worker("r1"), timeout=0.1)

        assert p.status == ProcessorStatus.READY

    @pytest.mark.asyncio
    async def test_stays_busy_when_other_replicas_in_flight(
        self, make_processor, make_request
    ):
        p = make_processor()
        p.replicas.add("r1")
        p.replicas.add("r2")
        p.replicas.begin_request("r2")  # simulate r2 executing
        p.status = ProcessorStatus.READY

        req = make_request()
        p.queue.put_nowait(req)

        with patch.object(p, '_execute_on_replica', new_callable=AsyncMock):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(p._replica_worker("r1"), timeout=0.1)

        assert p.status == ProcessorStatus.BUSY

    @pytest.mark.asyncio
    async def test_stops_when_cancelled(self, make_processor, make_request):
        p = make_processor()
        p.replicas.add("r1")
        p.status = ProcessorStatus.CANCELLED

        req = make_request()
        p.queue.put_nowait(req)

        await p._replica_worker("r1")

        # Request should NOT have been consumed
        assert p.queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_exits_when_replica_not_present(self, make_processor):
        """If replica_id not in replicas, worker exits immediately."""
        p = make_processor()
        # Don't add "r1" to replicas
        p.status = ProcessorStatus.READY

        await p._replica_worker("r1")
        # Should return without blocking

    @pytest.mark.asyncio
    async def test_calls_reply_after_dequeue(self, make_processor, make_request):
        p = make_processor()
        p.replicas.add("r1")
        p.status = ProcessorStatus.READY

        req = make_request()
        p.queue.put_nowait(req)

        with patch.object(p, '_execute_on_replica', new_callable=AsyncMock) as mock_exec, \
             patch.object(p, 'reply') as mock_reply:
            async def stop(replica_id, request):
                p.status = ProcessorStatus.CANCELLED
            mock_exec.side_effect = stop

            await p._replica_worker("r1")

        mock_reply.assert_called_once()


# ======================================================================
# _start_replica_worker()
# ======================================================================


class TestStartReplicaWorker:
    """Test _start_replica_worker() asyncio task creation."""

    @pytest.mark.asyncio
    async def test_creates_task(self, make_processor):
        p = make_processor()
        p.replicas.add("r1")

        with patch.object(p, '_replica_worker', new_callable=AsyncMock):
            p._start_replica_worker("r1")

            replica = p.replicas.get("r1")
            assert replica.worker_task is not None
            assert not replica.worker_task.done()

            replica.worker_task.cancel()
            try:
                await replica.worker_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_noop_if_already_running(self, make_processor):
        p = make_processor()
        p.replicas.add("r1")

        with patch.object(p, '_replica_worker', new_callable=AsyncMock):
            p._start_replica_worker("r1")
            original_task = p.replicas.get("r1").worker_task

            p._start_replica_worker("r1")
            assert p.replicas.get("r1").worker_task is original_task

            original_task.cancel()
            try:
                await original_task
            except asyncio.CancelledError:
                pass


# ======================================================================
# _initialize_and_start_replica()
# ======================================================================


class TestInitializeAndStartReplica:
    """Test _initialize_and_start_replica() wait-for-ready and worker start."""

    @pytest.mark.asyncio
    async def test_waits_for_ready_then_starts_worker(self, make_processor):
        """When status is READY (hot-add case), starts worker after ready."""
        from src.services.api.src.queue.replicas import Replica

        p = make_processor()
        p.replicas.add("r1")
        p.status = ProcessorStatus.READY

        with patch.object(Replica, 'wait_until_ready', new_callable=AsyncMock) as mock_wait, \
             patch.object(p, '_start_replica_worker') as mock_start:
            await p._initialize_and_start_replica("r1")

        mock_wait.assert_awaited_once()
        mock_start.assert_called_once_with("r1")

    @pytest.mark.asyncio
    async def test_no_worker_start_during_deploying(self, make_processor):
        """During normal init flow (DEPLOYING), worker is not started here."""
        from src.services.api.src.queue.replicas import Replica

        p = make_processor()
        p.replicas.add("r1")
        p.status = ProcessorStatus.DEPLOYING

        with patch.object(Replica, 'wait_until_ready', new_callable=AsyncMock), \
             patch.object(p, '_start_replica_worker') as mock_start:
            await p._initialize_and_start_replica("r1")

        mock_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_reports_eviction_and_error(self, make_processor):
        from src.services.api.src.queue.replicas import Replica

        p = make_processor()
        p.replicas.add("r1")
        p.status = ProcessorStatus.DEPLOYING
        err = RuntimeError("Model load failed")

        with patch.object(Replica, 'wait_until_ready', new_callable=AsyncMock, side_effect=err):
            await p._initialize_and_start_replica("r1")

        eviction = p.eviction_queue.get_nowait()
        assert eviction[0] == p.model_key
        assert eviction[2] == "r1"

        error = p.error_queue.get_nowait()
        assert error[0] == p.model_key
        assert error[1] is err

    @pytest.mark.asyncio
    async def test_error_removes_replica(self, make_processor):
        from src.services.api.src.queue.replicas import Replica

        p = make_processor()
        p.replicas.add("r1")
        p.replicas.add("r2")  # keep r2 so removing r1 doesn't cancel
        p.status = ProcessorStatus.DEPLOYING

        with patch.object(Replica, 'wait_until_ready', new_callable=AsyncMock,
                          side_effect=RuntimeError("fail")):
            await p._initialize_and_start_replica("r1")

        assert not p.replicas.has("r1")
        assert p.replicas.has("r2")


# ======================================================================
# initialize()
# ======================================================================


class TestInitialize:
    """Test initialize() which waits for all replicas to be ready."""

    @pytest.mark.asyncio
    async def test_all_replicas_succeed(self, make_processor):
        p = make_processor()
        p.replicas.add("r1")
        p.replicas.add("r2")
        p.status = ProcessorStatus.DEPLOYING

        with patch.object(p, '_initialize_and_start_replica', new_callable=AsyncMock):
            await p.initialize()

        assert p.status == ProcessorStatus.DEPLOYING  # unchanged on success
        assert p.replicas.has("r1")
        assert p.replicas.has("r2")

    @pytest.mark.asyncio
    async def test_no_replicas_cancels_and_purges(self, make_processor, make_request):
        p = make_processor()
        # No replicas added
        p.status = ProcessorStatus.DEPLOYING
        p.dedicated = True

        req = make_request()
        p.queue.put_nowait(req)

        await p.initialize()

        assert p.status == ProcessorStatus.CANCELLED
        # Purge sends ERROR to queued request
        req.create_response.assert_called()
        error_call = req.create_response.call_args
        assert error_call[0][0] == BackendResponseModel.JobStatus.ERROR

    @pytest.mark.asyncio
    async def test_partial_failure_continues(self, make_processor):
        """One replica fails init, the other succeeds — processor continues."""
        p = make_processor()
        p.replicas.add("r1")
        p.replicas.add("r2")
        p.status = ProcessorStatus.DEPLOYING

        async def fail_r2_only(replica_id):
            if replica_id == "r2":
                p.remove_replica(replica_id, "init failed")

        with patch.object(p, '_initialize_and_start_replica', new_callable=AsyncMock) as mock_init:
            mock_init.side_effect = fail_r2_only
            await p.initialize()

        assert p.status != ProcessorStatus.CANCELLED
        assert p.replicas.has("r1")
        assert not p.replicas.has("r2")

    @pytest.mark.asyncio
    async def test_all_fail_cancels_and_purges(self, make_processor, make_request):
        p = make_processor()
        p.replicas.add("r1")
        p.replicas.add("r2")
        p.status = ProcessorStatus.DEPLOYING
        p.dedicated = True

        req = make_request()
        p.queue.put_nowait(req)

        async def fail_all(replica_id):
            p.remove_replica(replica_id, "init failed")

        with patch.object(p, '_initialize_and_start_replica', new_callable=AsyncMock) as mock_init:
            mock_init.side_effect = fail_all
            await p.initialize()

        assert p.status == ProcessorStatus.CANCELLED
        assert p.replicas.replica_count == 0

    @pytest.mark.asyncio
    async def test_cancels_pending_tasks_on_early_cancel(self, make_processor):
        """If one init sets CANCELLED, remaining init tasks are cancelled."""
        p = make_processor()
        p.replicas.add("r1")
        p.replicas.add("r2")
        p.status = ProcessorStatus.DEPLOYING

        async def cancel_on_r1(replica_id):
            if replica_id == "r1":
                p.status = ProcessorStatus.CANCELLED
            elif replica_id == "r2":
                await asyncio.sleep(100)  # slow task that should be cancelled

        with patch.object(p, '_initialize_and_start_replica', new_callable=AsyncMock) as mock_init:
            mock_init.side_effect = cancel_on_r1
            # If r2's task weren't cancelled, this would hang for 100s
            await asyncio.wait_for(p.initialize(), timeout=5.0)

        assert p.status == ProcessorStatus.CANCELLED


# ======================================================================
# reply_worker()
# ======================================================================


class TestReplyWorker:
    """Test reply_worker() periodic status updates to queued users."""

    @pytest.mark.asyncio
    @patch(f"{UTIL_PREFIX}.asyncio.sleep", new_callable=AsyncMock)
    async def test_sends_provisioning_message(self, mock_sleep, make_processor):
        p = make_processor()
        p.status = ProcessorStatus.PROVISIONING

        async def stop_after_sleep(seconds):
            p.status = ProcessorStatus.READY
        mock_sleep.side_effect = stop_after_sleep

        with patch.object(p, 'reply') as mock_reply:
            await p.reply_worker()

        mock_reply.assert_called_with("Model Provisioning...")

    @pytest.mark.asyncio
    @patch(f"{UTIL_PREFIX}.asyncio.sleep", new_callable=AsyncMock)
    async def test_sends_deploying_message(self, mock_sleep, make_processor):
        p = make_processor()
        p.status = ProcessorStatus.DEPLOYING

        async def stop_after_sleep(seconds):
            p.status = ProcessorStatus.READY
        mock_sleep.side_effect = stop_after_sleep

        with patch.object(p, 'reply') as mock_reply:
            await p.reply_worker()

        mock_reply.assert_called_with("Model Deploying...")

    @pytest.mark.asyncio
    async def test_exits_on_ready(self, make_processor):
        p = make_processor()
        p.status = ProcessorStatus.READY

        # Should exit immediately without looping
        await p.reply_worker()

    @pytest.mark.asyncio
    async def test_exits_on_cancelled(self, make_processor):
        p = make_processor()
        p.status = ProcessorStatus.CANCELLED

        await p.reply_worker()


# ======================================================================
# processor_worker()
# ======================================================================


class TestProcessorWorker:
    """Test processor_worker() main lifecycle orchestrator."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_with_provision(self, make_processor):
        """provision=True: PROVISIONING → provision → DEPLOYING → initialize → READY."""
        p = make_processor()

        statuses_seen = []

        async def capture_provision():
            statuses_seen.append(('provision', p.status))
            p.replicas.add("r1")

        async def capture_initialize():
            statuses_seen.append(('initialize', p.status))

        p.provision = AsyncMock(side_effect=capture_provision)
        p.initialize = AsyncMock(side_effect=capture_initialize)
        p.reply_worker = AsyncMock()
        p._start_replica_worker = MagicMock()

        await p.processor_worker(provision=True)

        assert statuses_seen[0] == ('provision', ProcessorStatus.PROVISIONING)
        assert statuses_seen[1] == ('initialize', ProcessorStatus.DEPLOYING)
        assert p.status == ProcessorStatus.READY

    @pytest.mark.asyncio
    async def test_skip_provision(self, make_processor):
        """provision=False: provision() not called."""
        p = make_processor()

        p.provision = AsyncMock()
        p.initialize = AsyncMock()
        p.reply_worker = AsyncMock()
        p._start_replica_worker = MagicMock()

        await p.processor_worker(provision=False)

        p.provision.assert_not_called()
        assert p.status == ProcessorStatus.READY

    @pytest.mark.asyncio
    async def test_provision_cancelled_exits_early(self, make_processor):
        p = make_processor()

        async def cancel_during_provision():
            p.status = ProcessorStatus.CANCELLED

        p.provision = AsyncMock(side_effect=cancel_during_provision)
        p.initialize = AsyncMock()
        p.reply_worker = AsyncMock()

        await p.processor_worker(provision=True)

        p.initialize.assert_not_called()
        assert p.status == ProcessorStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_initialize_cancelled_exits_early(self, make_processor):
        p = make_processor()

        async def cancel_during_initialize():
            p.status = ProcessorStatus.CANCELLED

        p.provision = AsyncMock()
        p.initialize = AsyncMock(side_effect=cancel_during_initialize)
        p.reply_worker = AsyncMock()
        p._start_replica_worker = MagicMock()

        await p.processor_worker(provision=True)

        p._start_replica_worker.assert_not_called()
        assert p.status == ProcessorStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_starts_reply_worker(self, make_processor):
        p = make_processor()

        p.provision = AsyncMock()
        p.initialize = AsyncMock()
        p.reply_worker = AsyncMock()
        p._start_replica_worker = MagicMock()

        await p.processor_worker(provision=True)
        # Give the reply_worker task a chance to run
        await asyncio.sleep(0)

        p.reply_worker.assert_called_once()

    @pytest.mark.asyncio
    async def test_starts_all_replica_workers(self, make_processor):
        p = make_processor()

        async def add_replicas():
            p.replicas.add("r1")
            p.replicas.add("r2")

        p.provision = AsyncMock(side_effect=add_replicas)
        p.initialize = AsyncMock()
        p.reply_worker = AsyncMock()
        p._start_replica_worker = MagicMock()

        await p.processor_worker(provision=True)

        assert p._start_replica_worker.call_count == 2
        p._start_replica_worker.assert_any_call("r1")
        p._start_replica_worker.assert_any_call("r2")
