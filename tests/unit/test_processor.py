"""Unit tests for the Processor class.

Tests cover enqueue logic, provisioning, replica management,
request killing, state reporting, and purge behaviour.
All Ray and external dependencies are mocked.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.api.src.queue.processor import (
    DeploymentStatus,
    Processor,
    ProcessorStatus,
)
from src.common.schema import BackendResponseModel


# ======================================================================
# Helpers
# ======================================================================

UTIL_PREFIX = "src.services.api.src.queue.processor"


def _mock_response():
    """Return a mock response object whose respond() returns None."""
    resp = MagicMock()
    resp.respond.return_value = None
    return resp


# ======================================================================
# Constructor
# ======================================================================


class TestProcessorInit:
    """Test Processor constructor defaults."""

    def test_default_values(self, make_processor):
        p = make_processor()
        assert p.model_key == "meta-llama/Llama-2-7b"
        assert p.status == ProcessorStatus.UNINITIALIZED
        assert p.status_changed_at == 0
        assert p.dedicated is None
        assert p.requested_replica_count == 1
        assert p.replicas.replica_count == 0
        assert p.queue.empty()

    def test_explicit_replica_count(self, make_processor):
        p = make_processor(replica_count=4)
        assert p.requested_replica_count == 4

    def test_replica_count_minimum_is_one(self, make_processor):
        p = make_processor(replica_count=0)
        assert p.requested_replica_count == 1

        p2 = make_processor(replica_count=-5)
        assert p2.requested_replica_count == 1


# ======================================================================
# Status property
# ======================================================================


class TestProcessorStatus:
    """Test status getter/setter and timestamp update."""

    def test_status_setter_records_timestamp(self, make_processor):
        p = make_processor()
        assert p.status_changed_at == 0
        p.status = ProcessorStatus.PROVISIONING
        assert p.status == ProcessorStatus.PROVISIONING
        assert p.status_changed_at > 0

    def test_status_transitions(self, make_processor):
        p = make_processor()
        for s in ProcessorStatus:
            p.status = s
            assert p.status is s


# ======================================================================
# enqueue()
# ======================================================================


class TestEnqueue:
    """Test Processor.enqueue() acceptance/rejection logic."""

    def test_enqueue_dedicated_model(self, make_processor, make_request):
        p = make_processor()
        p.dedicated = True
        req = make_request(hotswapping=False)

        p.enqueue(req)
        assert p.queue.qsize() == 1
        req.create_response.assert_called()

    def test_enqueue_hotswap_on_non_dedicated(self, make_processor, make_request):
        p = make_processor()
        p.dedicated = False
        req = make_request(hotswapping=True)

        p.enqueue(req)
        assert p.queue.qsize() == 1

    def test_enqueue_non_hotswap_on_non_dedicated_rejected(self, make_processor, make_request):
        p = make_processor()
        p.dedicated = False
        req = make_request(hotswapping=False)

        p.enqueue(req)
        assert p.queue.qsize() == 0
        # Should get ERROR response
        req.create_response.assert_called_once()
        call_args = req.create_response.call_args
        assert call_args[0][0] == BackendResponseModel.JobStatus.ERROR

    def test_enqueue_when_dedicated_is_none(self, make_processor, make_request):
        """When dedicated is None (not yet determined), non-hotswap should be accepted."""
        p = make_processor()
        p.dedicated = None
        req = make_request(hotswapping=False)

        p.enqueue(req)
        # dedicated is None, which is not `False`, so it should pass
        assert p.queue.qsize() == 1

    def test_enqueue_multiple_requests_increments_queue(self, make_processor, make_request):
        p = make_processor()
        p.dedicated = True
        for i in range(5):
            p.enqueue(make_request(request_id=f"req-{i}"))
        assert p.queue.qsize() == 5

    def test_enqueue_sends_queued_response(self, make_processor, make_request):
        p = make_processor()
        p.dedicated = True
        req = make_request()

        p.enqueue(req)

        req.create_response.assert_called_once()
        call_args = req.create_response.call_args
        assert call_args[0][0] == BackendResponseModel.JobStatus.QUEUED
        req.create_response.return_value.respond.assert_called_once()


# ======================================================================
# check_dedicated()
# ======================================================================


class TestCheckDedicated:
    """Test check_dedicated() controller query logic."""

    @pytest.mark.asyncio
    @patch(f"{UTIL_PREFIX}.submit", new_callable=AsyncMock)
    async def test_returns_true_when_dedicated_deployment_exists(self, mock_submit, make_processor):
        mock_submit.return_value = {
            "replica-1": {"dedicated": True, "model_key": "m"},
            "replica-2": {"dedicated": False},
        }
        p = make_processor()
        handle = MagicMock()
        result = await p.check_dedicated(handle)

        assert result is True
        mock_submit.assert_awaited_once_with(handle, "get_deployment", p.model_key)

    @pytest.mark.asyncio
    @patch(f"{UTIL_PREFIX}.submit", new_callable=AsyncMock)
    async def test_returns_false_when_no_dedicated(self, mock_submit, make_processor):
        mock_submit.return_value = {
            "replica-1": {"dedicated": False},
        }
        p = make_processor()
        result = await p.check_dedicated(MagicMock())
        assert result is False

    @pytest.mark.asyncio
    @patch(f"{UTIL_PREFIX}.submit", new_callable=AsyncMock)
    async def test_returns_false_when_none(self, mock_submit, make_processor):
        mock_submit.return_value = None
        p = make_processor()
        result = await p.check_dedicated(MagicMock())
        assert result is False

    @pytest.mark.asyncio
    @patch(f"{UTIL_PREFIX}.submit", new_callable=AsyncMock)
    async def test_returns_false_when_not_dict(self, mock_submit, make_processor):
        mock_submit.return_value = "some_string"
        p = make_processor()
        result = await p.check_dedicated(MagicMock())
        assert result is False

    @pytest.mark.asyncio
    @patch(f"{UTIL_PREFIX}.submit", new_callable=AsyncMock)
    async def test_returns_false_when_not_found(self, mock_submit, make_processor):
        mock_submit.return_value = {"deployments_state": "not_found"}
        p = make_processor()
        result = await p.check_dedicated(MagicMock())
        assert result is False

    @pytest.mark.asyncio
    @patch(f"{UTIL_PREFIX}.submit", new_callable=AsyncMock)
    async def test_skips_non_dict_deployment_values(self, mock_submit, make_processor):
        mock_submit.return_value = {
            "replica-1": "not_a_dict",
            "replica-2": {"dedicated": True},
        }
        p = make_processor()
        result = await p.check_dedicated(MagicMock())
        assert result is True

    @pytest.mark.asyncio
    @patch(f"{UTIL_PREFIX}.submit", new_callable=AsyncMock)
    async def test_returns_false_when_all_non_dict_values(self, mock_submit, make_processor):
        mock_submit.return_value = {
            "replica-1": "not_a_dict",
            "replica-2": 42,
        }
        p = make_processor()
        result = await p.check_dedicated(MagicMock())
        assert result is False


# ======================================================================
# provision()
# ======================================================================


class TestProvision:
    """Test provision() deployment lifecycle."""

    @pytest.mark.asyncio
    @patch(f"{UTIL_PREFIX}.submit", new_callable=AsyncMock)
    @patch(f"{UTIL_PREFIX}.controller_handle")
    async def test_provision_deployed_sets_replica_ids(
        self, mock_ctrl_handle, mock_submit, make_processor, make_request
    ):
        model_key = "meta-llama/Llama-2-7b"
        mock_ctrl_handle.return_value = MagicMock()

        # check_dedicated returns dedicated model
        # deploy returns deployed status
        mock_submit.side_effect = [
            # check_dedicated call
            {"replica-1": {"dedicated": True}},
            # deploy call
            {
                "result": {(model_key, "r1"): "deployed"},
                "evictions": [],
            },
        ]

        p = make_processor()
        p.enqueue(make_request())

        await p.provision()

        assert p.dedicated is True
        assert p.replicas.has("r1")
        assert p.replicas.replica_count == 1

    @pytest.mark.asyncio
    @patch(f"{UTIL_PREFIX}.submit", new_callable=AsyncMock)
    @patch(f"{UTIL_PREFIX}.controller_handle")
    async def test_provision_reports_evictions(
        self, mock_ctrl_handle, mock_submit, make_processor, make_request
    ):
        model_key = "meta-llama/Llama-2-7b"
        mock_ctrl_handle.return_value = MagicMock()

        mock_submit.side_effect = [
            {"replica-1": {"dedicated": True}},
            {
                "result": {(model_key, "r1"): "deployed"},
                "evictions": [("other-model", "evicted-r1")],
            },
        ]

        p = make_processor()
        p.enqueue(make_request())
        await p.provision()

        eviction = p.eviction_queue.get_nowait()
        assert eviction[0] == "other-model"
        assert eviction[2] == "evicted-r1"

    @pytest.mark.asyncio
    @patch(f"{UTIL_PREFIX}.submit", new_callable=AsyncMock)
    @patch(f"{UTIL_PREFIX}.controller_handle")
    async def test_provision_cant_accommodate_removes_replica(
        self, mock_ctrl_handle, mock_submit, make_processor, make_request
    ):
        model_key = "meta-llama/Llama-2-7b"
        mock_ctrl_handle.return_value = MagicMock()

        mock_submit.side_effect = [
            {"r1": {"dedicated": True}},
            {
                "result": {(model_key, "r1"): "cant_accommodate"},
                "evictions": [],
            },
        ]

        p = make_processor()
        p.enqueue(make_request())
        await p.provision()

        # Only replica removed -> CANCELLED
        assert p.status == ProcessorStatus.CANCELLED
        assert p.replicas.replica_count == 0

    @pytest.mark.asyncio
    @patch(f"{UTIL_PREFIX}.submit", new_callable=AsyncMock)
    @patch(f"{UTIL_PREFIX}.controller_handle")
    async def test_provision_invalid_status_removes_replica(
        self, mock_ctrl_handle, mock_submit, make_processor, make_request
    ):
        model_key = "meta-llama/Llama-2-7b"
        mock_ctrl_handle.return_value = MagicMock()

        mock_submit.side_effect = [
            {"r1": {"dedicated": True}},
            {
                "result": {(model_key, "r1"): "totally_invalid_status"},
                "evictions": [],
            },
        ]

        p = make_processor()
        p.enqueue(make_request())
        await p.provision()

        assert p.status == ProcessorStatus.CANCELLED
        # Eviction queue should have the error message
        eviction = p.eviction_queue.get_nowait()
        assert "totally_invalid_status" in eviction[1]

    @pytest.mark.asyncio
    @patch(f"{UTIL_PREFIX}.submit", new_callable=AsyncMock)
    @patch(f"{UTIL_PREFIX}.controller_handle")
    async def test_provision_non_dedicated_filters_non_hotswap(
        self, mock_ctrl_handle, mock_submit, make_processor, make_request
    ):
        model_key = "meta-llama/Llama-2-7b"
        mock_ctrl_handle.return_value = MagicMock()

        # check_dedicated returns False
        # deploy called because a hotswap request exists
        mock_submit.side_effect = [
            {},  # no dedicated deployments -> False
            {
                "result": {(model_key, "r1"): "deployed"},
                "evictions": [],
            },
        ]

        p = make_processor()
        # Enqueue one hotswap and one non-hotswap before provision
        # (dedicated is None at enqueue time, so both are accepted)
        req_hotswap = make_request(request_id="hot-1", hotswapping=True)
        req_normal = make_request(request_id="norm-1", hotswapping=False)
        p.queue.put_nowait(req_hotswap)
        p.queue.put_nowait(req_normal)

        await p.provision()

        assert p.dedicated is False
        # Non-hotswap should have been removed and given ERROR
        req_normal.create_response.assert_called()
        # Queue should only have the hotswap request
        assert p.queue.qsize() == 1
        remaining = p.queue.get_nowait()
        assert remaining.id == "hot-1"

    @pytest.mark.asyncio
    @patch(f"{UTIL_PREFIX}.submit", new_callable=AsyncMock)
    @patch(f"{UTIL_PREFIX}.controller_handle")
    async def test_provision_non_dedicated_no_hotswap_cancels(
        self, mock_ctrl_handle, mock_submit, make_processor, make_request
    ):
        mock_ctrl_handle.return_value = MagicMock()

        # check_dedicated returns False
        mock_submit.return_value = {}

        p = make_processor()
        # Only non-hotswap requests
        req = make_request(hotswapping=False)
        p.queue.put_nowait(req)

        await p.provision()

        assert p.status == ProcessorStatus.CANCELLED
        # Eviction reported
        eviction = p.eviction_queue.get_nowait()
        assert eviction[0] == p.model_key

    @pytest.mark.asyncio
    @patch(f"{UTIL_PREFIX}.submit", new_callable=AsyncMock)
    @patch(f"{UTIL_PREFIX}.controller_handle")
    async def test_provision_exception_sets_cancelled(
        self, mock_ctrl_handle, mock_submit, make_processor, make_request
    ):
        mock_ctrl_handle.side_effect = RuntimeError("Controller unavailable")

        p = make_processor()
        p.enqueue(make_request())
        await p.provision()

        assert p.status == ProcessorStatus.CANCELLED
        # Error queue should have the exception
        err = p.error_queue.get_nowait()
        assert err[0] == p.model_key
        assert isinstance(err[1], RuntimeError)

    @pytest.mark.asyncio
    @patch(f"{UTIL_PREFIX}.submit", new_callable=AsyncMock)
    @patch(f"{UTIL_PREFIX}.controller_handle")
    async def test_provision_multiple_replicas(
        self, mock_ctrl_handle, mock_submit, make_processor, make_request
    ):
        model_key = "meta-llama/Llama-2-7b"
        mock_ctrl_handle.return_value = MagicMock()

        mock_submit.side_effect = [
            {"r1": {"dedicated": True}},
            {
                "result": {
                    (model_key, "r1"): "deployed",
                    (model_key, "r2"): "deployed",
                    (model_key, "r3"): "cached_and_free",
                },
                "evictions": [],
            },
        ]

        p = make_processor(replica_count=3)
        p.enqueue(make_request())
        await p.provision()

        assert p.replicas.replica_count == 3
        assert p.replicas.has("r1")
        assert p.replicas.has("r2")
        assert p.replicas.has("r3")

    @pytest.mark.asyncio
    @patch(f"{UTIL_PREFIX}.submit", new_callable=AsyncMock)
    @patch(f"{UTIL_PREFIX}.controller_handle")
    async def test_provision_mixed_statuses_partial_removal(
        self, mock_ctrl_handle, mock_submit, make_processor, make_request
    ):
        """One replica deployed, another cant_accommodate."""
        model_key = "meta-llama/Llama-2-7b"
        mock_ctrl_handle.return_value = MagicMock()

        mock_submit.side_effect = [
            {"r1": {"dedicated": True}},
            {
                "result": {
                    (model_key, "r1"): "deployed",
                    (model_key, "r2"): "cant_accommodate",
                },
                "evictions": [],
            },
        ]

        p = make_processor(replica_count=2)
        p.enqueue(make_request())
        await p.provision()

        # r2 removed, r1 still there
        assert p.replicas.has("r1")
        assert not p.replicas.has("r2")
        assert p.replicas.replica_count == 1
        # Status should NOT be CANCELLED because r1 is still alive
        assert p.status != ProcessorStatus.CANCELLED

    @pytest.mark.asyncio
    @patch(f"{UTIL_PREFIX}.submit", new_callable=AsyncMock)
    @patch(f"{UTIL_PREFIX}.controller_handle")
    async def test_provision_filters_other_model_keys(
        self, mock_ctrl_handle, mock_submit, make_processor, make_request
    ):
        """Deployment results for other model keys should be ignored for replica_ids."""
        model_key = "meta-llama/Llama-2-7b"
        mock_ctrl_handle.return_value = MagicMock()

        mock_submit.side_effect = [
            {"r1": {"dedicated": True}},
            {
                "result": {
                    (model_key, "r1"): "deployed",
                    ("other-model", "r-other"): "deployed",
                },
                "evictions": [],
            },
        ]

        p = make_processor()
        p.enqueue(make_request())
        await p.provision()

        # Only the replica for our model_key should be registered
        assert p.replicas.has("r1")
        assert not p.replicas.has("r-other")


# ======================================================================
# remove_replica()
# ======================================================================


class TestRemoveReplica:
    """Test remove_replica() including last-replica cancellation."""

    def test_remove_existing_replica(self, make_processor):
        p = make_processor()
        p.replicas.add("r1")
        p.replicas.add("r2")
        p.status = ProcessorStatus.READY

        p.remove_replica("r1", "gone")
        assert not p.replicas.has("r1")
        assert p.replicas.replica_count == 1

    def test_remove_last_replica_cancels_and_purges(self, make_processor, make_request):
        p = make_processor()
        p.replicas.add("r1")
        p.dedicated = True
        p.status = ProcessorStatus.READY

        # Add a request to the queue so purge has something to process
        req = make_request()
        p.enqueue(req)

        p.remove_replica("r1", "evicted")

        assert p.status == ProcessorStatus.CANCELLED
        assert p.replicas.replica_count == 0
        # Purge should have sent ERROR to queued request
        # The request should have received an ERROR via purge -> reply
        # create_response is called once by enqueue (QUEUED) and once by purge (ERROR)
        assert req.create_response.call_count == 2

    def test_remove_nonexistent_replica_does_nothing(self, make_processor):
        p = make_processor()
        p.replicas.add("r1")
        p.status = ProcessorStatus.READY

        # Should not raise, just log error
        p.remove_replica("r-nonexistent", "msg")
        assert p.replicas.replica_count == 1

    def test_remove_replica_sets_ready_when_no_in_flight(self, make_processor):
        p = make_processor()
        p.replicas.add("r1")
        p.replicas.add("r2")
        p.status = ProcessorStatus.BUSY
        # No in-flight requests
        assert p.replicas.in_flight == 0

        p.remove_replica("r1", "gone")
        assert p.status == ProcessorStatus.READY


# ======================================================================
# add_replica()
# ======================================================================


class TestAddReplica:
    """Test add_replica() and task spawning."""

    def test_add_new_replica(self, make_processor):
        p = make_processor()
        p.add_replica("r1")
        assert p.replicas.has("r1")
        assert p.replicas.replica_count == 1

    def test_add_duplicate_replica_does_nothing(self, make_processor):
        p = make_processor()
        p.replicas.add("r1")
        initial_count = p.replicas.replica_count

        p.add_replica("r1")
        assert p.replicas.replica_count == initial_count

    @patch(f"{UTIL_PREFIX}.Processor._initialize_and_start_replica", new_callable=AsyncMock)
    def test_add_replica_starts_worker_when_ready(self, mock_init, make_processor):
        """When status is READY, adding a replica should attempt to start a worker."""
        p = make_processor()
        p.status = ProcessorStatus.READY

        # We need a running event loop for create_task
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run the synchronous add_replica in a coroutine context so create_task works
            async def _do():
                p.add_replica("r1")
                # Give the created task a chance to start
                await asyncio.sleep(0)

            loop.run_until_complete(_do())
        finally:
            loop.close()

        assert p.replicas.has("r1")

    def test_add_replica_does_not_start_worker_when_provisioning(self, make_processor):
        p = make_processor()
        p.status = ProcessorStatus.PROVISIONING
        p.add_replica("r1")
        # Worker task should NOT be created
        assert p.replicas.get("r1").worker_task is None


# ======================================================================
# kill_request()
# ======================================================================


class TestKillRequest:
    """Test kill_request() for various scenarios."""

    @pytest.mark.asyncio
    async def test_kill_request_in_queue(self, make_processor, make_request):
        p = make_processor()
        p.dedicated = True
        req = make_request(request_id="req-to-kill")
        p.enqueue(req)
        assert p.queue.qsize() == 1

        result = await p.kill_request("req-to-kill")

        assert result["status"] == "removed_from_queue"
        assert p.queue.qsize() == 0
        # Request should receive ERROR response for cancellation
        # Called at enqueue (QUEUED) and at kill (ERROR)
        assert req.create_response.call_count == 2

    @pytest.mark.asyncio
    @patch("src.services.api.src.queue.replicas.submit", new_callable=AsyncMock)
    @patch("src.services.api.src.queue.replicas.get_model_actor_handle")
    async def test_kill_request_in_flight(
        self, mock_get_handle, mock_submit, make_processor, make_request
    ):
        mock_get_handle.return_value = MagicMock()
        mock_submit.return_value = None  # cancel returns None

        p = make_processor()
        p.replicas.add("r1")
        p.replicas.get("r1").set_current_request("req-running")

        result = await p.kill_request("req-running")

        assert result["status"] == "cancelled_execution"
        mock_submit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_kill_request_not_found(self, make_processor):
        p = make_processor()
        result = await p.kill_request("nonexistent-req")

        assert result["status"] == "not_found"

    @pytest.mark.asyncio
    @patch("src.services.api.src.queue.replicas.submit", new_callable=AsyncMock)
    @patch("src.services.api.src.queue.replicas.get_model_actor_handle")
    async def test_kill_request_in_flight_error(
        self, mock_get_handle, mock_submit, make_processor
    ):
        mock_get_handle.return_value = MagicMock()
        mock_submit.side_effect = RuntimeError("Actor crashed")

        p = make_processor()
        p.replicas.add("r1")
        p.replicas.get("r1").set_current_request("req-err")

        result = await p.kill_request("req-err")

        assert result["status"] == "error"
        assert "Actor crashed" in result["message"]

    @pytest.mark.asyncio
    async def test_kill_request_updates_queue_positions(self, make_processor, make_request):
        """After removing from queue, remaining requests should get position updates."""
        p = make_processor()
        p.dedicated = True
        req1 = make_request(request_id="req-1")
        req2 = make_request(request_id="req-2")
        req3 = make_request(request_id="req-3")

        p.enqueue(req1)
        p.enqueue(req2)
        p.enqueue(req3)

        result = await p.kill_request("req-2")

        assert result["status"] == "removed_from_queue"
        assert p.queue.qsize() == 2


# ======================================================================
# get_state()
# ======================================================================


class TestGetState:
    """Test get_state() snapshot."""

    def test_empty_state(self, make_processor):
        p = make_processor()
        state = p.get_state()

        assert state["model_key"] == "meta-llama/Llama-2-7b"
        assert state["status"] == ProcessorStatus.UNINITIALIZED.value
        assert state["request_ids"] == []
        assert state["dedicated"] is None
        assert state["replica_count"] == 0
        assert state["in_flight"] == 0

    def test_state_with_queued_requests(self, make_processor, make_request):
        p = make_processor()
        p.dedicated = True

        req1 = make_request(request_id="r1")
        req2 = make_request(request_id="r2")
        p.enqueue(req1)
        p.enqueue(req2)

        state = p.get_state()
        assert state["request_ids"] == ["r1", "r2"]

    def test_state_reflects_status_changes(self, make_processor):
        p = make_processor()
        p.status = ProcessorStatus.READY
        state = p.get_state()

        assert state["status"] == "ready"
        assert state["status_changed_at"] > 0

    def test_state_includes_replica_info(self, make_processor):
        p = make_processor()
        p.replicas.add("r1")
        p.replicas.add("r2")
        p.replicas.get("r1").set_current_request("req-active")

        state = p.get_state()
        assert state["replica_count"] == 2
        assert state["replica_ids"] == ["r1", "r2"]
        assert state["current_request_ids"]["r1"] == "req-active"
        assert state["current_request_ids"]["r2"] is None


# ======================================================================
# purge()
# ======================================================================


class TestPurge:
    """Test purge() error broadcasting."""

    def test_purge_sends_error_to_all_queued(self, make_processor, make_request):
        p = make_processor()
        p.dedicated = True

        reqs = [make_request(request_id=f"req-{i}") for i in range(3)]
        for r in reqs:
            p.enqueue(r)

        p.purge("System shutdown")

        for r in reqs:
            # enqueue sends QUEUED, purge sends ERROR
            assert r.create_response.call_count == 2
            last_call_args = r.create_response.call_args_list[-1]
            assert last_call_args[0][0] == BackendResponseModel.JobStatus.ERROR

    def test_purge_default_message(self, make_processor, make_request):
        p = make_processor()
        p.dedicated = True
        req = make_request()
        p.enqueue(req)

        p.purge()

        last_call_args = req.create_response.call_args_list[-1]
        assert "Critical server error" in last_call_args[0][2]

    def test_purge_custom_message(self, make_processor, make_request):
        p = make_processor()
        p.dedicated = True
        req = make_request()
        p.enqueue(req)

        p.purge("Custom error message")

        last_call_args = req.create_response.call_args_list[-1]
        assert last_call_args[0][2] == "Custom error message"

    def test_purge_empty_queue_is_noop(self, make_processor):
        p = make_processor()
        # Should not raise
        p.purge("msg")


# ======================================================================
# DeploymentStatus enum
# ======================================================================


class TestDeploymentStatus:
    """Test that all expected deployment statuses exist."""

    def test_all_values(self):
        assert DeploymentStatus.DEPLOYED.value == "deployed"
        assert DeploymentStatus.CACHED_AND_FREE.value == "cached_and_free"
        assert DeploymentStatus.FREE.value == "free"
        assert DeploymentStatus.CACHED_AND_FULL.value == "cached_and_full"
        assert DeploymentStatus.FULL.value == "full"
        assert DeploymentStatus.CANT_ACCOMMODATE.value == "cant_accommodate"

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            DeploymentStatus("nonexistent")


# ======================================================================
# ProcessorStatus enum
# ======================================================================


class TestProcessorStatusEnum:
    """Test ProcessorStatus enum values."""

    def test_all_values(self):
        assert ProcessorStatus.UNINITIALIZED.value == "uninitialized"
        assert ProcessorStatus.PROVISIONING.value == "provisioning"
        assert ProcessorStatus.DEPLOYING.value == "deploying"
        assert ProcessorStatus.READY.value == "ready"
        assert ProcessorStatus.BUSY.value == "busy"
        assert ProcessorStatus.CANCELLED.value == "cancelled"
