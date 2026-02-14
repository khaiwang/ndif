"""Unit tests for Replica and Replicas classes.

Tests cover per-replica state tracking, collection management,
in-flight bookkeeping, and Ray actor interactions (mocked).
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.api.src.queue.replicas import Replica, Replicas


# ======================================================================
# Replica tests
# ======================================================================


class TestReplicaInit:
    """Test Replica constructor and default state."""

    def test_initial_attributes(self):
        """Verify a new Replica has correct defaults for all attributes."""
        r = Replica("model-a", "replica-1")
        assert r.model_key == "model-a"
        assert r.replica_id == "replica-1"
        assert r.busy is False
        assert r.worker_task is None
        assert r.current_request_id is None
        assert r.current_request_started_at is None


class TestReplicaSetClearRequest:
    """Test set_current_request / clear_current_request."""

    def test_set_current_request_populates_fields(self):
        """Verify set_current_request stores the request ID and a timestamp."""
        r = Replica("m", "r1")
        before = time.time()
        r.set_current_request("req-42")
        after = time.time()

        assert r.current_request_id == "req-42"
        assert before <= r.current_request_started_at <= after

    def test_clear_current_request_resets_fields(self):
        """Verify clear_current_request resets request ID and timestamp to None."""
        r = Replica("m", "r1")
        r.set_current_request("req-42")
        r.clear_current_request()

        assert r.current_request_id is None
        assert r.current_request_started_at is None

    def test_set_overwrites_previous_request(self):
        """Verify calling set_current_request again overwrites the previous request ID."""
        r = Replica("m", "r1")
        r.set_current_request("req-1")
        r.set_current_request("req-2")

        assert r.current_request_id == "req-2"


class TestReplicaGetHandle:
    """Test Replica.get_handle() delegates to util.get_model_actor_handle."""

    @patch("src.services.api.src.queue.replicas.get_model_actor_handle")
    def test_get_handle_calls_util(self, mock_get_handle):
        """Verify get_handle delegates to get_model_actor_handle with correct args."""
        mock_get_handle.return_value = "fake_handle"
        r = Replica("model-a", "replica-1")
        handle = r.get_handle()

        mock_get_handle.assert_called_once_with("model-a", "replica-1")
        assert handle == "fake_handle"


class TestReplicaSubmit:
    """Test Replica.submit() RPC delegation."""

    @patch("src.services.api.src.queue.replicas.submit", new_callable=AsyncMock)
    @patch("src.services.api.src.queue.replicas.get_model_actor_handle")
    @pytest.mark.asyncio
    async def test_submit_calls_util_submit(self, mock_get_handle, mock_submit):
        """Verify submit() gets handle and delegates to util.submit with all args."""
        mock_get_handle.return_value = "fake_handle"
        mock_submit.return_value = "result_value"

        r = Replica("model-a", "replica-1")
        result = await r.submit("__call__", "arg1", key="val")

        mock_get_handle.assert_called_once_with("model-a", "replica-1")
        mock_submit.assert_awaited_once_with("fake_handle", "__call__", "arg1", key="val")
        assert result == "result_value"


class TestReplicaWaitUntilReady:
    """Test Replica.wait_until_ready() polling behaviour."""

    @patch("src.services.api.src.queue.replicas.submit", new_callable=AsyncMock)
    @patch("src.services.api.src.queue.replicas.get_model_actor_handle")
    @pytest.mark.asyncio
    async def test_ready_immediately(self, mock_get_handle, mock_submit):
        """Verify wait_until_ready returns immediately when __ray_ready__ succeeds."""
        mock_get_handle.return_value = "handle"
        mock_submit.return_value = None

        r = Replica("m", "r1")
        await r.wait_until_ready(poll_interval_s=0.0)

        mock_submit.assert_awaited_once_with("handle", "__ray_ready__")

    @patch("src.services.api.src.queue.replicas.submit", new_callable=AsyncMock)
    @patch("src.services.api.src.queue.replicas.get_model_actor_handle")
    @pytest.mark.asyncio
    async def test_retries_on_lookup_error(self, mock_get_handle, mock_submit):
        """Verify wait_until_ready retries when a 'Failed to look up actor' error occurs."""
        mock_get_handle.return_value = "handle"
        mock_submit.side_effect = [
            RuntimeError("Failed to look up actor XYZ"),
            None,
        ]

        r = Replica("m", "r1")
        await r.wait_until_ready(poll_interval_s=0.0)

        assert mock_submit.await_count == 2

    @patch("src.services.api.src.queue.replicas.submit", new_callable=AsyncMock)
    @patch("src.services.api.src.queue.replicas.get_model_actor_handle")
    @pytest.mark.asyncio
    async def test_raises_non_lookup_error(self, mock_get_handle, mock_submit):
        """Verify wait_until_ready propagates non-lookup errors immediately."""
        mock_get_handle.return_value = "handle"
        mock_submit.side_effect = ValueError("unexpected error")

        r = Replica("m", "r1")
        with pytest.raises(ValueError, match="unexpected error"):
            await r.wait_until_ready(poll_interval_s=0.0)


# ======================================================================
# Replicas collection tests
# ======================================================================


class TestReplicasInit:
    """Test Replicas constructor."""

    def test_empty_constructor(self):
        """Verify Replicas with no IDs starts empty with zero counts."""
        rs = Replicas("model-a")
        assert rs.model_key == "model-a"
        assert rs.replica_ids == []
        assert rs.replica_count == 0
        assert rs.in_flight == 0

    def test_constructor_with_ids(self):
        """Verify Replicas initializes with the given replica IDs and correct count."""
        rs = Replicas("model-a", ["r1", "r2", "r3"])
        assert rs.replica_ids == ["r1", "r2", "r3"]
        assert rs.replica_count == 3

    def test_constructor_deduplicates_ids(self):
        """Verify constructor removes duplicate replica IDs."""
        rs = Replicas("model-a", ["r1", "r2", "r1", "r3", "r2"])
        assert rs.replica_ids == ["r1", "r2", "r3"]
        assert rs.replica_count == 3

    def test_constructor_with_none(self):
        """Verify constructor handles None as empty replica list."""
        rs = Replicas("model-a", None)
        assert rs.replica_ids == []
        assert rs.replica_count == 0


class TestReplicasHasAndGet:
    """Test has() and get() lookups."""

    def test_has_returns_true_for_existing(self):
        """Verify has() returns True for replica IDs that exist in the collection."""
        rs = Replicas("m", ["r1", "r2"])
        assert rs.has("r1") is True
        assert rs.has("r2") is True

    def test_has_returns_false_for_missing(self):
        """Verify has() returns False for a replica ID not in the collection."""
        rs = Replicas("m", ["r1"])
        assert rs.has("r99") is False

    def test_get_returns_replica_object(self):
        """Verify get() returns a Replica instance with correct model_key and replica_id."""
        rs = Replicas("m", ["r1"])
        replica = rs.get("r1")
        assert isinstance(replica, Replica)
        assert replica.model_key == "m"
        assert replica.replica_id == "r1"

    def test_get_raises_keyerror_for_missing(self):
        """Verify get() raises KeyError for a nonexistent replica ID."""
        rs = Replicas("m", ["r1"])
        with pytest.raises(KeyError):
            rs.get("r99")


class TestReplicasAdd:
    """Test add() method."""

    def test_add_new_returns_true(self):
        """Verify add() returns True and registers a new replica ID."""
        rs = Replicas("m")
        result = rs.add("r1")
        assert result is True
        assert rs.has("r1")
        assert rs.replica_count == 1
        assert "r1" in rs.replica_ids

    def test_add_duplicate_returns_false(self):
        """Verify add() returns False and does not increment count for a duplicate ID."""
        rs = Replicas("m", ["r1"])
        result = rs.add("r1")
        assert result is False
        assert rs.replica_count == 1

    def test_add_multiple_unique(self):
        """Verify adding multiple unique IDs increments count and preserves order."""
        rs = Replicas("m")
        rs.add("r1")
        rs.add("r2")
        rs.add("r3")
        assert rs.replica_count == 3
        assert rs.replica_ids == ["r1", "r2", "r3"]

    def test_add_creates_proper_replica(self):
        """Verify add() creates a Replica with correct model_key, replica_id, and busy=False."""
        rs = Replicas("model-x")
        rs.add("rep-1")
        replica = rs.get("rep-1")
        assert replica.model_key == "model-x"
        assert replica.replica_id == "rep-1"
        assert replica.busy is False


class TestReplicasRemove:
    """Test remove() method."""

    def test_remove_existing_returns_true(self):
        """Verify remove() returns True and removes an existing replica."""
        rs = Replicas("m", ["r1", "r2"])
        result = rs.remove("r1")
        assert result is True
        assert not rs.has("r1")
        assert rs.replica_count == 1
        assert "r1" not in rs.replica_ids

    def test_remove_missing_returns_false(self):
        """Verify remove() returns False for a nonexistent replica ID."""
        rs = Replicas("m", ["r1"])
        result = rs.remove("r99")
        assert result is False
        assert rs.replica_count == 1

    def test_remove_cancels_running_worker_task(self):
        """Verify remove() cancels an in-progress worker task on the replica."""
        rs = Replicas("m", ["r1"])
        mock_task = MagicMock()
        mock_task.done.return_value = False
        rs.get("r1").worker_task = mock_task

        rs.remove("r1")
        mock_task.cancel.assert_called_once()

    def test_remove_does_not_cancel_done_task(self):
        """Verify remove() skips cancel if the worker task is already done."""
        rs = Replicas("m", ["r1"])
        mock_task = MagicMock()
        mock_task.done.return_value = True
        rs.get("r1").worker_task = mock_task

        rs.remove("r1")
        mock_task.cancel.assert_not_called()

    def test_remove_with_no_worker_task(self):
        """Verify remove() succeeds when replica has no worker task assigned."""
        rs = Replicas("m", ["r1"])
        assert rs.get("r1").worker_task is None
        result = rs.remove("r1")
        assert result is True

    def test_remove_last_replica_count_zero(self):
        """Verify removing the last replica leaves count at zero and empty ID list."""
        rs = Replicas("m", ["r1"])
        rs.remove("r1")
        assert rs.replica_count == 0
        assert rs.replica_ids == []


class TestReplicasSetReplicaIds:
    """Test set_replica_ids() sync behaviour."""

    def test_adds_new_replicas(self):
        """Verify set_replica_ids adds replicas not currently in the collection."""
        rs = Replicas("m")
        rs.set_replica_ids(["r1", "r2"])
        assert rs.has("r1")
        assert rs.has("r2")
        assert rs.replica_count == 2
        assert rs.replica_ids == ["r1", "r2"]

    def test_removes_stale_replicas(self):
        """Verify set_replica_ids removes replicas not in the new ID list."""
        rs = Replicas("m", ["r1", "r2", "r3"])
        rs.set_replica_ids(["r2"])
        assert not rs.has("r1")
        assert rs.has("r2")
        assert not rs.has("r3")
        assert rs.replica_count == 1

    def test_keeps_existing_replicas(self):
        """Verify set_replica_ids preserves replicas that appear in both old and new lists."""
        rs = Replicas("m", ["r1", "r2"])
        rs.set_replica_ids(["r1", "r2", "r3"])
        assert rs.has("r1")
        assert rs.has("r2")
        assert rs.has("r3")
        assert rs.replica_count == 3

    def test_deduplicates_input(self):
        """Verify set_replica_ids deduplicates the input ID list."""
        rs = Replicas("m")
        rs.set_replica_ids(["r1", "r1", "r2", "r2"])
        assert rs.replica_count == 2
        assert rs.replica_ids == ["r1", "r2"]

    def test_empty_set_removes_all(self):
        """Verify set_replica_ids([]) removes all replicas."""
        rs = Replicas("m", ["r1", "r2"])
        rs.set_replica_ids([])
        assert rs.replica_count == 0
        assert rs.replica_ids == []

    def test_cancels_running_tasks_for_removed_replicas(self):
        """Verify set_replica_ids cancels worker tasks on replicas being removed."""
        rs = Replicas("m", ["r1", "r2"])
        mock_task = MagicMock()
        mock_task.done.return_value = False
        rs.get("r1").worker_task = mock_task

        rs.set_replica_ids(["r2"])
        mock_task.cancel.assert_called_once()

    def test_noop_when_ids_unchanged(self):
        """Verify set_replica_ids preserves existing Replica objects when IDs are unchanged."""
        rs = Replicas("m", ["r1", "r2"])
        replica_r1 = rs.get("r1")
        rs.set_replica_ids(["r1", "r2"])
        # Existing replica object should be preserved (not recreated)
        assert rs.get("r1") is replica_r1


class TestReplicasBeginEndRequest:
    """Test begin_request / end_request in-flight bookkeeping."""

    def test_begin_increments_in_flight_and_sets_busy(self):
        """Verify begin_request increments in_flight counter and sets replica to busy."""
        rs = Replicas("m", ["r1"])
        rs.begin_request("r1")
        assert rs.in_flight == 1
        assert rs.get("r1").busy is True

    def test_end_decrements_in_flight_and_clears_busy(self):
        """Verify end_request decrements in_flight counter and clears busy flag."""
        rs = Replicas("m", ["r1"])
        rs.begin_request("r1")
        rs.end_request("r1")
        assert rs.in_flight == 0
        assert rs.get("r1").busy is False

    def test_multiple_replicas_in_flight(self):
        """Verify in_flight tracks multiple busy replicas and decrements correctly."""
        rs = Replicas("m", ["r1", "r2", "r3"])
        rs.begin_request("r1")
        rs.begin_request("r2")
        assert rs.in_flight == 2
        assert rs.get("r1").busy is True
        assert rs.get("r2").busy is True
        assert rs.get("r3").busy is False

        rs.end_request("r1")
        assert rs.in_flight == 1
        assert rs.get("r1").busy is False

    def test_begin_request_raises_on_unknown_replica(self):
        """Verify begin_request raises KeyError for a nonexistent replica ID."""
        rs = Replicas("m", ["r1"])
        with pytest.raises(KeyError):
            rs.begin_request("r99")


class TestReplicasProperties:
    """Test current_request_ids, current_request_started_ats, worker_tasks."""

    def test_current_request_ids_empty(self):
        """Verify current_request_ids returns None for all replicas with no active requests."""
        rs = Replicas("m", ["r1", "r2"])
        ids = rs.current_request_ids
        assert ids == {"r1": None, "r2": None}

    def test_current_request_ids_with_set_request(self):
        """Verify current_request_ids reflects a set request on one replica."""
        rs = Replicas("m", ["r1", "r2"])
        rs.get("r1").set_current_request("req-1")
        ids = rs.current_request_ids
        assert ids["r1"] == "req-1"
        assert ids["r2"] is None

    def test_current_request_started_ats_empty(self):
        """Verify current_request_started_ats returns None when no request is active."""
        rs = Replicas("m", ["r1"])
        ats = rs.current_request_started_ats
        assert ats == {"r1": None}

    def test_current_request_started_ats_populated(self):
        """Verify current_request_started_ats returns a positive float after set_current_request."""
        rs = Replicas("m", ["r1"])
        rs.get("r1").set_current_request("req-1")
        ats = rs.current_request_started_ats
        assert isinstance(ats["r1"], float)
        assert ats["r1"] > 0

    def test_worker_tasks_empty_when_no_tasks(self):
        """Verify worker_tasks returns empty dict when no replicas have tasks."""
        rs = Replicas("m", ["r1", "r2"])
        assert rs.worker_tasks == {}

    def test_worker_tasks_excludes_none(self):
        """Verify worker_tasks only includes replicas that have a non-None task."""
        rs = Replicas("m", ["r1", "r2"])
        mock_task = MagicMock(spec=asyncio.Task)
        rs.get("r1").worker_task = mock_task
        tasks = rs.worker_tasks
        assert "r1" in tasks
        assert "r2" not in tasks
        assert tasks["r1"] is mock_task


class TestReplicasGetState:
    """Test get_state() snapshot."""

    def test_empty_state(self):
        """Verify get_state returns zeroed snapshot for an empty Replicas collection."""
        rs = Replicas("m")
        state = rs.get_state()
        assert state == {
            "current_request_ids": {},
            "current_request_started_ats": {},
            "in_flight": 0,
            "replica_count": 0,
            "replica_ids": [],
        }

    def test_state_with_replicas_and_requests(self):
        """Verify get_state includes in_flight count, replica IDs, and request info."""
        rs = Replicas("m", ["r1", "r2"])
        rs.get("r1").set_current_request("req-1")
        rs.begin_request("r1")

        state = rs.get_state()
        assert state["in_flight"] == 1
        assert state["replica_count"] == 2
        assert state["replica_ids"] == ["r1", "r2"]
        assert state["current_request_ids"]["r1"] == "req-1"
        assert state["current_request_ids"]["r2"] is None
        assert isinstance(state["current_request_started_ats"]["r1"], float)

    def test_state_after_add_and_remove(self):
        """Verify get_state reflects changes after add() and remove() operations."""
        rs = Replicas("m", ["r1"])
        rs.add("r2")
        rs.remove("r1")
        state = rs.get_state()
        assert state["replica_count"] == 1
        assert state["replica_ids"] == ["r2"]
