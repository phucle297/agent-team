"""Tests for utils/agent_events.py - Agent status event tracking."""

import threading
import time

from utils.agent_events import AgentTracker, AgentStatus


class TestAgentStatus:
    """Test the AgentStatus enum."""

    def test_has_expected_values(self):
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.WORKING.value == "working"
        assert AgentStatus.DONE.value == "done"
        assert AgentStatus.ERROR.value == "error"


class TestAgentTracker:
    """Test the AgentTracker singleton."""

    def test_register_agent(self):
        tracker = AgentTracker()
        tracker.reset()
        tracker.register("planner")
        statuses = tracker.get_all()
        assert "planner" in statuses
        assert statuses["planner"]["status"] == AgentStatus.IDLE

    def test_register_multiple_agents(self):
        tracker = AgentTracker()
        tracker.reset()
        tracker.register("planner")
        tracker.register("coder")
        tracker.register("researcher")
        statuses = tracker.get_all()
        assert len(statuses) == 3

    def test_update_status(self):
        tracker = AgentTracker()
        tracker.reset()
        tracker.register("coder")
        tracker.update("coder", AgentStatus.WORKING, "Writing code...")
        status = tracker.get("coder")
        assert status["status"] == AgentStatus.WORKING
        assert status["detail"] == "Writing code..."

    def test_update_sets_timestamp(self):
        tracker = AgentTracker()
        tracker.reset()
        tracker.register("coder")
        tracker.update("coder", AgentStatus.WORKING)
        status = tracker.get("coder")
        assert "updated_at" in status
        assert status["updated_at"] is not None

    def test_get_returns_none_for_unknown(self):
        tracker = AgentTracker()
        tracker.reset()
        assert tracker.get("nonexistent") is None

    def test_reset_clears_all(self):
        tracker = AgentTracker()
        tracker.register("planner")
        tracker.register("coder")
        tracker.reset()
        assert tracker.get_all() == {}

    def test_get_all_returns_copy(self):
        tracker = AgentTracker()
        tracker.reset()
        tracker.register("planner")
        all_statuses = tracker.get_all()
        all_statuses["planner"]["status"] = AgentStatus.ERROR
        # Original should not be mutated
        assert tracker.get("planner")["status"] == AgentStatus.IDLE

    def test_thread_safety(self):
        """Multiple threads updating concurrently should not corrupt state."""
        tracker = AgentTracker()
        tracker.reset()
        agents = [f"worker_{i}" for i in range(10)]
        for a in agents:
            tracker.register(a)

        errors = []

        def update_agent(name):
            try:
                tracker.update(name, AgentStatus.WORKING, f"{name} working")
                time.sleep(0.01)
                tracker.update(name, AgentStatus.DONE, f"{name} done")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_agent, args=(a,)) for a in agents]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        statuses = tracker.get_all()
        assert len(statuses) == 10
        for a in agents:
            assert statuses[a]["status"] == AgentStatus.DONE

    def test_subscribe_callback(self):
        """Subscriber callbacks should fire on status updates."""
        tracker = AgentTracker()
        tracker.reset()
        tracker.register("planner")

        events = []

        def on_event(agent_name, status, detail):
            events.append((agent_name, status, detail))

        tracker.subscribe(on_event)
        tracker.update("planner", AgentStatus.WORKING, "Planning...")
        tracker.update("planner", AgentStatus.DONE, "Done")

        assert len(events) == 2
        assert events[0] == ("planner", AgentStatus.WORKING, "Planning...")
        assert events[1] == ("planner", AgentStatus.DONE, "Done")

    def test_unsubscribe(self):
        tracker = AgentTracker()
        tracker.reset()
        tracker.register("planner")

        events = []

        def on_event(agent_name, status, detail):
            events.append((agent_name, status, detail))

        tracker.subscribe(on_event)
        tracker.update("planner", AgentStatus.WORKING, "")
        tracker.unsubscribe(on_event)
        tracker.update("planner", AgentStatus.DONE, "")

        assert len(events) == 1  # Only first event recorded

    def test_register_workers_batch(self):
        """Register multiple worker agents at once."""
        tracker = AgentTracker()
        tracker.reset()
        tracker.register_workers(["worker_0", "worker_1", "worker_2"])
        statuses = tracker.get_all()
        assert len(statuses) == 3
        for i in range(3):
            assert f"worker_{i}" in statuses
