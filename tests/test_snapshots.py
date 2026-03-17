"""Tests for utils/snapshots.py - File snapshot and rollback system."""

import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from utils.snapshots import (
    backup_file,
    cleanup_old_snapshots,
    create_snapshot,
    list_snapshots,
    rollback_session,
)


@pytest.fixture
def snapshots_dir(tmp_path):
    """Override SNAPSHOTS_DIR to use a temp directory."""
    test_dir = tmp_path / "snapshots"
    test_dir.mkdir()
    with patch("utils.snapshots.SNAPSHOTS_DIR", test_dir):
        yield test_dir


@pytest.fixture
def workspace(tmp_path):
    """Create a temp workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


class TestCreateSnapshot:
    def test_creates_snapshot_dir(self, snapshots_dir):
        result = create_snapshot("sess_001", "/some/workspace")
        assert os.path.isdir(result)

    def test_creates_metadata_file(self, snapshots_dir):
        create_snapshot("sess_002", "/some/workspace")
        metadata_path = snapshots_dir / "sess_002" / "metadata.json"
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text())
        assert metadata["session_id"] == "sess_002"
        assert metadata["workspace"] == "/some/workspace"
        assert metadata["files"] == []

    def test_returns_snapshot_path(self, snapshots_dir):
        result = create_snapshot("sess_003", "/ws")
        assert str(snapshots_dir / "sess_003") == result


class TestBackupFile:
    def test_backs_up_existing_file(self, snapshots_dir, workspace):
        create_snapshot("sess_010", str(workspace))
        test_file = workspace / "hello.py"
        test_file.write_text("print('hello')")

        result = backup_file("sess_010", str(test_file))
        assert result is True

        metadata = json.loads(
            (snapshots_dir / "sess_010" / "metadata.json").read_text()
        )
        assert len(metadata["files"]) == 1
        assert metadata["files"][0]["existed"] is True
        assert metadata["files"][0]["backup_name"] is not None

    def test_records_new_file_creation(self, snapshots_dir, workspace):
        create_snapshot("sess_011", str(workspace))
        nonexistent = workspace / "new_file.py"

        result = backup_file("sess_011", str(nonexistent))
        assert result is True

        metadata = json.loads(
            (snapshots_dir / "sess_011" / "metadata.json").read_text()
        )
        assert metadata["files"][0]["existed"] is False
        assert metadata["files"][0]["backup_name"] is None

    def test_skips_duplicate_backup(self, snapshots_dir, workspace):
        create_snapshot("sess_012", str(workspace))
        test_file = workspace / "dup.py"
        test_file.write_text("x = 1")

        backup_file("sess_012", str(test_file))
        backup_file("sess_012", str(test_file))  # Second call

        metadata = json.loads(
            (snapshots_dir / "sess_012" / "metadata.json").read_text()
        )
        assert len(metadata["files"]) == 1

    def test_returns_false_for_missing_session(self, snapshots_dir):
        result = backup_file("nonexistent", "/some/file.py")
        assert result is False


class TestRollbackSession:
    def test_restores_modified_file(self, snapshots_dir, workspace):
        create_snapshot("sess_020", str(workspace))
        test_file = workspace / "restore_me.py"
        test_file.write_text("original content")

        backup_file("sess_020", str(test_file))

        # Simulate modification
        test_file.write_text("modified content")

        result = rollback_session("sess_020")
        assert str(test_file) in result["restored"]
        assert test_file.read_text() == "original content"

    def test_deletes_created_file(self, snapshots_dir, workspace):
        create_snapshot("sess_021", str(workspace))
        new_file = workspace / "created_by_agent.py"

        # Record as "new file" (didn't exist before)
        backup_file("sess_021", str(new_file))

        # Create the file (simulating agent creating it)
        new_file.write_text("agent code")

        result = rollback_session("sess_021")
        assert str(new_file) in result["deleted"]
        assert not new_file.exists()

    def test_returns_errors_for_missing_session(self, snapshots_dir):
        result = rollback_session("no_such_session")
        assert len(result["errors"]) > 0

    def test_handles_missing_backup_file(self, snapshots_dir, workspace):
        create_snapshot("sess_022", str(workspace))
        test_file = workspace / "missing_backup.py"
        test_file.write_text("data")
        backup_file("sess_022", str(test_file))

        # Delete the backup file manually
        metadata = json.loads(
            (snapshots_dir / "sess_022" / "metadata.json").read_text()
        )
        backup_name = metadata["files"][0]["backup_name"]
        (snapshots_dir / "sess_022" / backup_name).unlink()

        result = rollback_session("sess_022")
        assert len(result["errors"]) > 0


class TestListSnapshots:
    def test_returns_empty_when_no_snapshots(self, snapshots_dir):
        result = list_snapshots()
        assert result == []

    def test_returns_all_snapshots(self, snapshots_dir):
        create_snapshot("sess_030", "/ws1")
        create_snapshot("sess_031", "/ws2")
        result = list_snapshots()
        assert len(result) == 2

    def test_sorted_newest_first(self, snapshots_dir):
        create_snapshot("sess_a", "/ws")
        create_snapshot("sess_b", "/ws")
        result = list_snapshots()
        # Most recent should be first
        assert result[0]["session_id"] == "sess_b"


class TestCleanupOldSnapshots:
    def test_removes_excess_snapshots(self, snapshots_dir):
        for i in range(5):
            create_snapshot(f"sess_{i:03d}", "/ws")

        removed = cleanup_old_snapshots(keep=3)
        assert removed == 2
        remaining = list_snapshots()
        assert len(remaining) == 3

    def test_no_removal_when_under_limit(self, snapshots_dir):
        create_snapshot("sess_only", "/ws")
        removed = cleanup_old_snapshots(keep=5)
        assert removed == 0
