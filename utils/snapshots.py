"""Snapshot system for file backup and rollback.

Creates file snapshots before agent changes so users can roll back
any session to its pre-change state.

Snapshots are stored in ~/.agents/snapshots/<session_id>/
Each snapshot preserves the original file content before modification.
"""

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

SNAPSHOTS_DIR = Path.home() / ".agents" / "snapshots"


def _ensure_snapshots_dir() -> Path:
    """Ensure the snapshots directory exists."""
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    return SNAPSHOTS_DIR


def create_snapshot(session_id: str, workspace: str) -> str:
    """Create a new snapshot directory for a session.

    Args:
        session_id: Unique session identifier.
        workspace: The project directory being worked on.

    Returns:
        Path to the snapshot directory.
    """
    _ensure_snapshots_dir()
    snapshot_dir = SNAPSHOTS_DIR / session_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata = {
        "session_id": session_id,
        "workspace": workspace,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": [],
    }
    metadata_path = snapshot_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    logger.info("Created snapshot for session %s at %s", session_id, snapshot_dir)
    return str(snapshot_dir)


def backup_file(session_id: str, file_path: str) -> bool:
    """Back up a single file before it is modified.

    If the file doesn't exist (new file), records it as a creation
    so rollback knows to delete it.

    Args:
        session_id: Session this backup belongs to.
        file_path: Absolute path to the file to back up.

    Returns:
        True if backup succeeded, False otherwise.
    """
    snapshot_dir = SNAPSHOTS_DIR / session_id
    if not snapshot_dir.exists():
        logger.warning("Snapshot dir for session %s does not exist", session_id)
        return False

    metadata_path = snapshot_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        logger.error("Could not read snapshot metadata for session %s", session_id)
        return False

    # Skip if already backed up
    for entry in metadata["files"]:
        if entry["original_path"] == file_path:
            return True

    file_entry = {
        "original_path": file_path,
        "existed": os.path.exists(file_path),
        "backed_up_at": datetime.now(timezone.utc).isoformat(),
    }

    if os.path.exists(file_path):
        # Copy file content to snapshot dir with a safe name
        safe_name = file_path.replace("/", "__").replace("\\", "__")
        backup_path = snapshot_dir / safe_name
        try:
            shutil.copy2(file_path, backup_path)
            file_entry["backup_name"] = safe_name
        except OSError as e:
            logger.error("Failed to back up %s: %s", file_path, e)
            return False
    else:
        file_entry["backup_name"] = None  # File was created by agents

    metadata["files"].append(file_entry)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return True


def rollback_session(session_id: str) -> dict:
    """Rollback all file changes from a session.

    Restores backed-up files to their original state.
    Deletes files that were created during the session.

    Args:
        session_id: Session to rollback.

    Returns:
        Dict with 'restored', 'deleted', and 'errors' lists.
    """
    snapshot_dir = SNAPSHOTS_DIR / session_id
    result = {"restored": [], "deleted": [], "errors": []}

    if not snapshot_dir.exists():
        result["errors"].append(f"No snapshot found for session {session_id}")
        return result

    metadata_path = snapshot_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        result["errors"].append("Could not read snapshot metadata")
        return result

    for entry in metadata["files"]:
        original_path = entry["original_path"]
        existed = entry["existed"]

        try:
            if existed:
                # Restore original file
                backup_name = entry.get("backup_name")
                if backup_name:
                    backup_path = snapshot_dir / backup_name
                    if backup_path.exists():
                        # Ensure parent dir exists
                        os.makedirs(os.path.dirname(original_path), exist_ok=True)
                        shutil.copy2(str(backup_path), original_path)
                        result["restored"].append(original_path)
                    else:
                        result["errors"].append(
                            f"Backup file missing: {backup_name}"
                        )
                else:
                    result["errors"].append(
                        f"No backup name for existing file: {original_path}"
                    )
            else:
                # File was created by agents -- delete it
                if os.path.exists(original_path):
                    os.remove(original_path)
                    result["deleted"].append(original_path)
        except OSError as e:
            result["errors"].append(f"Failed to rollback {original_path}: {e}")

    logger.info(
        "Rolled back session %s: %d restored, %d deleted, %d errors",
        session_id,
        len(result["restored"]),
        len(result["deleted"]),
        len(result["errors"]),
    )
    return result


def list_snapshots() -> list[dict]:
    """List all available snapshots sorted by creation time (newest first).

    Returns:
        List of snapshot metadata dicts.
    """
    _ensure_snapshots_dir()
    snapshots = []

    for entry in SNAPSHOTS_DIR.iterdir():
        if entry.is_dir():
            metadata_path = entry / "metadata.json"
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text())
                    snapshots.append(metadata)
                except json.JSONDecodeError:
                    continue

    snapshots.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    return snapshots


def cleanup_old_snapshots(keep: int = 20) -> int:
    """Remove old snapshots beyond the keep limit.

    Args:
        keep: Number of most recent snapshots to keep.

    Returns:
        Number of snapshots removed.
    """
    snapshots = list_snapshots()
    removed = 0

    if len(snapshots) <= keep:
        return 0

    for snapshot in snapshots[keep:]:
        session_id = snapshot.get("session_id", "")
        snapshot_dir = SNAPSHOTS_DIR / session_id
        if snapshot_dir.exists():
            try:
                shutil.rmtree(snapshot_dir)
                removed += 1
            except OSError as e:
                logger.error("Failed to remove snapshot %s: %s", session_id, e)

    return removed
