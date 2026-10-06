"""Interrupted flow-collector state: read-only diagnosis and guarded quarantine.

A hard kill leaves ``.collect-lock``; a Ctrl-C or exception leaves a run dir
without ``manifest.json``. Either keeps ``foreign_flows.health`` partial every
day. Diagnosis never writes. Quarantine is dry-run unless ``apply=True``; it
MOVES (never deletes) orphan run dirs to ``quarantine/<UTC stamp>/`` with a
``reason.json``, refuses while a collector could plausibly still be running and
holds the collector lock while moving so no collector can start mid-move.

Collector liveness is a heuristic (lock age and last file activity). The lock
carries no PID, so an operator must still confirm no python collector process
is running before ``--apply``.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import uuid

from ..data.foreign_flows import atomic_json

LOCK_NAME = ".collect-lock"
DEFAULT_STALE_AFTER_SECONDS = 2 * 3600  # a full 105-symbol batch takes minutes, not hours
REASON = ("run dir has no manifest.json: collector was interrupted before commit; "
          "not a successful run, never re-hashed or repaired")


def _now(value: datetime | None = None) -> datetime:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return value


def _tree(path: Path) -> tuple[int, int, float]:
    files = size = 0
    newest = path.stat().st_mtime
    for item in path.rglob("*"):
        stat = item.stat()
        newest = max(newest, stat.st_mtime)
        if item.is_file():
            files += 1
            size += stat.st_size
    return files, size, newest


def _orphans(root: Path, now_ts: float) -> list[dict]:
    found = []
    for run in sorted((root / "runs").glob("*")):
        if run.is_dir() and not (run / "manifest.json").exists():
            files, size, newest = _tree(run)
            found.append({"name": run.name, "path": str(run.resolve()), "files": files, "bytes": size,
                          "last_activity_age_seconds": round(now_ts - newest)})
    return found


def diagnose(root, *, now=None, stale_after=DEFAULT_STALE_AFTER_SECONDS) -> dict:
    """Name a stale lock and run dirs without manifest.json. Read-only."""
    root = Path(root)
    now_ts = _now(now).timestamp()
    orphans = _orphans(root, now_ts)
    recent = any(o["last_activity_age_seconds"] < stale_after for o in orphans)
    lock_path = root / LOCK_NAME
    lock = None
    if lock_path.exists():
        age = round(now_ts - lock_path.stat().st_mtime)
        stale = age >= stale_after
        lock = {"present": True, "path": str(lock_path.resolve()), "age_seconds": age, "stale": stale}
    # The collector holds the lock for the whole batch, so without a lock nothing is running.
    plausible = bool(lock) and (not lock["stale"] or recent)
    if plausible:
        assessment = (f"collector may still be running (lock younger than {stale_after}s or recent run activity); "
                      "do not touch the lock or run dirs")
    elif lock or orphans:
        assessment = ("no collector plausibly running; after confirming no python collector process exists, "
                      "run foreign_refresh --quarantine-orphans (dry run) then --quarantine-orphans --apply")
    else:
        assessment = "no lock and no orphan run dirs"
    return {"lock": lock, "orphan_runs": orphans, "collector_plausibly_running": plausible,
            "stale_after_seconds": stale_after, "orphan_assessment": assessment}


def quarantine_orphans(root, *, apply=False, now=None, stale_after=DEFAULT_STALE_AFTER_SECONDS) -> dict:
    root = Path(root)
    now = _now(now)
    diag = diagnose(root, now=now, stale_after=stale_after)
    lock_path = root / LOCK_NAME
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    destination = root / "quarantine" / stamp
    base = {"mode": "apply" if apply else "dry_run", "lock": diag["lock"],
            "stale_after_seconds": stale_after, "destination": str(destination.resolve())}
    if diag["collector_plausibly_running"]:
        return {**base, "status": "refused", "reason": diag["orphan_assessment"],
                "would_move": [o["name"] for o in diag["orphan_runs"]]}
    if not diag["lock"] and not diag["orphan_runs"]:
        return {**base, "status": "nothing_to_do", "destination": None}
    if not apply:
        return {**base, "status": "dry_run", "would_move": [o["name"] for o in diag["orphan_runs"]],
                "would_remove_stale_lock": bool(diag["lock"]),
                "next": "re-run with --apply after confirming no collector process is running"}

    removed_lock = None
    if diag["lock"]:
        try:
            lock_path.rmdir()  # an empty marker dir; refuses if anything was put inside it
        except OSError as exc:
            return {**base, "status": "refused", "reason": f"cannot remove stale lock: {type(exc).__name__}"}
        removed_lock = {"path": diag["lock"]["path"], "age_seconds": diag["lock"]["age_seconds"]}
    try:
        lock_path.mkdir()  # hold the collector lock while moving
    except FileExistsError:
        return {**base, "status": "refused", "stale_lock_removed": removed_lock,
                "reason": "a collector acquired the lock during quarantine; nothing moved"}
    moved: list[dict] = []
    record = {**base, "status": "in_progress", "reason": REASON, "quarantined_at": now.isoformat(),
              "stale_lock_removed": removed_lock, "moved": moved,
              "note": "moved, not deleted; health() no longer counts these dirs; evidence kept for investigation"}
    try:
        targets = _orphans(root, now.timestamp())  # re-listed under the lock
        if targets:
            destination.mkdir(parents=True, exist_ok=False)
            atomic_json(destination / "reason.json", record)
        try:
            for orphan in targets:
                source = root / "runs" / orphan["name"]
                os.rename(source, destination / orphan["name"])
                moved.append({**orphan, "from": str(source.resolve()),
                              "to": str((destination / orphan["name"]).resolve())})
        except Exception as exc:
            record.update(status="failed", error=type(exc).__name__)
            atomic_json(destination / "reason.json", record)
            raise
        record["status"] = "quarantined"
        if targets or removed_lock:
            destination.mkdir(parents=True, exist_ok=True)
            atomic_json(destination / "reason.json", record)
    finally:
        lock_path.rmdir()
    return {**record, "moved": [m["name"] for m in moved]}
