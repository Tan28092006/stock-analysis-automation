"""Per-session EOD step status records — observability only.

Each scheduled step (``foreign_refresh`` collection, ``paper_runner --run``)
merges one entry into ``data/pipeline/eod_status/<session>.json`` (override
with ``VN30_EOD_STATUS_DIR``). The file is replaced atomically. It separates
"paper recorded" from "scoring blocked" so a recorded session is not hidden
behind exit code 2. Writing status never changes a step's exit code.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

DEFAULT_DIR = Path("data/pipeline/eod_status")
ENV = "VN30_EOD_STATUS_DIR"
SCHEMA_VERSION = 1


def status_dir() -> Path:
    return Path(os.environ.get(ENV) or DEFAULT_DIR)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def record_step(session: str, step: str, entry: dict, *, directory: Path | None = None,
                wait_seconds: float = 5.0) -> Path:
    """Merge ``entry`` as ``steps[step]`` of the session record, under a short mkdir lock."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(session)) or not re.fullmatch(r"[a-z_]{1,40}", step):
        raise ValueError("Invalid EOD status session or step name")
    directory = Path(directory) if directory else status_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{session}.json"
    lock = directory / f".{session}.lock"
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"EOD status lock busy: {lock}")
            time.sleep(0.1)
    try:
        record = {"schema_version": SCHEMA_VERSION, "session": str(session), "steps": {}}
        if path.exists():
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(previous, dict) and isinstance(previous.get("steps"), dict):
                    record = previous
                else:
                    record["previous_record_unreadable"] = True
            except ValueError:
                record["previous_record_unreadable"] = True
        attempt = int(record["steps"].get(step, {}).get("attempt", 0)) + 1
        record["steps"][step] = {"step": step, **entry, "attempt": attempt}
        record["updated_at"] = utc_now()
        _atomic_json(path, record)
    finally:
        lock.rmdir()
    return path


def safe_record_step(session: str, step: str, entry: dict) -> Path | None:
    """Best effort: a status failure is reported on stderr, never raised."""
    try:
        return record_step(session, step, entry)
    except Exception as exc:
        print(f"WARNING: eod_status not written for {step} {session}: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return None
