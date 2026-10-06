"""Append-only, hash-chained JSONL logs (recommendations and pre-trade checks).

Each row carries `prev_sha256` (previous row hash, GENESIS for the first) and `row_sha256`
(sha256 of the canonical row without `row_sha256`). A sidecar `<log>.head` records the row
count and last hash, so truncating the final row is also detected. Rows are never edited or
deleted; losing rows stay. `freeze` refuses rows at/after 09:00 VN of the target session.
"""
from __future__ import annotations

from datetime import date, datetime, time, timezone
import json
import os
from pathlib import Path
import uuid

from ..data.exchange_calendar import VN_TIMEZONE, is_trading_day
from .config import canonical, sha256_bytes

GENESIS = "0" * 64
SOURCES = {"desk", "single", "baseline", "control"}
ANALYST_SOURCES = {"desk", "single"}
REQUIRED = ("ticker", "action", "entry", "stop", "target", "max_hold_sessions", "data_as_of")
ANALYST_REQUIRED = REQUIRED + ("p_win_stated", "stop_basis", "earliest_sell", "invalidation", "sentiment")
ACTIONS = {"BUY", "HOLD", "SELL", "AVOID"}


def _hash(row: dict) -> str:
    return sha256_bytes(canonical({k: v for k, v in row.items() if k != "row_sha256"}))


def _head_path(log: Path) -> Path:
    return log.with_name(log.name + ".head")


def read_rows(log: Path) -> list[dict]:
    log = Path(log)
    if not log.is_file():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_chain(log: Path) -> dict:
    log = Path(log)
    errors = []
    try:
        rows = read_rows(log)
    except ValueError as exc:
        return {"ok": False, "rows": 0, "errors": [f"unparseable log: {exc}"]}
    prev = GENESIS
    for i, row in enumerate(rows, start=1):
        if row.get("prev_sha256") != prev:
            errors.append(f"row {i}: prev_sha256 does not match previous row (deleted/reordered)")
        if row.get("row_sha256") != _hash(row):
            errors.append(f"row {i}: row_sha256 mismatch (edited)")
        prev = row.get("row_sha256")
    head = _head_path(log)
    if rows or head.exists():
        try:
            h = json.loads(head.read_text(encoding="utf-8"))
            if h.get("rows") != len(rows) or h.get("last_sha256") != (rows[-1]["row_sha256"] if rows else GENESIS):
                errors.append(f"head mismatch: head says {h.get('rows')} rows, log has {len(rows)} (truncated?)")
        except (OSError, ValueError):
            errors.append("head file missing or unreadable")
    return {"ok": not errors, "rows": len(rows), "errors": errors}


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def append_many(log: Path, rows: list[dict]) -> list[dict]:
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    lock = log.with_name(log.name + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError(f"log is locked by another writer: {lock}") from None
    try:
        state = verify_chain(log)
        if not state["ok"]:
            raise ValueError(f"refusing to append to a broken chain: {state['errors']}")
        existing = read_rows(log)
        prev = existing[-1]["row_sha256"] if existing else GENESIS
        out = []
        for row in rows:
            row = {k: v for k, v in row.items() if k not in ("prev_sha256", "row_sha256")}
            row["prev_sha256"] = prev
            row["row_sha256"] = _hash(row)
            prev = row["row_sha256"]
            out.append(row)
        with log.open("a", encoding="utf-8") as stream:
            for row in out:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _atomic_write(_head_path(log), json.dumps({"rows": len(existing) + len(out), "last_sha256": prev}))
        return out
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)


def append(log: Path, row: dict) -> dict:
    return append_many(log, [row])[0]


def freeze_deadline(target_session: date) -> datetime:
    return datetime.combine(target_session, time(9, 0), VN_TIMEZONE)


def validate_row(row: dict, source: str, session_as_of: str) -> None:
    required = ANALYST_REQUIRED if source in ANALYST_SOURCES else REQUIRED
    missing = [k for k in required if k not in row or row[k] in (None, "")]
    if missing:
        raise ValueError(f"recommendation {row.get('ticker', '?')}: missing required fields {missing}")
    if row["action"] not in ACTIONS:
        raise ValueError(f"action must be one of {sorted(ACTIONS)}")
    if row["data_as_of"] != session_as_of:
        raise ValueError(f"data_as_of {row['data_as_of']} != state session {session_as_of}")
    if row["action"] == "BUY":
        if row.get("gate") == "BLOCK":
            raise ValueError(f"{row['ticker']}: BUY against a BLOCK gate is not allowed")
        entry = row["entry"]
        if not isinstance(entry, dict) or entry.get("type") not in {"limit", "ato"}:
            raise ValueError("entry must be {type: limit|ato, ...}")
        if entry["type"] == "limit" and not entry.get("limit_price"):
            raise ValueError("limit entry needs limit_price")
        ref = entry.get("limit_price") or row.get("entry_ref")
        if ref and not (0 < float(row["stop"]) < float(ref) < float(row["target"])):
            raise ValueError(f"{row['ticker']}: need stop < entry < target")
    if source in ANALYST_SOURCES:
        p = row["p_win_stated"]
        if not isinstance(p, (int, float)) or not 0 <= p <= 1:
            raise ValueError("p_win_stated must be a probability in [0, 1]")


def freeze(log: Path, rows: list[dict], *, source: str, target_session: date, session_as_of: str,
           now: datetime, meta: dict) -> list[dict]:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {sorted(SOURCES)}")
    if not is_trading_day(target_session):
        raise ValueError(f"target session {target_session} is not a trading day")
    deadline = freeze_deadline(target_session)
    if now >= deadline:
        raise ValueError(f"too late: rows for {target_session} must be frozen before 09:00 VN ({deadline.isoformat()})")
    for row in rows:
        validate_row(row, source, session_as_of)
    created = now.astimezone(timezone.utc).isoformat()
    framed = []
    for i, row in enumerate(rows):
        framed.append({**row, "id": f"{target_session}-{source}-{i + 1:02d}-{row['ticker']}",
                       "source": source, "created_at": created, "frozen_before": deadline.isoformat(),
                       "session_as_of": session_as_of, "target_session": str(target_session), **meta})
    return append_many(log, framed) if framed else []
