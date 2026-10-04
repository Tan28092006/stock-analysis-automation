"""R2: immutable raw objects + SQLite catalog; publication boundary is one transaction.

Protocol: stage raw bytes (content-addressed, never overwritten) -> validate ->
one SQLite transaction inserts the attempt result, vintages and the generation row.
Readers only see rows of published generations and hash-verify raw bytes first.
SQLite cannot roll back filesystem writes: a crash before commit leaves an orphan
raw file that is invisible; a missing/corrupt file after commit fails closed.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import uuid
from pathlib import Path

import pandas as pd

from .contracts import PRICE_BAR, ContractError, validate_source

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_object(sha256 TEXT PRIMARY KEY, size INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS ingestion_attempt(
  attempt_id TEXT PRIMARY KEY, request_key TEXT NOT NULL, provider TEXT NOT NULL, scope TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('started','succeeded','partial','failed')),
  raw_sha256 TEXT REFERENCES raw_object(sha256), retry INTEGER NOT NULL DEFAULT 0,
  fetched_at TEXT NOT NULL, committed_at TEXT, expected_sessions INTEGER, received_sessions INTEGER,
  error TEXT, seq INTEGER);
CREATE TABLE IF NOT EXISTS generation(
  generation_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL REFERENCES ingestion_attempt(attempt_id),
  committed_at TEXT NOT NULL, watermark TEXT, ready INTEGER NOT NULL, manifest_sha256 TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS price_bar_vintage(
  symbol TEXT NOT NULL, session_date TEXT NOT NULL, provider TEXT NOT NULL, vintage INTEGER NOT NULL,
  open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL,
  available_at TEXT NOT NULL, generation_id TEXT NOT NULL REFERENCES generation(generation_id),
  PRIMARY KEY(symbol, session_date, provider, vintage));
"""
VALUES = ["open", "high", "low", "close", "volume"]


class CatalogError(RuntimeError):
    """Catalog integrity failure; readers must not continue with partial data."""


class Fault(RuntimeError):
    """Injected crash for tests."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(frame: pd.DataFrame) -> bytes:
    f = frame.copy()
    f["session_date"] = pd.to_datetime(f["session_date"]).dt.strftime("%Y-%m-%d")
    f["available_at"] = f["available_at"].map(lambda t: pd.Timestamp(t).tz_convert("UTC").isoformat())
    f = f.sort_values(["symbol", "session_date"], kind="stable")
    return json.dumps(f.to_dict("records"), sort_keys=True, default=str).encode()


class Catalog:
    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "raw").mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "catalog.sqlite"
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # --- raw store -------------------------------------------------------
    def _stage_raw(self, data: bytes) -> str:
        sha = _sha(data)
        path = self.root / "raw" / sha
        if not path.exists():  # immutable: never overwrite
            tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        return sha

    def _raw(self, sha: str) -> bytes:
        path = self.root / "raw" / sha
        if not path.exists():
            raise CatalogError(f"raw object missing: {sha}")
        data = path.read_bytes()
        if _sha(data) != sha:
            raise CatalogError(f"raw object corrupt: {sha}")
        return data

    # --- ingest ----------------------------------------------------------
    def record_failure(self, *, request_key: str, provider: str, scope: str, status: str,
                       fetched_at: str, error: str = "", retry: int = 0) -> str:
        if status not in ("failed", "partial", "started"):
            raise ValueError("use publish_prices for succeeded attempts")
        attempt = uuid.uuid4().hex
        with self._conn() as c:
            c.execute("INSERT INTO ingestion_attempt(attempt_id,request_key,provider,scope,status,retry,"
                      "fetched_at,error,seq) VALUES(?,?,?,?,?,?,?,?,(SELECT COALESCE(MAX(seq),0)+1 "
                      "FROM ingestion_attempt))",
                      (attempt, request_key, provider, scope, status, retry, fetched_at, error))
        return attempt

    def publish_prices(self, frame: pd.DataFrame, *, request_key: str, provider: str, scope: str,
                       fetched_at: str, expected_sessions: list[str], committed_at: str,
                       retry: int = 0, fault: str | None = None) -> dict:
        """Returns {generation_id, ready, new_vintages, duplicates}. Idempotent per content."""
        data = validate_source(frame, PRICE_BAR)
        if (data.provider != provider).any():
            raise ContractError("frame provider differs from declared provider")
        sha = self._stage_raw(canonical_bytes(data))
        if fault == "after_raw":
            raise Fault("crash after raw staging, before catalog commit")
        got = sorted(set(data.session_date.dt.strftime("%Y-%m-%d")))
        missing = sorted(set(expected_sessions) - set(got))
        status = "succeeded" if not missing else "partial"
        gen = uuid.uuid4().hex
        manifest = _sha(json.dumps({"raw": sha, "expected": sorted(expected_sessions)}).encode())
        new = dup = 0
        conn = self._conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT OR IGNORE INTO raw_object VALUES(?,?)", (sha, (self.root / "raw" / sha).stat().st_size))
            attempt = uuid.uuid4().hex
            conn.execute("INSERT INTO ingestion_attempt VALUES(?,?,?,?,?,?,?,?,?,?,?,?,"
                         "(SELECT COALESCE(MAX(seq),0)+1 FROM ingestion_attempt))",
                         (attempt, request_key, provider, scope, status, sha, retry, fetched_at, committed_at,
                          len(expected_sessions), len(got), f"missing_sessions={missing[:10]}" if missing else None))
            conn.execute("INSERT INTO generation VALUES(?,?,?,?,?,?)",
                         (gen, attempt, committed_at, got[-1] if got and not missing else None,
                          int(not missing), manifest))
            for r in data.itertuples(index=False):
                key = (r.symbol, r.session_date.strftime("%Y-%m-%d"), provider)
                last = conn.execute("SELECT vintage,open,high,low,close,volume FROM price_bar_vintage "
                                    "WHERE symbol=? AND session_date=? AND provider=? "
                                    "ORDER BY vintage DESC LIMIT 1", key).fetchone()
                vals = tuple(float(getattr(r, c)) for c in VALUES)
                if last and tuple(last[1:]) == vals:
                    dup += 1  # same content again: never double-count
                    continue
                conn.execute("INSERT INTO price_bar_vintage VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                             (*key, (last[0] + 1) if last else 1, *vals,
                              pd.Timestamp(r.available_at).tz_convert("UTC").isoformat(), gen))
                new += 1
            if fault == "before_commit":
                raise Fault("crash before commit")
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        return {"generation_id": gen, "ready": not missing, "new_vintages": new, "duplicates": dup,
                "missing_sessions": missing}

    # --- read ------------------------------------------------------------
    def verify(self) -> None:
        with self._conn() as c:
            shas = [r[0] for r in c.execute("SELECT DISTINCT raw_sha256 FROM ingestion_attempt "
                                            "WHERE raw_sha256 IS NOT NULL")]
        for sha in shas:
            self._raw(sha)

    def read_prices(self, knowledge_cutoff: pd.Timestamp, *, require_ready: bool = True) -> pd.DataFrame:
        """Vintages in *ready* generations knowable at cutoff. Fails closed on bad files."""
        self.verify()
        cut = pd.Timestamp(knowledge_cutoff)
        if cut.tzinfo is None:
            raise ContractError("knowledge_cutoff must be timezone-aware")
        where = "g.ready=1" if require_ready else "1=1"
        with self._conn() as c:
            df = pd.read_sql_query(
                f"SELECT v.* FROM price_bar_vintage v JOIN generation g USING(generation_id) WHERE {where}", c)
        df["available_at"] = pd.to_datetime(df.available_at, utc=True)
        df["session_date"] = pd.to_datetime(df.session_date)
        return df[df.available_at <= cut.tz_convert("UTC")].reset_index(drop=True)

    def coverage(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT attempt_id,provider,scope,status,expected_sessions,"
                                     "received_sessions,error FROM ingestion_attempt ORDER BY seq", c)

    # --- projection rebuild / backup ---------------------------------------
    def rebuild_projection(self) -> int:
        """Re-derive vintages from immutable raw objects in attempt order (idempotent)."""
        with self._conn() as c:
            attempts = c.execute("SELECT a.attempt_id,a.raw_sha256,a.provider,g.generation_id FROM "
                                 "ingestion_attempt a JOIN generation g USING(attempt_id) "
                                 "WHERE a.raw_sha256 IS NOT NULL ORDER BY a.seq").fetchall()
        records = [(a, self._raw(sha), prov, gen) for a, sha, prov, gen in attempts]
        conn = self._conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM price_bar_vintage")
            n = 0
            for _, data, prov, gen in records:
                for r in json.loads(data):
                    key = (r["symbol"], r["session_date"], prov)
                    last = conn.execute("SELECT vintage,open,high,low,close,volume FROM price_bar_vintage "
                                        "WHERE symbol=? AND session_date=? AND provider=? "
                                        "ORDER BY vintage DESC LIMIT 1", key).fetchone()
                    vals = tuple(float(r[c]) for c in VALUES)
                    if last and tuple(last[1:]) == vals:
                        continue
                    conn.execute("INSERT INTO price_bar_vintage VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                                 (*key, (last[0] + 1) if last else 1, *vals, r["available_at"], gen))
                    n += 1
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        return n

    def backup(self, dest: Path) -> None:
        dest = Path(dest)
        (dest / "raw").mkdir(parents=True, exist_ok=True)
        src = self._conn()
        dst = sqlite3.connect(dest / "catalog.sqlite")
        try:
            src.backup(dst)
        finally:
            dst.close(); src.close()
        for p in (self.root / "raw").glob("*"):
            if not p.name.endswith(".tmp"):
                shutil.copy2(p, dest / "raw" / p.name)

    @classmethod
    def restore(cls, backup_dir: Path, target: Path) -> "Catalog":
        shutil.copytree(backup_dir, target, dirs_exist_ok=False)
        cat = cls(target)
        cat.verify()  # coherent recovery scope: catalog + raw must agree
        return cat
