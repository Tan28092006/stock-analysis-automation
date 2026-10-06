"""Offline fixture history stores for the swing desk (imported, not collected).

Stores are built through the real `history_store.build_store` with a fake VCI `post` and a
fake clock (no network, no sleeping), so verification, quarantine and hashing are the real code.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from stock_agent.swing_desk.config import DeskPaths
from tests.swing_desk_support import AS_OF, NOW_EOD, make_repo, market_frames, payload, write_snapshot


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def write_store(root: Path, frames: dict[str, pd.DataFrame], *, store_id: str = "fx1", as_of: date = AS_OF,
                fail=(), edit=None, completed_at: str | None = None, aborted: str | None = None) -> Path:
    """Build a store; `fail` symbols get HTTP 404; `edit(symbol, payload)` may alter the provider bytes."""
    from stock_agent.data import history_store as hs
    payloads = {}
    for sym, df in frames.items():
        body = payload(sym, df)
        if edit is not None:
            body = edit(sym, body) or body
        payloads[sym] = json.dumps(body).encode()

    def post(url, json=None, headers=None, timeout=None):
        sym = json["symbols"][0]
        if sym in fail:
            return SimpleNamespace(status_code=404, content=b'{"error":"x"}')
        return SimpleNamespace(status_code=200, content=payloads[sym])

    clock = _Clock()
    start = min(df["date"].iloc[0] for df in frames.values())
    store, _ = hs.build_store(list(frames), root, start=start, as_of=as_of, now=NOW_EOD, post=post,
                              clock=clock.clock, sleep=clock.sleep, store_id=store_id)
    if completed_at is not None or aborted is not None:
        # manifest.json is not itself hashed by verify: pin ordering / mark an aborted run for tests
        path = store / "manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if completed_at is not None:
            data["completed_at"] = completed_at
        if aborted is not None:
            data["aborted"] = aborted
            for sym in fail:
                data["symbols"][sym]["error"] = f"not attempted: {aborted}"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return store


def tail(frames: dict[str, pd.DataFrame], n: int) -> dict[str, pd.DataFrame]:
    return {s: df.tail(n).reset_index(drop=True) for s, df in frames.items()}


def history_desk(tmp_path, monkeypatch, *, long_n: int = 600, snap_n: int = 150, store: bool = True,
                 history: str = "on", **store_kw):
    """Repo + verified snapshot (last `snap_n` sessions) + fixture store (`long_n` sessions).

    Returns (paths, snapshot manifest, generated long frames, snapshot frames as loaded, store dir).
    """
    repo = make_repo(tmp_path)
    paths = DeskPaths(repo=repo, data_root=repo, desk=repo / "swing_desk")
    if history != "on":
        paths = replace(paths, history=history)
    long = market_frames(n=long_n)
    snap = tail(long, snap_n)
    manifest = write_snapshot(paths.paper_snapshots, snap, monkeypatch)
    store_dir = write_store(paths.history_root, long, **store_kw) if store else None
    from stock_agent.swing_desk.data import load_frames
    return paths, manifest, long, load_frames(manifest), store_dir   # snapshot frames as the desk reads them
