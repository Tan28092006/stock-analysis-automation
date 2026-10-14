"""Offline VN100-style fixtures for the swing desk universe switch (imported, not collected).

Paper basket (primary snapshot, `configs/universe_vn30.json`): AAA, BBB, VCB (+VNINDEX).
Active universe "vn100" (`configs/universe_vn100.json`): the paper basket plus CCC, DDD, NAB.
Non-primary symbols get their daily data from desk-owned universe snapshots
(`<desk>/snapshots/universe/<UTC>/`), built through the real reconciliation code.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from stock_agent.swing_desk.config import DeskPaths
from tests.swing_desk_support import UNIVERSE, frame, make_repo, market_frames, uptrend, with_dip, write_snapshot

EXTRA = ["CCC", "DDD", "NAB"]
VN100 = UNIVERSE + EXTRA


def registry(default: str = "vn100") -> dict:
    return {"schema_version": 1, "default": default, "primary_snapshot": "vn30",
            "universes": {"vn30": {"symbols_file": "configs/universe_vn30.json", "sector_map": None},
                          "vn100": {"symbols_file": "configs/universe_vn100.json",
                                    "sector_map": "configs/swing_desk/sectors_vn100.json"}}}


def sector_map(unverified=()) -> dict:
    sectors = {"AAA": "industrials", "BBB": "industrials", "VCB": "financials", "CCC": "industrials",
               "DDD": "real_estate", "NAB": "financials"}
    return {"schema_version": 1, "max_positions_per_sector": 2, "groups": {"bank": ["VCB", "NAB"]},
            "symbols": {s: ({"sector": None, "status": "UNVERIFIED"} if s in unverified
                            else {"sector": sec, "status": "verified"}) for s, sec in sectors.items()}}


def make_vn100_repo(tmp_path: Path, *, default: str = "vn100", unverified=()) -> Path:
    repo = make_repo(tmp_path)
    real = Path(__file__).resolve().parents[1]
    for name in ("bottomfish_v1.json", "bottomfish_prospective_v1.json"):
        (repo / "configs/swing_desk" / name).write_bytes((real / "configs/swing_desk" / name).read_bytes())
    (repo / "configs/swing_desk/universe.json").write_text(json.dumps(registry(default)), encoding="utf-8")
    (repo / "configs/universe_vn100.json").write_text(
        json.dumps({"name": "VN100", "member_count": len(VN100), "symbols": VN100}), encoding="utf-8")
    (repo / "configs/swing_desk/sectors_vn100.json").write_text(json.dumps(sector_map(unverified)), encoding="utf-8")
    return repo


def vn100_frames(dip: str | None = "AAA") -> dict:
    frames = market_frames(dip=dip)
    for i, sym in enumerate(EXTRA):
        closes = uptrend(300, 15000.0 + 500 * i, seed=31 + i)
        if sym == "CCC":
            closes = with_dip(uptrend(300, 15000.0, drift=0.008, wiggle=0.002, seed=5), (-0.03,) * 4)
        frames[sym] = frame(closes, volume=3e6)
    return frames


def vn100_desk(tmp_path, monkeypatch, *, extra=tuple(EXTRA), frames=None, universe_name=None, unverified=()):
    """Paper snapshot (AAA, BBB, VCB, VNINDEX) + a universe snapshot holding `extra` + VNINDEX."""
    repo = make_vn100_repo(tmp_path, unverified=unverified)
    paths = DeskPaths(repo=repo, data_root=repo, desk=repo / "swing_desk")
    if universe_name:
        paths = replace(paths, universe_name=universe_name)
    frames = frames or vn100_frames()
    primary = write_snapshot(paths.paper_snapshots, {s: frames[s] for s in UNIVERSE + ["VNINDEX"]}, monkeypatch)
    uni = None
    if extra:
        uni = write_snapshot(paths.universe_snapshots, {s: frames[s] for s in list(extra) + ["VNINDEX"]}, monkeypatch,
                             name="20260921T100500000000Z")
    return paths, primary, uni, frames
