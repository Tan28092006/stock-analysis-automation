"""`refresh --held`: isolated extra snapshot for held symbols outside VN30 (mocked fetcher, no network)."""
from datetime import datetime, timezone
import json

import pytest

from tests.swing_desk_bf_support import PNJ_AS_OF, PNJ_NOW, bf_repo, pnj_frame, pnj_index
from tests.swing_desk_support import frame, payload, uptrend, write_snapshot


class FakeVCI:
    def __init__(self, frames, fail=()):
        self.frames, self.fail, self.calls = frames, set(fail), []

    def __call__(self, symbol, start, end):
        from stock_agent.data import reconciliation as rec
        self.calls.append(symbol)
        if symbol in self.fail:
            raise RuntimeError("provider error")
        raw = json.dumps(payload(symbol, self.frames[symbol])).encode()
        return rec.parse_vci_history(json.loads(raw), symbol, start, end), {
            "source": "VCI", "fetched_at": "2026-10-06T09:25:00+00:00", "raw_bytes": raw}


def setup(tmp_path, monkeypatch, journal):
    from stock_agent.data import reconciliation as rec
    paths = bf_repo(tmp_path)
    idx = pnj_index()
    main = {"VNINDEX": idx, **{s: frame(uptrend(300, 20000.0, seed=11 + i), end=PNJ_AS_OF, volume=3e6)
                               for i, s in enumerate(("AAA", "BBB", "VCB"))}}
    write_snapshot(paths.paper_snapshots, main, monkeypatch, as_of=PNJ_AS_OF, name="20261006T100000000000Z")
    monkeypatch.setattr(rec, "completed_session_date", lambda: PNJ_AS_OF)
    paths.desk.mkdir(parents=True, exist_ok=True)
    paths.portfolio.write_text(journal, encoding="utf-8")
    return paths, {"PNJ": pnj_frame(), "VNINDEX": idx, "MWG": frame(uptrend(300, 60000.0, seed=4), end=PNJ_AS_OF)}


def pin_clock(manifest_path):
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["fetched_at"] = data["completed_at"] = "2026-10-06T09:30:00+00:00"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")


JOURNAL = ("2026-10-01|NAV||11200000||||||\n2026-10-02|BUY|AAA||20000|5|S1|PASS|calm||\n"
           "2026-10-06|BUY|PNJ||20150|5|other|NONE|hope||\n")


def tree(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")) if root.exists() else []


def test_fetches_only_held_non_vn30_plus_vnindex_into_new_extra_dir(tmp_path, monkeypatch):
    from stock_agent.swing_desk import refresh, service
    paths, frames = setup(tmp_path, monkeypatch, JOURNAL)
    data_before = tree(paths.data_root / "data")
    fake, said = FakeVCI(frames), []
    rep = refresh.refresh_held(paths, now=PNJ_NOW, fetcher=fake, announce=lambda m: said.append((m, list(fake.calls))))
    assert fake.calls == ["PNJ", "VNINDEX"] and rep["held_outside_vn30"] == ["PNJ"]
    assert said and said[0][1] == [] and "PNJ, VNINDEX" in said[0][0]            # announced before fetching
    assert rep["action"] == "fetched" and rep["status"] == "verified"
    out = paths.extra_snapshots
    assert rep["snapshot"].startswith(str(out)) and len(list(out.iterdir())) == 1
    assert tree(paths.data_root / "data") == data_before                          # nothing written under data/
    pin_clock(out / next(out.iterdir()).name / "manifest.json")
    r = service.check(paths, "PNJ", now=PNJ_NOW, mode="bottomfish", profile="P1", record=False)
    assert "BF-G0" not in r["failed"] and "BF-G1" in r["failed"] and r["snapshot"]["source"] == "extra"
    again = refresh.refresh_held(paths, now=PNJ_NOW, fetcher=fake)
    assert again["action"] == "reused" and fake.calls == ["PNJ", "VNINDEX"]


def test_failed_symbol_blocks_manifest_and_g0_fails_closed(tmp_path, monkeypatch):
    from stock_agent.swing_desk import refresh, service
    paths, frames = setup(tmp_path, monkeypatch, JOURNAL)
    rep = refresh.refresh_held(paths, now=PNJ_NOW, fetcher=FakeVCI(frames, fail={"PNJ"}))
    assert rep["action"] == "blocked" and rep["failures"][0]["symbol"] == "PNJ"
    r = service.check(paths, "PNJ", now=PNJ_NOW, mode="bottomfish", profile="P1", record=False)
    assert "BF-G0" in r["failed"] and r["verdict"] == "BLOCK"


def test_nothing_to_fetch_without_held_outside_vn30(tmp_path, monkeypatch):
    from stock_agent.swing_desk import refresh
    paths, frames = setup(tmp_path, monkeypatch, "2026-10-01|NAV||11200000||||||\n2026-10-02|BUY|AAA||20000|5|S1|PASS|calm||\n")
    fake = FakeVCI(frames)
    rep = refresh.refresh_held(paths, now=PNJ_NOW, fetcher=fake)
    assert fake.calls == [] and rep["action"].startswith("nothing") and not paths.extra_snapshots.exists()


def test_cli_refresh_without_held_does_not_touch_extra(tmp_path, monkeypatch):
    from stock_agent.swing_desk import refresh
    paths, frames = setup(tmp_path, monkeypatch, JOURNAL)
    calls = []
    monkeypatch.setattr(refresh, "refresh_held", lambda *a, **k: calls.append(1))
    from stock_agent.swing_desk.__main__ import build_parser
    args = build_parser().parse_args(["refresh"])
    assert args.held is False and not paths.extra_snapshots.exists() and calls == []
