from datetime import datetime

import pytest

from stock_agent.data.exchange_calendar import VN_TIMEZONE
from stock_agent.swing_desk.config import DeskPaths
from tests.swing_desk_support import NOW_EOD, make_repo, market_frames, write_snapshot

EVENING = datetime(2026, 9, 21, 19, 0, tzinfo=VN_TIMEZONE)


def desk(tmp_path):
    repo = make_repo(tmp_path)
    return DeskPaths(repo=repo, data_root=repo, desk=repo / "swing_desk")


class FakeRunner:
    def __init__(self, paths, monkeypatch, make_snapshot=True):
        self.calls, self.paths, self.mp, self.make = [], paths, monkeypatch, make_snapshot

    def __call__(self, cmd, cwd):
        self.calls.append(cmd)
        assert "--run" not in cmd and "--score" not in cmd
        if "stock_agent.pipeline.paper_runner" in cmd and self.make:
            write_snapshot(self.paths.desk_snapshots, market_frames(), self.mp)
        return 0


def test_reuses_fresh_snapshot_without_fetching(tmp_path, monkeypatch):
    from stock_agent.swing_desk import refresh
    paths = desk(tmp_path)
    write_snapshot(paths.paper_snapshots, market_frames(), monkeypatch)
    fake = FakeRunner(paths, monkeypatch)
    rep = refresh.refresh(paths, now=NOW_EOD, runner=fake, s2_override=[])
    assert rep["snapshot_action"] == "reused" and fake.calls == []
    assert (paths.state / "LATEST").read_text() == "2026-09-21"


def test_fetches_into_isolated_desk_dir_with_check_only(tmp_path, monkeypatch):
    from stock_agent.swing_desk import refresh
    paths = desk(tmp_path)
    fake = FakeRunner(paths, monkeypatch)
    rep = refresh.refresh(paths, now=NOW_EOD, runner=fake, s2_override=[])
    cmd = fake.calls[0]
    assert cmd[1:5] == ["-m", "stock_agent.pipeline.paper_runner", "--refresh", "--check"]
    assert cmd[cmd.index("--output-dir") + 1] == str(paths.desk / "snapshots")
    assert rep["snapshot_action"] == "fetched" and str(paths.desk_snapshots) in rep["snapshot"]
    assert not (paths.data_root / "data" / "paper" / "runs").exists()


def test_failed_fetch_builds_no_state(tmp_path, monkeypatch):
    from stock_agent.swing_desk import refresh
    paths = desk(tmp_path)
    rep = refresh.refresh(paths, now=NOW_EOD, runner=FakeRunner(paths, monkeypatch, make_snapshot=False))
    assert rep["snapshot_action"] == "failed" and "not built" in rep["state"]
    assert not (paths.state / "LATEST").exists()


@pytest.mark.parametrize("hour,minute,called", [(16, 30, False), (17, 5, False), (18, 29, False),
                                                (18, 30, True), (19, 0, True), (8, 0, True)])
def test_flows_collection_respects_blackout_window(tmp_path, monkeypatch, hour, minute, called):
    from stock_agent.swing_desk import refresh
    paths = desk(tmp_path)
    fake = FakeRunner(paths, monkeypatch)
    now = datetime(2026, 9, 22 if hour < 16 else 21, hour, minute, tzinfo=VN_TIMEZONE)
    rep = refresh.refresh(paths, now=now, with_flows=True, runner=fake, build_state=False)
    flow_calls = [c for c in fake.calls if "stock_agent.pipeline.foreign_refresh" in c]
    assert bool(flow_calls) is called
    if called:
        assert flow_calls[0][flow_calls[0].index("--root") + 1] == str(paths.flows_root)
        assert "--migrate-legacy" not in flow_calls[0]
    else:
        assert "blackout" in rep["flows_action"]


def test_flows_not_collected_without_flag(tmp_path, monkeypatch):
    from stock_agent.swing_desk import refresh
    paths = desk(tmp_path)
    fake = FakeRunner(paths, monkeypatch)
    rep = refresh.refresh(paths, now=EVENING, runner=fake, build_state=False)
    assert not [c for c in fake.calls if "stock_agent.pipeline.foreign_refresh" in c]
    assert rep["flows_action"].startswith("not_requested")


def test_crash_mid_build_keeps_previous_latest(tmp_path, monkeypatch):
    from stock_agent.swing_desk import refresh, service
    paths = desk(tmp_path)
    write_snapshot(paths.paper_snapshots, market_frames(), monkeypatch)
    refresh.refresh(paths, now=NOW_EOD, runner=FakeRunner(paths, monkeypatch), s2_override=[])
    before = (paths.state / "2026-09-21" / "manifest.json").read_text(encoding="utf-8")

    def boom(state):
        raise RuntimeError("crash while writing")
    monkeypatch.setattr(service, "gate_table", boom)
    with pytest.raises(RuntimeError):
        refresh.refresh(paths, now=NOW_EOD, runner=FakeRunner(paths, monkeypatch), s2_override=[])
    assert (paths.state / "LATEST").read_text() == "2026-09-21"
    assert (paths.state / "2026-09-21" / "manifest.json").read_text(encoding="utf-8") == before
    assert not [p for p in paths.state.iterdir() if p.name.startswith(".tmp-")]


def test_scan_state_freshness_detects_other_snapshot(tmp_path, monkeypatch):
    from stock_agent.swing_desk import refresh, service
    paths = desk(tmp_path)
    write_snapshot(paths.paper_snapshots, market_frames(), monkeypatch)
    refresh.refresh(paths, now=NOW_EOD, runner=FakeRunner(paths, monkeypatch), s2_override=[])
    assert service.state_freshness(paths, NOW_EOD)["ok"]
    ticker = paths.state / "2026-09-21" / "tickers" / "AAA.json"
    ticker.write_text(ticker.read_text(encoding="utf-8").replace('"input_sha256": "', '"input_sha256": "0'), encoding="utf-8")
    assert "hash" in service.state_freshness(paths, NOW_EOD)["reason"]
