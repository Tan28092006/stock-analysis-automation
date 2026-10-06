"""`refresh` reports history-store status; `refresh --history` fetches + verifies through the CLI (mocked)."""
import json

import pytest

from tests.swing_desk_history_support import history_desk, write_store
from tests.swing_desk_support import NOW_EOD


@pytest.fixture(autouse=True)
def _fresh_cache():
    from stock_agent.swing_desk import history
    history.clear_cache()


class FakeRunner:
    """Stands in for subprocess: `history_store fetch` writes a fixture store; nothing touches the network."""

    def __init__(self, paths, long, *, make_store=True):
        self.paths, self.long, self.make, self.calls = paths, long, make_store, []

    def __call__(self, cmd, cwd):
        self.calls.append(list(cmd))
        assert "stock_agent.pipeline.paper_runner" not in cmd     # the snapshot is fresh in these tests
        if cmd[1:4] == ["-m", "stock_agent.data.history_store", "fetch"] and self.make:
            write_store(self.paths.history_root, self.long, store_id="fetched")
        return 0


def _held(paths, *tickers):
    paths.desk.mkdir(parents=True, exist_ok=True)
    lines = ["2026-09-01|NAV||1000000000||||||"]
    lines += [f"2026-09-10|BUY|{t}||20000|100|S1|PASS|calm||" for t in tickers]
    paths.portfolio.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_refresh_reports_store_status_and_per_ticker_coverage(tmp_path, monkeypatch):
    from stock_agent.swing_desk import refresh
    paths, _, long, _, store = history_desk(tmp_path, monkeypatch, fail=("VCB",))
    fake = FakeRunner(paths, long)
    rep = refresh.refresh(paths, now=NOW_EOD, runner=fake, s2_override=[])
    h = rep["history"]
    assert fake.calls == [] and rep["history_fetch"] is None
    assert h["newest_store"] == store.name and h["as_of"] == "2026-09-21" and h["stale"] is False
    assert h["sessions_behind"] == 0 and h["snapshot_as_of"] == "2026-09-21"
    assert h["coverage"]["AAA"]["history_source"] == "history_store"
    assert h["coverage"]["AAA"]["history_store_id"] == store.name
    assert h["coverage"]["VCB"]["history_source"] == "snapshot_650d"


def test_refresh_without_store_reports_stale_and_does_not_fetch(tmp_path, monkeypatch):
    from stock_agent.swing_desk import refresh
    paths, _, long, _, _ = history_desk(tmp_path, monkeypatch, store=False)
    fake = FakeRunner(paths, long)
    rep = refresh.refresh(paths, now=NOW_EOD, runner=fake, s2_override=[])
    assert fake.calls == [] and rep["history"]["stale"] is True and rep["history"]["newest_store"] is None
    assert {c["history_source"] for c in rep["history"]["coverage"].values()} == {"snapshot_650d"}
    assert "refresh --history" in rep["history"]["hint"]


def test_refresh_history_fetches_vn30_index_and_held_then_verifies(tmp_path, monkeypatch):
    from stock_agent.swing_desk import refresh
    paths, _, long, _, _ = history_desk(tmp_path, monkeypatch, store=False)
    _held(paths, "PNJ", "AAA")
    fake = FakeRunner(paths, long)
    rep = refresh.refresh(paths, now=NOW_EOD, runner=fake, s2_override=[], fetch_history=True)
    fetch, verify = fake.calls
    assert fetch[1:4] == ["-m", "stock_agent.data.history_store", "fetch"]
    assert fetch[fetch.index("--universe") + 1: fetch.index("--universe") + 3] == ["vn30", "vnindex"]
    assert fetch[fetch.index("--symbols") + 1: fetch.index("--symbols") + 3] == ["AAA", "PNJ"]
    assert fetch[fetch.index("--out") + 1] == str(paths.history_root)
    assert fetch[fetch.index("--as-of") + 1] == "2026-09-21" and fetch[fetch.index("--years") + 1] == "10"
    assert "--dry-run" not in fetch
    new = next(p for p in paths.history_root.iterdir() if p.name.endswith("_fetched"))
    assert verify[1:4] == ["-m", "stock_agent.data.history_store", "verify"] and verify[4] == str(new)
    hf = rep["history_fetch"]
    assert hf["action"] == "fetched" and hf["store"] == new.name and hf["fetch_exit_code"] == 0
    assert hf["verify_exit_code"] == 0
    assert rep["history"]["coverage"]["AAA"]["history_store_id"] == new.name    # state built on the new store
    state = json.loads((paths.state / "2026-09-21" / "manifest.json").read_text(encoding="utf-8"))
    assert state["history"]["stores"] == {new.name: rep["history"]["coverage"]["AAA"]["history_store_manifest_sha256"]}


def test_refresh_history_skips_fetch_when_a_fresh_verified_store_covers_everything(tmp_path, monkeypatch):
    from stock_agent.swing_desk import refresh
    paths, _, long, _, store = history_desk(tmp_path, monkeypatch)
    fake = FakeRunner(paths, long)
    rep = refresh.refresh(paths, now=NOW_EOD, runner=fake, s2_override=[], fetch_history=True)
    assert fake.calls == [] and rep["history_fetch"]["action"] == "fresh"
    assert rep["history_fetch"]["store"] == store.name


def test_refresh_history_refetches_when_a_required_symbol_is_missing(tmp_path, monkeypatch):
    from stock_agent.swing_desk import refresh
    paths, _, long, _, _ = history_desk(tmp_path, monkeypatch, fail=("AAA",))
    fake = FakeRunner(paths, long)
    rep = refresh.refresh(paths, now=NOW_EOD, runner=fake, s2_override=[], fetch_history=True)
    assert "AAA" in rep["history_fetch"]["reason"] and len(fake.calls) == 2


def test_refresh_history_does_not_start_a_second_concurrent_fetch(tmp_path, monkeypatch):
    from stock_agent.swing_desk import refresh
    paths, _, long, _, _ = history_desk(tmp_path, monkeypatch, store=False)
    (paths.history_root / "20260921T095500Z_running" / "raw").mkdir(parents=True)   # 5 min old, no manifest
    fake = FakeRunner(paths, long)
    rep = refresh.refresh(paths, now=NOW_EOD, runner=fake, s2_override=[], fetch_history=True)
    assert fake.calls == [] and rep["history_fetch"]["action"] == "skipped_in_progress"
    assert "20260921T095500Z_running" in rep["history_fetch"]["reason"]


def test_refresh_history_failed_fetch_keeps_snapshot_and_reports(tmp_path, monkeypatch):
    from stock_agent.swing_desk import refresh
    paths, _, long, _, _ = history_desk(tmp_path, monkeypatch, store=False)
    fake = FakeRunner(paths, long, make_store=False)
    rep = refresh.refresh(paths, now=NOW_EOD, runner=fake, s2_override=[], fetch_history=True)
    assert len(fake.calls) == 1 and rep["history_fetch"]["action"] == "fetch_failed"
    assert rep["history"]["coverage"]["AAA"]["history_source"] == "snapshot_650d"


def test_status_shows_history_store_line(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    paths, _, _, _, store = history_desk(tmp_path, monkeypatch)
    st = service.status(paths, NOW_EOD, with_flows=False)
    h = st["history_store"]
    assert h["newest_store"] == store.name and h["stale"] is False and h["mode"] == "on"
    assert h["matches_snapshot_as_of"] is True and h["counts"] == {"ok": 4, "partial": 0, "failed": 0}


def test_cli_history_flags(tmp_path, monkeypatch):
    from stock_agent.swing_desk import __main__ as cli, refresh
    seen = {}
    monkeypatch.setattr(refresh, "refresh", lambda paths, **kw: seen.update(paths=paths, **kw) or {})
    assert cli.main(["--repo", str(tmp_path), "refresh", "--history"]) == 0
    assert seen["fetch_history"] is True and seen["paths"].history == "on"
    assert cli.main(["--repo", str(tmp_path), "--history", "off", "refresh"]) == 0
    assert seen["fetch_history"] is False and seen["paths"].history == "off"
    args = cli.build_parser().parse_args(["--history", "off", "check", "TCB"])
    assert cli._paths(args).history == "off"
