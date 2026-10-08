"""T19: configurable desk universe (VN100 default, `--universe vn30` kept), per-ticker daily freshness,
the desk-owned universe snapshot trigger, row provenance and sector concentration caps."""
from dataclasses import replace
from datetime import date
import hashlib
import json
from pathlib import Path

import pytest

from stock_agent.swing_desk.config import DeskPaths
from tests.swing_desk_support import NOW_EOD, UNIVERSE, make_repo
from tests.swing_desk_universe_support import EXTRA, VN100, make_vn100_repo, vn100_desk, vn100_frames

ROOT = Path(__file__).resolve().parents[1]


def _lf_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _rule(result, rule_id):
    return next(r for r in result["rules"] if r["id"] == rule_id)


# ------------------------------------------------------------------ loader

def test_real_repo_default_universe_is_current_vn100():
    from stock_agent.swing_desk.config import load_primary_symbols, load_universe_symbols, universe_info
    paths = DeskPaths.default(repo=ROOT)
    info = universe_info(paths)
    assert info["name"] == "vn100" and info["label"] == "VN100"
    symbols = load_universe_symbols(paths)
    assert len(symbols) == 100 == len(set(symbols))
    assert set(load_primary_symbols(paths)) <= set(symbols)
    assert info["sha256"] == _lf_sha(ROOT / "configs/universe_vn100.json")
    assert info["sector_map"] is not None


def test_vn30_stays_selectable():
    from stock_agent.swing_desk.config import load_universe_symbols, universe_fields
    paths = replace(DeskPaths.default(repo=ROOT), universe_name="vn30")
    want = json.loads((ROOT / "configs/universe_vn30.json").read_text(encoding="utf-8"))["symbols"]
    assert load_universe_symbols(paths) == want
    assert universe_fields(paths) == {"universe": "vn30",
                                      "universe_source_sha": _lf_sha(ROOT / "configs/universe_vn30.json")}


def test_repo_without_registry_is_legacy_vn30(tmp_path):
    from stock_agent.swing_desk.config import load_universe_symbols, load_sector_map, universe_info
    repo = make_repo(tmp_path)
    paths = DeskPaths(repo=repo, data_root=repo, desk=repo / "swing_desk")
    assert universe_info(paths)["name"] == "vn30" and load_universe_symbols(paths) == UNIVERSE
    assert load_sector_map(paths) is None


def test_unknown_universe_is_an_error(tmp_path):
    from stock_agent.swing_desk.config import load_universe_symbols
    repo = make_vn100_repo(tmp_path)
    paths = DeskPaths(repo=repo, data_root=repo, desk=repo / "swing_desk", universe_name="vn500")
    with pytest.raises(ValueError, match="vn500"):
        load_universe_symbols(paths)


@pytest.mark.parametrize("body,match", [
    ({"symbols": ["AAA", "AAA"]}, "duplicate"),
    ({"member_count": 3, "symbols": ["AAA", "BBB"]}, "member_count"),
    ({"symbols": ["AA-A"]}, "invalid"),
    ({"symbols": []}, "empty"),
])
def test_universe_file_is_validated(tmp_path, body, match):
    from stock_agent.swing_desk.config import load_universe_symbols
    repo = make_vn100_repo(tmp_path)
    (repo / "configs/universe_vn100.json").write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_universe_symbols(DeskPaths(repo=repo, data_root=repo, desk=repo / "swing_desk"))


def test_real_vn100_and_sector_configs_are_complete_and_cited():
    uni = json.loads((ROOT / "configs/universe_vn100.json").read_text(encoding="utf-8"))
    assert uni["member_count"] == len(uni["symbols"]) == 100
    assert uni["source"].startswith("https://staticfile.hsx.vn/") and len(uni["source_sha256"]) == 64
    assert uni["effective_from"] == "2026-08-03"
    vn30 = json.loads((ROOT / "configs/universe_vn30.json").read_text(encoding="utf-8"))["symbols"]
    assert set(vn30) <= set(uni["symbols"])
    smap = json.loads((ROOT / "configs/swing_desk/sectors_vn100.json").read_text(encoding="utf-8"))
    assert set(smap["symbols"]) == set(uni["symbols"])
    assert smap["source_sha256"] == uni["source_sha256"]
    for sym, entry in smap["symbols"].items():
        assert entry["status"] in ("verified", "UNVERIFIED"), sym
        assert (entry["sector"] is None) == (entry["status"] == "UNVERIFIED"), sym
    desk = json.loads((ROOT / "configs/swing_desk/desk_v1.json").read_text(encoding="utf-8"))
    banks = set(smap["groups"]["bank"])
    assert {"NAB", "OCB"} <= banks and set(desk["sectors"]["bank"]) & set(uni["symbols"]) <= banks
    assert banks <= {s for s, e in smap["symbols"].items() if e["sector"] == "financials"}


# ------------------------------------------------------------------ per-ticker freshness

def test_universe_tickers_are_fresh_per_ticker(tmp_path, monkeypatch):
    from stock_agent.swing_desk import data as dl
    paths, primary, uni, _ = vn100_desk(tmp_path, monkeypatch)
    fresh = dl.freshness(paths, NOW_EOD)
    assert fresh["ok"] and fresh["manifest"] == str(primary)
    assert fresh["tickers"]["AAA"]["source"] == "primary" and fresh["tickers"]["AAA"]["ok"]
    t = fresh["tickers"]["CCC"]
    assert t["ok"] and t["source"] == "universe" and t["manifest"] == str(uni)
    assert set(fresh["file_sha256"]) >= set(VN100)
    frames = dl.load_desk_frames(fresh)
    assert set(frames) == set(VN100) | {"VNINDEX"}


def test_vn30_mode_freshness_is_unchanged(tmp_path, monkeypatch):
    from stock_agent.swing_desk import data as dl
    paths, _, _, _ = vn100_desk(tmp_path, monkeypatch, universe_name="vn30")
    fresh = dl.freshness(paths, NOW_EOD)
    assert fresh["ok"] and "tickers" not in fresh and set(fresh["file_sha256"]) == set(UNIVERSE) | {"VNINDEX"}


def test_missing_universe_ticker_is_unverified_never_pass(tmp_path, monkeypatch):
    from stock_agent.swing_desk import data as dl, service
    paths, _, _, _ = vn100_desk(tmp_path, monkeypatch, extra=("CCC", "NAB"))
    fresh = dl.freshness(paths, NOW_EOD)
    assert fresh["ok"] and not fresh["tickers"]["DDD"]["ok"] and "refresh" in fresh["tickers"]["DDD"]["reason"]
    r = service.check(paths, "DDD", now=NOW_EOD, s2_override=[], record=False)
    assert r["verdict"] == "WARN"
    assert _rule(r, "G0")["status"] == "UNVERIFIED" and _rule(r, "G1")["status"] == "PASS"
    state = service.compute_state(paths, NOW_EOD, s2_override=[], with_flows_data=False)
    assert set(state["tickers"]) == set(VN100)
    assert state["tickers"]["DDD"]["structural_gate"] == "WARN" and state["tickers"]["DDD"]["input_sha256"] is None
    assert state["manifest"]["ticker_data"]["unverified"] == ["DDD"]
    assert "| DDD |" in service.gate_table(state) and "G0 UNVERIFIED" in service.gate_table(state)


def test_stale_universe_snapshot_is_unverified(tmp_path, monkeypatch):
    from stock_agent.swing_desk import data as dl
    from tests.swing_desk_support import write_snapshot
    paths, _, _, frames = vn100_desk(tmp_path, monkeypatch, extra=())
    old = date(2026, 9, 18)
    write_snapshot(paths.universe_snapshots, {s: frames[s][frames[s]["date"] <= old] for s in EXTRA + ["VNINDEX"]},
                   monkeypatch, as_of=old, name="20260918T100000000000Z")
    fresh = dl.freshness(paths, NOW_EOD)
    assert fresh["ok"] and not fresh["tickers"]["CCC"]["ok"] and "stale" in fresh["tickers"]["CCC"]["reason"]


def test_tampered_universe_snapshot_is_unverified(tmp_path, monkeypatch):
    from stock_agent.swing_desk import data as dl, service
    paths, _, uni, _ = vn100_desk(tmp_path, monkeypatch)
    csv = uni.parent / "prices" / "CCC.csv"
    csv.write_text(csv.read_text().replace("\n2026-09-21,", "\n2026-09-21,1"), encoding="utf-8")
    fresh = dl.freshness(paths, NOW_EOD)
    assert fresh["ok"] and not fresh["tickers"]["CCC"]["ok"] and "verification" in fresh["tickers"]["CCC"]["reason"]
    r = service.check(paths, "CCC", now=NOW_EOD, s2_override=[], record=False)
    assert r["verdict"] != "PASS" and _rule(r, "G0")["status"] == "UNVERIFIED"


def test_vnindex_disagreement_between_snapshots_is_unverified(tmp_path, monkeypatch):
    from stock_agent.swing_desk import data as dl
    from tests.swing_desk_support import write_snapshot
    paths, _, _, frames = vn100_desk(tmp_path, monkeypatch, extra=())
    idx = frames["VNINDEX"].copy()
    idx.loc[idx.index[-1], "close"] = idx["close"].iloc[-1] * 1.01
    idx.loc[idx.index[-1], "high"] = max(idx["high"].iloc[-1], idx["close"].iloc[-1])
    write_snapshot(paths.universe_snapshots, {**{s: frames[s] for s in EXTRA}, "VNINDEX": idx}, monkeypatch,
                   name="20260921T100500000000Z")
    fresh = dl.freshness(paths, NOW_EOD)
    assert not fresh["tickers"]["CCC"]["ok"] and "VNINDEX" in fresh["tickers"]["CCC"]["reason"]


def test_stale_primary_still_blocks_everything(tmp_path, monkeypatch):
    from datetime import datetime
    from stock_agent.data.exchange_calendar import VN_TIMEZONE
    from stock_agent.swing_desk import service
    paths, _, _, _ = vn100_desk(tmp_path, monkeypatch)
    later = datetime(2026, 9, 22, 17, 0, tzinfo=VN_TIMEZONE)
    r = service.check(paths, "CCC", now=later, s2_override=[], record=False)
    assert r["verdict"] == "BLOCK" and r["failed"] == ["G0"]


# ------------------------------------------------------------------ G1 / BF-G1 membership

def test_g1_tests_membership_in_the_active_universe(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    paths, _, _, _ = vn100_desk(tmp_path, monkeypatch)
    r = service.check(paths, "CCC", now=NOW_EOD, s2_override=[], record=False)
    assert _rule(r, "G1")["status"] == "PASS" and r["indicators"] is not None and not r["out_of_scope"]
    assert r["setups"] == ["S1"]
    r = service.check(paths, "XYZ", now=NOW_EOD, s2_override=[], record=False)
    assert r["verdict"] == "BLOCK" and "VN100" in _rule(r, "G1")["reason"]
    vn30 = replace(paths, universe_name="vn30")
    r = service.check(vn30, "CCC", now=NOW_EOD, s2_override=[], record=False)
    assert r["out_of_scope"] and _rule(r, "G1")["reason"] == "OUT_OF_SCOPE: CCC not in current VN30 universe"


def test_bf_g1_tests_membership_in_the_active_universe(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    paths, _, _, _ = vn100_desk(tmp_path, monkeypatch)
    r = service.check(paths, "CCC", now=NOW_EOD, mode="bottomfish", profile="P1", record=False)
    assert _rule(r, "BF-G1")["status"] == "PASS" and _rule(r, "BF-G0")["status"] == "PASS"
    assert r["snapshot"]["source"] == "universe" and r["in_scope"]
    r = service.check(replace(paths, universe_name="vn30"), "CCC", now=NOW_EOD, mode="bottomfish", profile="P1",
                      record=False)
    assert _rule(r, "BF-G1")["status"] == "FAIL" and "VN30" in _rule(r, "BF-G1")["reason"]


def test_s2_is_not_claimed_for_names_outside_the_paper_basket(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    paths, _, _, _ = vn100_desk(tmp_path, monkeypatch)
    monkeypatch.setattr("stock_agent.swing_desk.data.s2_symbols", lambda m, a: (["AAA"], "OK"))
    r = service.check(paths, "CCC", now=NOW_EOD, record=False)
    assert "S2" not in r["setups"] and r["s2_status"].startswith("NA")


def test_market_regime_and_breadth_stay_on_the_paper_basket(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    paths, _, _, _ = vn100_desk(tmp_path, monkeypatch)
    a = service.compute_state(paths, NOW_EOD, s2_override=[], with_flows_data=False)
    b = service.compute_state(replace(paths, universe_name="vn30"), NOW_EOD, s2_override=[], with_flows_data=False)
    for key in ("regime", "breadth_pct", "index_close", "limit_up_count", "limit_down_count"):
        assert a["market"][key] == b["market"][key], key


# ------------------------------------------------------------------ scans cover the universe

def test_scans_cover_the_whole_universe(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    from stock_agent.swing_desk.bf_service import scan_rows, scan_table
    paths, _, _, _ = vn100_desk(tmp_path, monkeypatch, extra=("CCC", "NAB"))
    state = service.compute_state(paths, NOW_EOD, s2_override=[], with_flows_data=False)
    table = service.gate_table(state)
    for sym in VN100:
        assert f"| {sym} |" in table, sym
    scan = scan_rows(paths, NOW_EOD)
    assert {r["ticker"] for r in scan["rows"]} == set(VN100) - {"DDD"} and scan["unverified"] == ["DDD"]
    text = scan_table(scan)
    assert "VN100" in text and "| DDD |" in text and "BF-G0" in text


# ------------------------------------------------------------------ refresh trigger

class FakeFetcher:
    """Stands in for the VCI call (no network); returns the payload write_snapshot would use."""

    def __init__(self, frames, fail=()):
        self.frames, self.fail, self.calls = frames, set(fail), []

    def __call__(self, symbol, start, end):
        from datetime import datetime, timezone
        from stock_agent.data import reconciliation as rec
        from tests.swing_desk_support import payload
        self.calls.append(symbol)
        if symbol in self.fail:
            raise TimeoutError(f"{symbol}: read timeout")
        df = self.frames[symbol]
        raw = json.dumps(payload(symbol, df)).encode()
        return rec.parse_vci_history(json.loads(raw), symbol, start, end), {
            "source": "VCI", "fetched_at": datetime(2026, 9, 21, 9, 25, tzinfo=timezone.utc).isoformat(),
            "raw_bytes": raw}


def _pin(monkeypatch):
    """Fixture clock for build_market_snapshot (fetched_at/completed_at must not be after NOW_EOD)."""
    from datetime import datetime, timezone
    from stock_agent.data import reconciliation as rec

    class FixedDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 21, 9, 30, tzinfo=timezone.utc)

    monkeypatch.setattr(rec, "completed_session_date", lambda: date(2026, 9, 21))
    monkeypatch.setattr(rec, "datetime", FixedDT)


def test_refresh_builds_an_isolated_universe_snapshot_and_state(tmp_path, monkeypatch):
    from stock_agent.swing_desk import data as dl, refresh
    paths, primary, _, frames = vn100_desk(tmp_path, monkeypatch, extra=())
    _pin(monkeypatch)
    paper_before = sorted(p.name for p in paths.paper_snapshots.iterdir())
    fetch = FakeFetcher(frames)
    rep = refresh.refresh(paths, now=NOW_EOD, runner=lambda c, d: pytest.fail(f"unexpected {c}"),
                          s2_override=[], universe_fetcher=fetch)
    u = rep["universe_fetch"]
    assert u["action"] == "fetched" and sorted(fetch.calls) == sorted(EXTRA + ["VNINDEX"])
    assert Path(u["snapshot"]).parent.parent == paths.universe_snapshots
    assert sorted(p.name for p in paths.paper_snapshots.iterdir()) == paper_before
    assert not (paths.data_root / "data" / "paper" / "runs").exists()
    fresh = dl.freshness(paths, NOW_EOD)
    assert all(fresh["tickers"][s]["ok"] for s in VN100)
    assert rep["snapshot_action"] == "reused" and (paths.state / "LATEST").read_text() == "2026-09-21"
    assert rep["universe_fetch"]["fresh"] == len(VN100) and rep["universe_fetch"]["unverified"] == []
    again = FakeFetcher(frames)
    rep2 = refresh.refresh(paths, now=NOW_EOD, runner=lambda c, d: 0, s2_override=[], universe_fetcher=again)
    assert rep2["universe_fetch"]["action"] == "reused" and again.calls == []


def test_refresh_keeps_succeeded_symbols_when_one_fails(tmp_path, monkeypatch):
    from stock_agent.swing_desk import data as dl, refresh
    paths, _, _, frames = vn100_desk(tmp_path, monkeypatch, extra=())
    _pin(monkeypatch)
    fetch = FakeFetcher(frames, fail={"DDD"})
    rep = refresh.refresh(paths, now=NOW_EOD, runner=lambda c, d: 0, s2_override=[], universe_fetcher=fetch)
    u = rep["universe_fetch"]
    assert u["action"] == "fetched_partial" and [f["symbol"] for f in u["failures"]] == ["DDD"]
    assert fetch.calls.count("CCC") == 1          # the verified subset is rebuilt from the fetched bytes
    manifests = [json.loads(p.read_text()) for p in paths.universe_snapshots.glob("*/manifest.json")]
    assert sorted(m["status"] for m in manifests) == ["blocked", "verified"]
    fresh = dl.freshness(paths, NOW_EOD)
    assert fresh["tickers"]["CCC"]["ok"] and not fresh["tickers"]["DDD"]["ok"]
    assert u["unverified"] == ["DDD"]


def test_refresh_fetches_only_stale_symbols(tmp_path, monkeypatch):
    from stock_agent.swing_desk import refresh
    paths, _, _, frames = vn100_desk(tmp_path, monkeypatch, extra=("CCC",))
    _pin(monkeypatch)
    fetch = FakeFetcher(frames)
    refresh.refresh(paths, now=NOW_EOD, runner=lambda c, d: 0, build_state=False, universe_fetcher=fetch)
    assert sorted(fetch.calls) == ["DDD", "NAB", "VNINDEX"]


def test_refresh_paces_real_vci_calls(tmp_path, monkeypatch):
    from stock_agent.data import reconciliation as rec
    from stock_agent.swing_desk import refresh
    paths, _, _, frames = vn100_desk(tmp_path, monkeypatch, extra=())
    _pin(monkeypatch)
    fake = FakeFetcher(frames)
    sleeps = []
    monkeypatch.setattr(rec, "fetch_vci_history", fake)
    monkeypatch.setattr(refresh.time, "sleep", sleeps.append)
    refresh.refresh(paths, now=NOW_EOD, runner=lambda c, d: 0, build_state=False)
    assert len(fake.calls) == 4 and sleeps == [refresh.UNIVERSE_PACING_SECONDS] * 3
    assert refresh.UNIVERSE_PACING_SECONDS >= 3.1


def test_refresh_in_vn30_mode_has_no_universe_step(tmp_path, monkeypatch):
    from stock_agent.swing_desk import refresh
    paths, _, _, frames = vn100_desk(tmp_path, monkeypatch, extra=(), universe_name="vn30")
    fetch = FakeFetcher(frames)
    rep = refresh.refresh(paths, now=NOW_EOD, runner=lambda c, d: 0, s2_override=[], universe_fetcher=fetch)
    assert fetch.calls == [] and rep["universe_fetch"]["action"].startswith("not_needed")


def test_refresh_can_skip_the_universe_fetch(tmp_path, monkeypatch):
    from stock_agent.swing_desk import refresh
    paths, _, _, frames = vn100_desk(tmp_path, monkeypatch, extra=())
    fetch = FakeFetcher(frames)
    rep = refresh.refresh(paths, now=NOW_EOD, runner=lambda c, d: 0, s2_override=[], universe_fetcher=fetch,
                          fetch_universe=False)
    assert fetch.calls == [] and rep["universe_fetch"]["action"].startswith("not_requested")
    assert rep["universe_fetch"]["unverified"] == sorted(EXTRA)


def test_history_fetch_command_covers_the_active_universe(tmp_path, monkeypatch):
    from stock_agent.swing_desk import refresh
    paths, _, _, _ = vn100_desk(tmp_path, monkeypatch, extra=())
    cmd = refresh.history_fetch_command(paths, "2026-09-21", ["PNJ"])
    assert cmd[cmd.index("--universe") + 1] == "vnindex"
    syms = cmd[cmd.index("--symbols") + 1:cmd.index("--as-of")]
    assert syms == sorted(set(VN100) | {"PNJ"})
    legacy = refresh.history_fetch_command(replace(paths, universe_name="vn30"), "2026-09-21", ["PNJ"])
    assert legacy[legacy.index("--universe"):legacy.index("--universe") + 3] == ["--universe", "vn30", "vnindex"]


# ------------------------------------------------------------------ row provenance

def test_check_rows_record_universe_and_source_sha(tmp_path, monkeypatch):
    from stock_agent.swing_desk import reclog, service
    paths, _, uni, _ = vn100_desk(tmp_path, monkeypatch)
    service.check(paths, "CCC", now=NOW_EOD, s2_override=[])
    service.check(paths, "CCC", now=NOW_EOD, mode="bottomfish", profile="P1")
    service.check(replace(paths, universe_name="vn30"), "AAA", now=NOW_EOD, s2_override=[])
    rows = reclog.read_rows(paths.checks_log)
    sha100 = _lf_sha(paths.repo / "configs/universe_vn100.json")
    assert [(r["universe"], r["universe_source_sha"]) for r in rows] == [
        ("vn100", sha100), ("vn100", sha100), ("vn30", _lf_sha(paths.repo / "configs/universe_vn30.json"))]
    from stock_agent.swing_desk.data import manifest_sha256
    assert rows[0]["ticker_snapshot_manifest_sha256"] == manifest_sha256(uni)
    assert reclog.verify_chain(paths.checks_log)["ok"]


def test_frozen_rows_record_universe(tmp_path, monkeypatch):
    from stock_agent.swing_desk import baseline, reclog, service
    paths, _, _, _ = vn100_desk(tmp_path, monkeypatch)
    state = service.compute_state(paths, NOW_EOD, s2_override=[], with_flows_data=False)
    service.write_state(paths, state)
    assert state["manifest"]["universe"] == "vn100"
    out = baseline.freeze_code_arms(paths, now=NOW_EOD)
    assert {r["ticker"] for r in out["baseline"]} <= set(VN100) and "CCC" in {r["ticker"] for r in out["baseline"]}
    rows = reclog.read_rows(paths.rec_log)
    sha = _lf_sha(paths.repo / "configs/universe_vn100.json")
    assert rows and all(r["universe"] == "vn100" and r["universe_source_sha"] == sha for r in rows)


def test_state_is_stale_when_a_universe_snapshot_appears(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    from tests.swing_desk_support import write_snapshot
    paths, _, _, frames = vn100_desk(tmp_path, monkeypatch, extra=("CCC", "NAB"))
    service.write_state(paths, service.compute_state(paths, NOW_EOD, s2_override=[], with_flows_data=False))
    assert service.state_freshness(paths, NOW_EOD)["ok"]
    write_snapshot(paths.universe_snapshots, {"DDD": frames["DDD"], "VNINDEX": frames["VNINDEX"]}, monkeypatch,
                   name="20260921T110000000000Z")
    fresh = service.state_freshness(paths, NOW_EOD)
    assert not fresh["ok"] and "DDD" in fresh["reason"]


# ------------------------------------------------------------------ sector concentration caps

def _plan(cfg):
    from stock_agent.swing_desk.sizing import plan
    return plan(20000.0, 1_000_000_000.0, cfg)


def _portfolio(*held):
    return {"nav": 1e9, "positions": {t: {"qty": 100, "avg_cost": 20000.0, "cost_value": 2e6} for t in held}}


def test_sector_cap_applies_with_the_vn100_sector_map(tmp_path):
    from stock_agent.swing_desk.config import concentration, load_config
    from stock_agent.swing_desk.gate import _g9
    repo = make_vn100_repo(tmp_path)
    paths = DeskPaths(repo=repo, data_root=repo, desk=repo / "swing_desk")
    cfg = load_config(paths.config)
    conc = concentration(paths, cfg)
    assert conc["max_per_sector"] == 2 and set(conc["groups"]["bank"]) >= {"VCB", "NAB"}
    r = _g9("CCC", _plan(cfg), _portfolio("AAA", "BBB"), cfg, conc)
    assert r["status"] == "FAIL" and "industrials positions 3 > 2" in r["reason"]
    assert _g9("CCC", _plan(cfg), _portfolio("AAA"), cfg, conc)["status"] == "PASS"
    assert _g9("CCC", _plan(cfg), _portfolio("AAA", "BBB"), cfg, None)["status"] == "PASS"   # legacy vn30


def test_bank_group_is_the_union_with_the_sector_map(tmp_path):
    from stock_agent.swing_desk.config import concentration, load_config
    from stock_agent.swing_desk.gate import _g9
    repo = make_vn100_repo(tmp_path)
    paths = DeskPaths(repo=repo, data_root=repo, desk=repo / "swing_desk")
    cfg = load_config(paths.config)
    conc = concentration(paths, cfg)
    conc = {**conc, "max_per_sector": 9}
    r = _g9("NAB", _plan(cfg), _portfolio("VCB", "ACB"), cfg, conc)
    assert r["status"] == "FAIL" and "bank positions 3 > 2" in r["reason"]


def test_unverified_sector_mapping_is_unverified_not_guessed(tmp_path):
    from stock_agent.swing_desk.config import concentration, load_config
    from stock_agent.swing_desk.gate import _g9
    repo = make_vn100_repo(tmp_path, unverified=("DDD",))
    paths = DeskPaths(repo=repo, data_root=repo, desk=repo / "swing_desk")
    cfg = load_config(paths.config)
    r = _g9("DDD", _plan(cfg), _portfolio("AAA"), cfg, concentration(paths, cfg))
    assert r["status"] == "UNVERIFIED" and "sector" in r["reason"]


def test_check_applies_the_sector_cap(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    paths, _, _, _ = vn100_desk(tmp_path, monkeypatch)
    paths.desk.mkdir(parents=True, exist_ok=True)
    paths.portfolio.write_text("2026-09-01|NAV||1000000000||||||\n2026-09-10|BUY|AAA||20000|100|S1|PASS|calm||\n"
                               "2026-09-10|BUY|BBB||20000|100|S1|PASS|calm||\n", encoding="utf-8")
    r = service.check(paths, "CCC", now=NOW_EOD, s2_override=[], record=False)
    assert _rule(r, "G9")["status"] == "FAIL" and "industrials" in _rule(r, "G9")["reason"]


# ------------------------------------------------------------------ golden invariance

def test_vn30_mode_in_a_vn100_repo_equals_the_legacy_repo(tmp_path, monkeypatch):
    from stock_agent.swing_desk import service
    from tests.swing_desk_support import market_frames, write_snapshot
    legacy_repo = make_repo(tmp_path / "legacy")
    legacy = DeskPaths(repo=legacy_repo, data_root=legacy_repo, desk=legacy_repo / "swing_desk", history="off")
    write_snapshot(legacy.paper_snapshots, market_frames(), monkeypatch)
    paths, _, _, _ = vn100_desk(tmp_path / "new", monkeypatch, universe_name="vn30")
    paths = replace(paths, history="off")
    drop = {"snapshot_manifest_sha256", "desk_code_hash", "checked_at"}
    for ticker in ("AAA", "CCC", "XYZ"):
        a = service.check(legacy, ticker, now=NOW_EOD, s2_override=[], record=False)
        b = service.check(paths, ticker, now=NOW_EOD, s2_override=[], record=False)
        assert {k: v for k, v in a.items() if k not in drop} == {k: v for k, v in b.items() if k not in drop}, ticker
    ta = service.gate_table(service.compute_state(legacy, NOW_EOD, s2_override=[], with_flows_data=False))
    tb = service.gate_table(service.compute_state(paths, NOW_EOD, s2_override=[], with_flows_data=False))
    strip = lambda t: "\n".join(line for line in t.splitlines() if "snapshot sha" not in line)  # noqa: E731
    assert strip(ta) == strip(tb)


def test_cli_universe_option(tmp_path, monkeypatch, capsys):
    from stock_agent.swing_desk import __main__ as cli
    paths, _, _, _ = vn100_desk(tmp_path, monkeypatch)
    args = cli.build_parser().parse_args(["--universe", "vn30", "status"])
    assert cli._paths(args).universe_name == "vn30"
    assert cli._paths(cli.build_parser().parse_args(["status"])).universe_name is None
    from stock_agent.swing_desk import refresh
    refresh.refresh(paths, now=NOW_EOD, runner=lambda c, d: 0, s2_override=[], fetch_universe=False)
    code = cli.main(["--repo", str(paths.repo), "--now", NOW_EOD.isoformat(), "scan"])
    out = capsys.readouterr().out
    assert code == 0 and all(f"| {s} |" in out for s in VN100)


# ------------------------------------------------------------------ docs

def test_skill_docs_describe_the_active_universe_and_odd_lots():
    for rel in (".claude/skills/swing-desk/SKILL.md", ".codex/skills/swing-desk/SKILL.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "outside the active universe (default VN100)" in text, rel
        assert "Recommend outside VN30" not in text and "lot-100" not in text and "lot 100" not in text, rel
        assert "--universe vn30" in text and "NEWS_SOURCES.md" in text, rel
    readme = (ROOT / "swing_desk/README.md").read_text(encoding="utf-8")
    assert "--universe vn30" in readme and "snapshots/universe/" in readme
    hist = (ROOT / "docs/HISTORY-STORE.md").read_text(encoding="utf-8")
    assert "20261008T035457Z_9683ccc6" in hist
    amend = (ROOT / "configs/swing_desk/protocol_amendments.md").read_text(encoding="utf-8")
    assert "universe_source_sha" in amend and "VN100" in amend
