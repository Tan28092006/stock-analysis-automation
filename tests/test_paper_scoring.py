from datetime import date, datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pytest

from stock_agent.pipeline import paper_runner as runner
from stock_agent.data import reconciliation as rec
from tests.test_market_runtime_hardening import make_snapshot


def record(tmp_path, track="mr", *, tracks=None, transform=None):
    manifest = make_snapshot(tmp_path / "original", transform=transform)
    now = datetime(2026, 9, 21, 10, tzinfo=timezone.utc)
    payload = runner.run_paper(manifest.parent / "prices", ["AAA"], manifest_path=manifest, now=now)
    close = float(pd.read_csv(manifest.parent / "prices/AAA.csv").close.iloc[-1])
    payload["recommendations"] = [{"track": selected, "symbol": "AAA", "date": "2026-09-21", "close": close,
        "entry_reference": close, "stop_loss": close - 500, "take_profit": close + 2000, "max_hold_days": 15}
        for selected in (tracks or [track])]
    runner.commit_paper_run(payload, tmp_path / "out", now=now)
    return manifest


def test_pending_is_not_a_loss_or_profit_and_original_is_immutable(tmp_path):
    from stock_agent.pipeline.paper_scoring import score_paper_runs
    manifest = record(tmp_path)
    path = tmp_path / "out/runs/2026-09-21/paper.json"
    before = path.read_bytes()
    result = score_paper_runs(tmp_path / "out", manifest.parent / "prices", manifest_path=manifest)
    assert result["pending"] == 1 and result["resolved"] == 0 and not result["errors"]
    assert result["records"][0]["net_return_pct"] is None
    assert path.read_bytes() == before


def test_stop_gap_uses_worse_open_not_optimistic_stop(tmp_path):
    from stock_agent.pipeline.paper_scoring import score_paper_runs
    record(tmp_path)
    def crash(items):
        if items[0]["symbol"] == "AAA":
            for i, stamp in enumerate(items[0]["t"]):
                if str(pd.Timestamp(stamp, unit="s").date()) == "2026-09-24":
                    for key, value in {"o": 8000, "l": 7900, "h": 10000, "c": 9000}.items():
                        items[0][key][i] = value
        return items
    current = make_snapshot(tmp_path / "current", end=date(2026, 9, 25), transform=crash)
    result = score_paper_runs(tmp_path / "out", current.parent / "prices", manifest_path=current)
    assert not result["errors"] and result["resolved"] == 1
    row = result["records"][0]
    assert row["exit_price"] == 8000 and row["reason"] == "stop"
    assert row["net_return_pct"] == pytest.approx((8000 / row["entry_price"] - 1) * 100 - .6)


@pytest.mark.parametrize("defect", ["malformed", "tampered", "source_revision", "wrong_directory"])
def test_scoring_fails_closed_for_corrupt_records_or_revised_history(tmp_path, defect):
    from stock_agent.pipeline.paper_scoring import score_paper_runs
    original = record(tmp_path)
    path = tmp_path / "out/runs/2026-09-21/paper.json"
    if defect == "malformed":
        path.write_text("{")
    elif defect == "tampered":
        row = json.loads(path.read_text(encoding="utf-8"))
        row["recommendations"][0]["stop_loss"] = 1
        path.write_text(json.dumps(row))
    def revise(items):
        if defect == "source_revision" and items[0]["symbol"] == "AAA":
            items[0]["c"][-5] += 1
        return items
    current = make_snapshot(tmp_path / "current", end=date(2026, 9, 25), transform=revise)
    prices = original.parent / "prices" if defect == "wrong_directory" else current.parent / "prices"
    if defect == "wrong_directory":
        with pytest.raises(ValueError, match="directory"):
            score_paper_runs(tmp_path / "out", prices, manifest_path=current)
    else:
        result = score_paper_runs(tmp_path / "out", prices, manifest_path=current)
        assert result["errors"] and result["resolved"] == result["pending"] == 0


def test_momentum_is_informational_and_remains_pending(tmp_path):
    from stock_agent.pipeline.paper_scoring import score_paper_runs
    manifest = record(tmp_path, track="momentum")
    result = score_paper_runs(tmp_path / "out", manifest.parent / "prices", manifest_path=manifest)
    assert result["records"][0]["metric"] == "informational_close_return_21_sessions"
    assert result["pending"] == 1


def test_no_records_is_an_empty_observation_set(tmp_path):
    from stock_agent.pipeline.paper_scoring import score_paper_runs
    manifest = make_snapshot(tmp_path / "source")
    result = score_paper_runs(tmp_path / "out", manifest.parent / "prices", manifest_path=manifest)
    assert result["pending"] == result["resolved"] == 0
    assert result["status"] == "ok"


def rolling_snapshot(root, start=date(2025, 2, 5), transform=None, symbols=("AAA",), end=date(2026, 9, 25)):
    """Keep absolute prices fixed while changing the provider request boundary."""
    full = make_snapshot(root / "full", symbols=symbols, end=end)

    def fetch(symbol, requested_start, end):
        source = json.loads((full.parent / "raw" / f"{symbol}.json").read_bytes())
        if transform:
            source = transform(source)
        raw = json.dumps(source).encode()
        return rec.parse_vci_history(source, symbol, requested_start, end), {
            "raw_bytes": raw, "fetched_at": f"{end}T10:00:00+00:00"}

    rec.build_market_snapshot(list(symbols) + ["VNINDEX"], root / "rolling", start,
                              end, min_rows=1, fetcher=fetch)
    return root / "rolling/manifest.json"


def drop_date(items, day):
    row = items[0]
    keep = [i for i, stamp in enumerate(row["t"])
            if str(pd.Timestamp(stamp, unit="s").date()) != day]
    for column in ("t", "o", "h", "l", "c", "v"):
        row[column] = [row[column][i] for i in keep]
    return items


@pytest.mark.parametrize("track", ["mr", "momentum"])
@pytest.mark.parametrize("start", [date(2025, 2, 5), date(2025, 3, 3)])
def test_requested_prefix_roll_preserves_outcomes_and_archived_evidence(tmp_path, track, start):
    from stock_agent.pipeline.paper_scoring import score_paper_runs
    original = record(tmp_path, track=track)

    def future_crash(items):
        if items[0]["symbol"] == "AAA":
            i = next(i for i, stamp in enumerate(items[0]["t"])
                     if str(pd.Timestamp(stamp, unit="s").date()) == "2026-09-24")
            for column, value in {"o": 8000, "l": 7900, "h": 10000, "c": 9000}.items():
                items[0][column][i] = value
        return items

    complete = rolling_snapshot(tmp_path / "complete", date(2025, 2, 3), future_crash)
    rolled = rolling_snapshot(tmp_path / "rolled", start, future_crash)
    evidence = [tmp_path / "out/runs/2026-09-21/paper.json"]
    evidence += [p for parent in (original.parent, complete.parent, rolled.parent)
                 for p in parent.rglob("*") if p.is_file()]
    before = {p: p.read_bytes() for p in evidence}
    baseline = score_paper_runs(tmp_path / "out", complete.parent / "prices", manifest_path=complete)
    result = score_paper_runs(tmp_path / "out", rolled.parent / "prices", manifest_path=rolled)
    assert result["status"] == "ok" and not result["errors"]
    assert len(result["records"]) == len(baseline["records"]) == 1
    row, original_row = result["records"][0].copy(), baseline["records"][0].copy()
    check = row.pop("history_check")
    original_row.pop("history_check")
    assert row == original_row
    assert check["contract"] == "requested_window_v1"
    assert check["requested_start"] == str(start)
    assert check["archived_prefix_rows_not_reobserved"] > 0
    assert check["compared_rows"] + check["archived_prefix_rows_not_reobserved"] == len(pd.read_csv(original.parent / "prices/AAA.csv"))
    assert row["status"] == ("resolved" if track == "mr" else "pending")
    if track == "momentum":
        assert row["net_return_pct"] is None
    assert {p: p.read_bytes() for p in evidence} == before


@pytest.mark.parametrize("column", ["o", "h", "l", "c"])
def test_rolled_request_does_not_excuse_shared_price_revisions(tmp_path, column):
    from stock_agent.pipeline.paper_scoring import score_paper_runs
    record(tmp_path)

    def revise(items):
        if items[0]["symbol"] == "AAA":
            items[0][column][-5] += 1  # Signal date; prices remain exact-match.
        return items

    current = rolling_snapshot(tmp_path / "current", transform=revise)
    result = score_paper_runs(tmp_path / "out", current.parent / "prices", manifest_path=current)
    assert result["status"] == "blocked" and result["errors"]
    assert result["records"] == []


@pytest.mark.parametrize("missing", ["2025-02-05", "2025-03-05", "2026-09-21", "2026-09-23"])
def test_rolled_request_still_requires_boundary_history_signal_and_outcome_sessions(tmp_path, missing):
    from stock_agent.pipeline.paper_scoring import score_paper_runs
    record(tmp_path)
    current = rolling_snapshot(tmp_path / "current", transform=lambda rows: drop_date(rows, missing))
    result = score_paper_runs(tmp_path / "out", current.parent / "prices", manifest_path=current)
    assert result["status"] == "blocked" and not result["records"]


def test_request_start_after_signal_is_not_a_valid_overlap(tmp_path):
    from stock_agent.pipeline.paper_scoring import score_paper_runs
    record(tmp_path)
    current = rolling_snapshot(tmp_path / "current", start=date(2026, 9, 22))
    result = score_paper_runs(tmp_path / "out", current.parent / "prices", manifest_path=current)
    assert result["status"] == "blocked" and not result["records"]


def test_rolled_request_does_not_bypass_original_snapshot_integrity(tmp_path):
    from stock_agent.pipeline.paper_scoring import score_paper_runs
    original = record(tmp_path)
    current = rolling_snapshot(tmp_path / "current")
    old_raw = original.parent / "raw/AAA.json"
    old_raw.write_bytes(old_raw.read_bytes() + b" ")
    result = score_paper_runs(tmp_path / "out", current.parent / "prices", manifest_path=current)
    assert result["status"] == "blocked" and not result["records"]


def test_ci_runs_the_paper_operational_contracts():
    workflow = Path(".github/workflows/research-gate.yml").read_text(encoding="utf-8")
    for filename in ("tests/test_paper_scoring.py", "tests/test_paper_cli.py", "tests/test_market_runtime_hardening.py"):
        assert filename in workflow


def test_one_actual_revision_quarantines_the_entire_run_after_valid_rolled_signal(tmp_path):
    from stock_agent.pipeline.paper_scoring import score_paper_runs
    symbols = ["AAA", "BBB"]
    original = make_snapshot(tmp_path / "original", symbols=symbols)
    now = datetime(2026, 9, 21, 10, tzinfo=timezone.utc)
    payload = runner.run_paper(original.parent / "prices", symbols, manifest_path=original, now=now)
    payload["recommendations"] = [
        {"track": "momentum", "symbol": symbol, "date": "2026-09-21",
         "close": float(pd.read_csv(original.parent / "prices" / f"{symbol}.csv").close.iloc[-1])}
        for symbol in symbols]
    runner.commit_paper_run(payload, tmp_path / "out", now=now)

    def revise_second(items):
        if items[0]["symbol"] == "AAA":
            items[0]["v"][-5] += 1  # Accepted only if the rest of this run is valid.
        if items[0]["symbol"] == "BBB":
            items[0]["c"][-5] += 1
        return items

    current = rolling_snapshot(tmp_path / "current", transform=revise_second, symbols=symbols)
    result = score_paper_runs(tmp_path / "out", current.parent / "prices", manifest_path=current)
    assert result["status"] == "blocked" and not result["records"]
    assert "BBB" in result["errors"][0]["error"]
    assert result["revision_warnings"] == []
    assert result["volume_revision_rows"] == result["volume_revision_runs"] == 0


def revise_volume(items, value, day="2026-09-21"):
    if items[0]["symbol"] == "AAA":
        index = next(i for i, stamp in enumerate(items[0]["t"])
                     if str(pd.Timestamp(stamp, unit="s").date()) == day)
        items[0]["v"][index] = value
    return items


@pytest.mark.parametrize("track", ["mr", "momentum"])
@pytest.mark.parametrize("end", [date(2026, 9, 25), date(2026, 10, 21)])
@pytest.mark.parametrize("volume", [999999, 1000001, 100000000])
def test_positive_volume_revisions_flag_but_do_not_change_sealed_outcomes(
        tmp_path, monkeypatch, track, end, volume):
    from stock_agent.pipeline.paper_scoring import score_paper_runs
    original = record(tmp_path, track=track)
    # Synthetic forward clock only; never request future bars from a provider.
    monkeypatch.setattr(rec, "completed_session_date", lambda: end)
    baseline = rolling_snapshot(tmp_path / "baseline", end=end)
    revised = rolling_snapshot(tmp_path / "revised", end=end,
                               transform=lambda rows: revise_volume(rows, volume))
    evidence = [tmp_path / "out/runs/2026-09-21/paper.json"]
    evidence += [p for parent in (original.parent, baseline.parent, revised.parent)
                 for p in parent.rglob("*") if p.is_file()]
    before = {p: p.read_bytes() for p in evidence}

    def no_scan(*args, **kwargs):
        pytest.fail("Scoring must not regenerate the original signal")

    monkeypatch.setattr(runner.mr_scan, "_compute", no_scan)
    monkeypatch.setattr(runner.momentum_scan, "_compute", no_scan)
    control = score_paper_runs(tmp_path / "out", baseline.parent / "prices", manifest_path=baseline)
    result = score_paper_runs(tmp_path / "out", revised.parent / "prices", manifest_path=revised)
    assert result["status"] == "ok_with_revisions" and not result["errors"]
    assert control["status"] == "ok" and control["revision_warnings"] == []
    assert result["volume_revision_rows"] == result["volume_revision_runs"] == 1
    warning = result["revision_warnings"][0]
    assert warning == {"path": str(evidence[0]), "symbol": "AAA", "date": "2026-09-21",
                       "original_volume": 1000000, "outcome_volume": volume}
    row, expected = result["records"][0].copy(), control["records"][0].copy()
    check, unchanged = row.pop("history_check"), expected.pop("history_check")
    assert check["revision_policy"] == "positive_volume_only_v1"
    assert check["decision_inputs_revised"] is True and check["volume_revision_rows"] == 1
    assert unchanged["decision_inputs_revised"] is False
    assert row == expected
    assert row["status"] == ("resolved" if end.month == 10 else "pending")
    assert result["assumptions"]["actual_execution_verified"] is False
    assert result["assumptions"]["portfolio_pnl"] is False
    assert {p: p.read_bytes() for p in evidence} == before


def test_revision_counts_are_per_symbol_date_not_per_strategy(tmp_path):
    from stock_agent.pipeline.paper_scoring import score_paper_runs
    record(tmp_path, tracks=["mr", "momentum"])

    def revision(rows):
        revise_volume(rows, 2000000, day="2026-09-18")
        return revise_volume(rows, 999999)

    revised = rolling_snapshot(tmp_path / "revised", transform=revision)
    result = score_paper_runs(tmp_path / "out", revised.parent / "prices", manifest_path=revised)
    assert result["status"] == "ok_with_revisions" and len(result["records"]) == 2
    assert result["volume_revision_rows"] == 2 and result["volume_revision_runs"] == 1
    assert [w["date"] for w in result["revision_warnings"]] == ["2026-09-18", "2026-09-21"]
    assert all(r["history_check"]["volume_revision_rows"] == 2 for r in result["records"])


@pytest.mark.parametrize("original_volume,outcome_volume", [(0, 1000000), (1000000, 0), (0, 0)])
def test_zero_volume_cannot_be_revised_into_or_out_of_tradability(tmp_path, original_volume, outcome_volume):
    from stock_agent.pipeline.paper_scoring import score_paper_runs
    # A past zero-volume bar is source-valid; the signal bar must still be liquid.
    day = "2026-08-03"
    record(tmp_path, transform=lambda rows: revise_volume(rows, original_volume, day))
    current = rolling_snapshot(tmp_path / "current", transform=lambda rows: revise_volume(rows, outcome_volume, day))
    result = score_paper_runs(tmp_path / "out", current.parent / "prices", manifest_path=current)
    if original_volume == outcome_volume:
        assert result["status"] == "ok" and len(result["records"]) == 1
    else:
        assert result["status"] == "blocked" and result["records"] == []
    assert result["revision_warnings"] == []


@pytest.mark.parametrize("column,value", [("volume", -1), ("volume", float("nan")),
                                         ("volume", float("inf")), ("extra", 2)])
def test_history_comparison_refuses_invalid_volume_or_nonvolume_change(column, value):
    from stock_agent.pipeline.paper_scoring import _check_history
    old = pd.DataFrame({"date": ["2026-09-21"], "close": [10000], "volume": [1000000.0], "extra": [1]})
    frame = old.copy()
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match="history revision"):
        _check_history(old, frame, symbol="AAA", requested_start="2026-09-21", session="2026-09-21")
