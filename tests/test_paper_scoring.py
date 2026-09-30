from datetime import date, datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pytest

from stock_agent.pipeline import paper_runner as runner
from stock_agent.data import reconciliation as rec
from tests.test_market_runtime_hardening import make_snapshot


def record(tmp_path, track="mr"):
    manifest = make_snapshot(tmp_path / "original")
    now = datetime(2026, 9, 21, 10, tzinfo=timezone.utc)
    payload = runner.run_paper(manifest.parent / "prices", ["AAA"], manifest_path=manifest, now=now)
    close = float(pd.read_csv(manifest.parent / "prices/AAA.csv").close.iloc[-1])
    payload["recommendations"] = [{"track": track, "symbol": "AAA", "date": "2026-09-21", "close": close,
        "entry_reference": close, "stop_loss": close - 500, "take_profit": close + 2000, "max_hold_days": 15}]
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


def rolling_snapshot(root, start=date(2025, 2, 5), transform=None):
    """Keep absolute prices fixed while changing the provider request boundary."""
    full = make_snapshot(root / "full", end=date(2026, 9, 25))

    def fetch(symbol, requested_start, end):
        source = json.loads((full.parent / "raw" / f"{symbol}.json").read_bytes())
        if transform:
            source = transform(source)
        raw = json.dumps(source).encode()
        return rec.parse_vci_history(source, symbol, requested_start, end), {
            "raw_bytes": raw, "fetched_at": "2026-09-25T10:00:00+00:00"}

    rec.build_market_snapshot(["AAA", "VNINDEX"], root / "rolling", start,
                              date(2026, 9, 25), min_rows=1, fetcher=fetch)
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


@pytest.mark.parametrize("column", ["o", "h", "l", "c", "v"])
def test_rolled_request_does_not_excuse_real_shared_history_revisions(tmp_path, column):
    from stock_agent.pipeline.paper_scoring import score_paper_runs
    record(tmp_path)

    def revise(items):
        if items[0]["symbol"] == "AAA":
            items[0][column][-5] += 1  # Signal date, also tests GAS-style volume revisions.
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
