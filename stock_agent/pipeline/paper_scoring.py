"""Read-only paper observations: delayed outcomes, never broker fills or account P&L."""
from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from ..data.exchange_calendar import symbol_trading_days_between
from ..data.reconciliation import verify_snapshot
from ..features.mr_exit import simulate_mr_exit
from .paper_runner import _digest, _source, _window

COST_PCT = .6
IDENTITY_KEYS = ("session", "input_snapshot", "rules_hash", "code_hash", "universe", "recommendations", "model")


def _outcome(signal: dict, frame: pd.DataFrame, session: str) -> dict:
    symbol, track = signal["symbol"], signal["track"]
    if signal.get("date") != session or track not in {"mr", "momentum"}:
        raise ValueError("Unknown track or signal date mismatch")
    hits = frame.index[frame.date == session]
    if len(hits) != 1:
        raise ValueError(f"{symbol}: original signal session unavailable")
    i = int(hits[0])
    close = float(frame.at[i, "close"])
    if not math.isclose(float(signal["close"]), close, rel_tol=0, abs_tol=.011):
        raise ValueError(f"{symbol}: recorded close differs from verified source")
    row = {"session": session, "symbol": symbol, "track": track, "status": "pending",
           "net_return_pct": None, "reason": None,
           "metric": "mr_daily_bar_simulation_net_pct" if track == "mr" else "informational_close_return_21_sessions"}
    if track == "momentum":
        if i + 21 < len(frame):
            row.update(status="resolved", net_return_pct=(float(frame.at[i + 21, "close"]) / close - 1) * 100,
                       exit_date=str(frame.at[i + 21, "date"]))
        return row
    stop, target = float(signal["stop_loss"]), float(signal["take_profit"])
    hold = signal["max_hold_days"]
    if (not all(math.isfinite(v) for v in (stop, target, close)) or not 0 < stop < close < target
            or isinstance(hold, bool) or not isinstance(hold, int) or not 2 <= hold <= 60):
        raise ValueError(f"{symbol}: invalid recorded risk plan")
    if i + 1 >= len(frame):
        return row
    entry = float(frame.at[i + 1, "open"])
    row.update(entry_price=entry, entry_date=str(frame.at[i + 1, "date"]))
    if entry <= stop or entry >= target or float(frame.at[i + 1, "volume"]) <= 0:
        row.update(status="not_entered", reason="next_open_outside_risk_plan_or_no_volume")
        return row
    j, exit_price, reason, resolved = simulate_mr_exit(frame, i + 1, stop, target, hold, settle_lock=2)
    if resolved:
        # Gap-down stops cannot assume a fill at a price above the opening market.
        if reason == "stop":
            exit_price = min(exit_price, float(frame.at[j, "open"]))
        if float(frame.at[j, "volume"]) <= 0 or float(frame.at[j, "high"]) == float(frame.at[j, "low"]):
            row.update(reason="execution_unverifiable_on_zero_volume_or_locked_bar")
            return row
        row.update(status="resolved", exit_price=exit_price, exit_date=str(frame.at[j, "date"]),
                   reason=reason, net_return_pct=(exit_price / entry - 1) * 100 - COST_PCT)
    return row


def score_paper_runs(output_dir: Path, prices_dir: Path, *, manifest_path: Path) -> dict:
    """Return a separate report; original records and snapshots are never changed.

    Hashes detect accidental corruption, not malicious edits to both data and hash.
    Daily OHLC cannot prove intraday T+2 availability, queue fills or liquidity.
    """
    manifest_path = Path(manifest_path).resolve()
    current = verify_snapshot(manifest_path)
    prices_dir = Path(prices_dir).resolve()
    if prices_dir != manifest_path.parent / "prices":
        raise ValueError("Outcome manifest/prices directory mismatch")
    result = {"status": "ok", "as_of": current["as_of"], "records": [], "errors": [],
              "pending": 0, "resolved": 0, "not_entered": 0, "runs": 0,
              "outcome_manifest": str(manifest_path),
              "outcome_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "assumptions": {"cost_pct": COST_PCT, "entry": "next_open", "settle_lock_bars": 2,
                              "stop_priority": True, "gap_stop": "min(stop,open)",
                              "actual_execution_verified": False, "portfolio_pnl": False}}
    for path in sorted((Path(output_dir) / "runs").glob("*/paper.json")):
        result["runs"] += 1
        try:
            paper = json.loads(path.read_text(encoding="utf-8"))
            session = paper["session"]
            recorded = datetime.fromisoformat(paper["recorded_at"])
            if (paper.get("mode") != "paper" or not paper.get("source_verified")
                    or not paper.get("data_ready") or path.parent.name != session
                    or recorded.tzinfo is None or not _window(date.fromisoformat(session), recorded)):
                raise ValueError("Not an ex-ante verified paper record")
            if _digest({k: paper.get(k) for k in IDENTITY_KEYS}) != paper["record_identity"]:
                raise ValueError("Paper record identity mismatch")
            original_path = Path(paper["snapshot_manifest"])
            checked = _source(original_path.parent / "prices", original_path, paper["universe"], session, now=recorded)
            if any(checked[key] != paper.get(key) for key in checked):
                raise ValueError("Original source provenance changed")
            staged, seen = [], set()
            for signal in paper["recommendations"]:
                symbol = signal["symbol"]
                key = (signal["track"], symbol)
                if key in seen or symbol not in paper["universe"] or symbol not in current["files"]:
                    raise ValueError("Duplicate signal or incomplete outcome universe")
                seen.add(key)
                old = pd.read_csv(original_path.parent / "prices" / f"{symbol}.csv")
                frame = pd.read_csv(prices_dir / f"{symbol}.csv")
                # Any revision to the original feature history quarantines this run;
                # do not mix adjusted future prices with an older signal price basis.
                matched = frame.set_index("date").reindex(old.date).reset_index()
                pd.testing.assert_frame_equal(old, matched, check_dtype=False, check_exact=True)
                segment = frame[frame.date >= session]
                expected = {str(d) for d in symbol_trading_days_between(symbol, date.fromisoformat(session), date.fromisoformat(current["as_of"]))}
                if set(segment.date) != expected:
                    raise ValueError(f"{symbol}: missing outcome sessions")
                staged.append(_outcome(signal, frame, session))
            result["records"].extend(staged)
        except Exception as exc:
            result["errors"].append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    for row in result["records"]:
        result[row["status"]] += 1
    result["status"] = "blocked" if result["errors"] else "ok"
    return result
