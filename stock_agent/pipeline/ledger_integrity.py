"""Conservative eligibility checks; unknown provenance is not evidence of a loss.

Never infer old model versions, horizons or append times from today's configuration.
Quarantine here is a derived classification; original records remain untouched.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime, time, timezone
import json
import math

import pandas as pd

from ..data.exchange_calendar import VN_TIMEZONE, next_trading_day, is_trading_day


def provenance_reasons(record: dict) -> list[str]:
    reasons = []
    required = ("recorded_at", "mode", "model_version", "rules_hash", "input_snapshot", "data_source")
    if record.get("schema_version") != 2 or any(not record.get(k) for k in required):
        reasons.append("missing_provenance")
    if record.get("mode") not in {None, "paper"}:
        reasons.append("non_paper_record")
    if record.get("data_source") not in {None, "prices_hist"}:
        reasons.append("unverified_price_source")
    try:
        sd = date.fromisoformat(record["signal_date"])
        if not is_trading_day(sd):
            reasons.append("non_trading_signal_date")
        if record.get("logged_date", str(sd)) != str(sd):
            reasons.append("stale_signal")
        stamp = datetime.fromisoformat(record["recorded_at"])
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("ambiguous timestamp")
        after_eod = datetime.combine(sd, time(16), VN_TIMEZONE)
        before_entry = datetime.combine(next_trading_day(sd), time(9), VN_TIMEZONE)
        if not after_eod <= stamp < before_entry or stamp > datetime.now(timezone.utc):
            reasons.append("not_recorded_in_ex_ante_window")
        if record.get("win_prob") is not None or record.get("ml_probability") is not None:
            trained = datetime.fromisoformat(record["model_trained_at"])
            if trained.tzinfo is None or trained > stamp:
                reasons.append("model_not_available_at_prediction")
    except (KeyError, TypeError, ValueError):
        reasons.append("unverifiable_timing")
    return reasons


def price_reasons(record: dict, frame: pd.DataFrame | None) -> list[str]:
    if frame is None or frame.empty:
        return ["missing_prices"]
    sd = record.get("signal_date")
    matched = frame[frame["date"].astype(str).str[:10] == sd]
    if len(matched) != 1:
        return ["missing_or_duplicate_signal_bar"]
    try:
        ref = float(record.get("entry_reference") or record.get("close"))
        close = float(matched.iloc[0]["close"])
        if not (math.isfinite(ref) and math.isfinite(close) and ref > 0 and close > 0):
            return ["invalid_reference"]
        if abs(ref / close - 1) > .05:
            return ["reference_price_mismatch"]
    except (TypeError, ValueError):
        return ["invalid_reference"]
    return []


def audit_rows(rows: list[dict], frame_for, *, legacy: bool = False) -> dict:
    groups = defaultdict(list)
    for line, r in enumerate(rows, 1):
        key = (r.get("engine", "legacy"), r.get("symbol"), r.get("signal_date"), r.get("kind", ""))
        groups[key].append((line, r))
    classified = []
    for group in groups.values():
        canonical = {json.dumps(r, sort_keys=True, ensure_ascii=False) for _, r in group}
        for j, (line, r) in enumerate(group):
            reasons = provenance_reasons(r) + price_reasons(r, frame_for(r.get("symbol", "")))
            if legacy and not r.get("label_contract"):
                reasons.append("unknown_original_label_contract")
            if len(canonical) > 1:
                reasons.append("conflicting_duplicate")
            elif j:
                reasons.append("duplicate")
            classified.append({"source_line": line, "eligible": not reasons,
                               "reasons": sorted(set(reasons)), "record": r})
    classified.sort(key=lambda r: r["source_line"])
    return {"rows": len(rows), "eligible": sum(r["eligible"] for r in classified),
            "quarantined": sum(not r["eligible"] for r in classified),
            "reason_counts": dict(Counter(reason for r in classified for reason in r["reasons"])),
            "records": classified}
