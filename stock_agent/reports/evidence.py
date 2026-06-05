from __future__ import annotations

import re
from typing import Any

ALLOWED_DECISIONS = {"BUY_SETUP", "WATCH", "REJECT"}


def build_evidence_bundle(scan_payload: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for item in scan_payload.get("candidates", []):
        candidates.append(
            {
                "symbol": item["symbol"],
                "decision": item["decision"],
                "score": item["score"],
                "confidence": item["confidence"],
                "latest_close": item["latest_close"],
                "latest_date": item["latest_date"],
                "allocation_weight": item.get("allocation_weight", 0.0),
                "data_quality": item.get("data_quality", {}),
                "risk_plan": item.get("risk_plan"),
                "warnings": item.get("warnings", []),
                "evidence": [
                    {
                        "evidence_id": ev["evidence_id"],
                        "name": ev["name"],
                        "passed": ev["passed"],
                        "detail": ev["detail"],
                        "value": ev.get("value"),
                    }
                    for ev in item.get("evidence", [])
                ],
            }
        )
    return {
        "scan_id": scan_payload.get("scan_id"),
        "created_at": scan_payload.get("created_at"),
        "mode": scan_payload.get("mode"),
        "universe_name": scan_payload.get("universe_name"),
        "symbols_scanned": scan_payload.get("symbols_scanned"),
        "candidates": candidates,
        "data_warnings": scan_payload.get("warnings", []),
        "rules": {
            "llm_may_not_change_decision": True,
            "llm_may_not_create_evidence": True,
            "llm_may_not_create_prices": True,
        },
    }


def validate_ai_summary(ai_payload: dict[str, Any], evidence_bundle: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    by_symbol = {item["symbol"]: item for item in evidence_bundle.get("candidates", [])}
    known_evidence_ids = {
        ev["evidence_id"]
        for item in evidence_bundle.get("candidates", [])
        for ev in item.get("evidence", [])
    }

    if ai_payload.get("scan_id") != evidence_bundle.get("scan_id"):
        errors.append("scan_id mismatch")

    for item in ai_payload.get("candidates", []):
        symbol = item.get("symbol")
        decision = item.get("decision")
        if symbol not in by_symbol:
            errors.append(f"unknown symbol in AI output: {symbol}")
            continue
        if decision not in ALLOWED_DECISIONS:
            errors.append(f"{symbol}: invalid decision {decision}")
        if decision != by_symbol[symbol]["decision"]:
            errors.append(f"{symbol}: AI changed decision from {by_symbol[symbol]['decision']} to {decision}")
        for evidence_id in item.get("evidence_ids", []):
            if evidence_id not in known_evidence_ids:
                errors.append(f"{symbol}: unknown evidence_id {evidence_id}")
    return errors


def _extract_numbers(text: str) -> list[float]:
    """Extract numeric values from text (integers and decimals)."""
    matches = re.findall(r"(?<![a-zA-Z_])\d[\d,.]*\d|\b\d+\b", text)
    numbers = []
    for m in matches:
        cleaned = m.replace(",", "")
        try:
            numbers.append(float(cleaned))
        except ValueError:
            continue
    return numbers


def _build_allowed_numbers(candidate: dict[str, Any]) -> set[float]:
    """Collect all legitimate numbers from evidence bundle for a candidate."""
    allowed: set[float] = set()

    # Basic fields
    for key in ("score", "confidence", "latest_close", "allocation_weight"):
        val = candidate.get(key)
        if val is not None:
            allowed.add(float(val))

    # Risk plan numbers
    rp = candidate.get("risk_plan") or {}
    for key in ("entry_reference", "stop_loss", "take_profit_1", "take_profit_2",
                "stop_loss_pct", "reward_risk", "holding_period_days"):
        val = rp.get(key)
        if val is not None:
            allowed.add(float(val))

    # Evidence values
    for ev in candidate.get("evidence", []):
        val = ev.get("value")
        if isinstance(val, (int, float)):
            allowed.add(float(val))
        elif isinstance(val, dict):
            for v in val.values():
                if isinstance(v, (int, float)):
                    allowed.add(float(v))

    # Common constants that are always OK
    allowed.update({0, 1, 2, 3, 5, 7, 9, 14, 20, 21, 26, 50, 100})

    return allowed


def validate_reason_content(
    analysis: dict[str, Any],
    candidate: dict[str, Any],
    tolerance: float = 0.05,
) -> list[str]:
    """Check if AI analysis text contains fabricated numbers.

    Cross-references all numbers found in summary/risk_notes against
    allowed values from the evidence bundle.
    """
    errors: list[str] = []
    text = " ".join([
        analysis.get("summary", ""),
        analysis.get("risk_notes", ""),
        analysis.get("confidence_note", ""),
    ])

    numbers = _extract_numbers(text)
    allowed = _build_allowed_numbers(candidate)

    for num in numbers:
        if num == 0:
            continue
        # Check if number is close to any allowed value
        matched = False
        for allowed_val in allowed:
            if allowed_val == 0:
                continue
            if abs(num - allowed_val) / max(abs(allowed_val), 1e-9) <= tolerance:
                matched = True
                break
        if not matched:
            # Only flag large numbers (prices, volumes) — small integers are usually OK
            if num > 100 or (num > 10 and "." in str(num)):
                errors.append(
                    f"{analysis.get('symbol', '?')}: possible fabricated number "
                    f"{num} in AI text (no match in evidence bundle)"
                )

    return errors


