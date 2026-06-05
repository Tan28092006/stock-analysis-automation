from __future__ import annotations

from typing import Any


def detect_contradictions(candidate: dict[str, Any]) -> list[str]:
    """Detect logical contradictions in a scan candidate.

    These are rule-based checks (no LLM needed) that highlight
    inconsistencies between score, risk plan, data quality, and decision.
    Results are injected into the AI prompt so the analyst can comment.
    """
    issues: list[str] = []
    score = candidate.get("score", 0)
    decision = candidate.get("decision", "")
    risk_plan = candidate.get("risk_plan") or {}
    data_quality = candidate.get("data_quality") or {}
    warnings = candidate.get("warnings") or []

    rr = risk_plan.get("reward_risk", 0)
    sl_pct = risk_plan.get("stop_loss_pct", 0)
    entry = risk_plan.get("entry_reference", 0)
    tp1 = risk_plan.get("take_profit_1", 0)
    dq_status = data_quality.get("status", "")
    cross_check = data_quality.get("cross_check_status", "")

    # 1. High score but weak risk/reward
    if score >= 80 and rr < 1.5:
        issues.append(
            f"Score cao ({score:.0f}) nhưng R:R thấp ({rr:.2f}) — "
            f"upside có thể không đủ bù rủi ro"
        )

    # 2. BUY_SETUP with degraded data
    if decision == "BUY_SETUP" and dq_status == "degraded":
        issues.append(
            f"Decision BUY_SETUP nhưng data quality \"{dq_status}\" "
            f"(cross-check: {cross_check}) — cần cẩn thận"
        )

    # 3. Stop loss too wide for T+2
    if sl_pct > 4.5 and decision in ("BUY_SETUP", "WATCH"):
        issues.append(
            f"Stop loss {sl_pct:.1f}% quá rộng cho chiến lược T+2 "
            f"(ngưỡng khuyến nghị ≤ 4.5%)"
        )

    # 4. TP1 barely above entry — risk/reward meaningless
    if entry > 0 and tp1 > 0:
        tp1_pct = (tp1 - entry) / entry * 100
        if tp1_pct < 1.0 and decision in ("BUY_SETUP", "WATCH"):
            issues.append(
                f"TP1 chỉ cách entry {tp1_pct:.2f}% — target quá gần, "
                f"phí + slippage có thể ăn hết lợi nhuận"
            )

    # 5. Many failed evidence but still BUY
    evidence = candidate.get("evidence") or []
    failed_count = sum(1 for ev in evidence if not ev.get("passed", True))
    total_count = len(evidence)
    if total_count > 0 and failed_count >= total_count * 0.5 and decision == "BUY_SETUP":
        issues.append(
            f"{failed_count}/{total_count} evidence rules FAIL nhưng vẫn BUY_SETUP "
            f"— kiểm tra lại scoring weights"
        )

    # 6. Single source data for BUY
    if cross_check == "single_source" and decision == "BUY_SETUP":
        issues.append(
            "Chỉ có 1 nguồn dữ liệu (single source) — "
            "không có cross-check, confidence bị giới hạn"
        )

    # 7. Warning flood
    if len(warnings) >= 4 and decision in ("BUY_SETUP", "WATCH"):
        issues.append(
            f"Có {len(warnings)} cảnh báo — nhiều red flags đồng thời"
        )

    return issues
