from __future__ import annotations

from ..schemas import ScanResult


def render_markdown(result: ScanResult) -> str:
    lines = [
        f"# Scan {result.scan_id}",
        "",
        f"- Mode: {result.mode}",
        f"- Created: {result.created_at}",
        f"- Universe: {result.universe_name}",
        f"- Symbols scanned: {result.symbols_scanned}",
        f"- Rejected: {result.rejected_count}",
        "",
        "## Candidates",
    ]
    for item in result.candidates:
        lines.extend(
            [
                "",
                f"### {item.symbol} - {item.decision}",
                f"- Score: {item.score}",
                f"- Confidence: {item.confidence}",
                f"- Latest close: {item.latest_close} ({item.latest_date})",
                f"- Allocation weight: {item.allocation_weight:.2%}",
                f"- Data quality: {item.data_quality.status} / {item.data_quality.cross_check_status}",
            ]
        )
        if item.risk_plan:
            lines.append(
                f"- Risk: SL {item.risk_plan.stop_loss}, TP1 {item.risk_plan.take_profit_1}, "
                f"TP2 {item.risk_plan.take_profit_2}, RR {item.risk_plan.reward_risk}"
            )
        if item.warnings:
            lines.append(f"- Warnings: {'; '.join(item.warnings)}")
        passed = [ev.name for ev in item.evidence if ev.passed]
        if passed:
            lines.append(f"- Passed evidence: {', '.join(passed)}")
    if result.warnings:
        lines.extend(["", "## Data Warnings"])
        lines.extend([f"- {warning}" for warning in result.warnings])
    return "\n".join(lines) + "\n"

