"""Performance Tracker — Rolling model performance metrics and drift monitoring.

Tracks daily model predictions vs. actual outcomes to detect performance
degradation (drift) and produce weekly summaries.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

PERF_LOG_PATH = Path("data/pipeline/performance_log.jsonl")
WEEKLY_REPORT_PATH = Path("data/pipeline/weekly_reports")


def log_daily_performance(
    scan_date: date,
    total_symbols: int,
    ml_buy_count: int,
    ml_watch_count: int,
    ml_reject_count: int,
    avg_probability: float,
    ensemble_agreement_avg: float,
    drift_status: str,
    model_vintage: str,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Log daily performance entry."""
    PERF_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "date": str(scan_date),
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "total_symbols": total_symbols,
        "ml_buy_count": ml_buy_count,
        "ml_watch_count": ml_watch_count,
        "ml_reject_count": ml_reject_count,
        "avg_probability": round(avg_probability, 4),
        "ensemble_agreement_avg": round(ensemble_agreement_avg, 3),
        "drift_status": drift_status,
        "model_vintage": model_vintage,
        "metrics": metrics or {},
    }
    with PERF_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_performance_history(last_n_days: int = 60) -> list[dict]:
    """Load last N days of performance logs."""
    if not PERF_LOG_PATH.exists():
        return []

    records = []
    with PERF_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if last_n_days and records:
        cutoff = str(date.today().isoformat())
        records = records[-last_n_days:]

    return records


def compute_rolling_metrics(
    resolved_outcomes: list[dict],
    window: int = 20,
) -> dict[str, Any]:
    """Compute rolling performance metrics from resolved outcomes.

    Parameters
    ----------
    resolved_outcomes : list[dict]
        Each dict has: predicted_win, actual_win, net_return_pct
    """
    if not resolved_outcomes:
        return {"status": "no_data"}

    recent = resolved_outcomes[-window:]
    predicted_wins = [o for o in recent if o.get("predicted_win")]
    all_returns = [o.get("net_return_pct", 0.0) for o in predicted_wins]

    if not predicted_wins:
        return {
            "status": "no_predictions",
            "window": window,
            "total_outcomes": len(recent),
        }

    correct = sum(1 for o in predicted_wins if o.get("actual_win"))
    win_rate = correct / len(predicted_wins) * 100 if predicted_wins else 0

    gains = sum(r for r in all_returns if r > 0)
    losses = abs(sum(r for r in all_returns if r < 0))
    profit_factor = gains / losses if losses > 0 else (float("inf") if gains > 0 else 0.0)

    avg_return = np.mean(all_returns) if all_returns else 0.0
    std_return = np.std(all_returns) if len(all_returns) > 1 else 0.0
    sharpe = avg_return / std_return if std_return > 0 else 0.0

    return {
        "status": "ok",
        "window": window,
        "total_outcomes": len(recent),
        "total_predictions": len(predicted_wins),
        "correct_predictions": correct,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else None,
        "avg_return_pct": round(float(avg_return), 4),
        "sharpe_ratio": round(float(sharpe), 4),
    }


def generate_weekly_report(
    resolved_outcomes: list[dict],
    performance_history: list[dict],
) -> dict[str, Any]:
    """Generate a weekly performance summary."""
    WEEKLY_REPORT_PATH.mkdir(parents=True, exist_ok=True)

    week_outcomes = resolved_outcomes[-5:] if resolved_outcomes else []  # ~1 trading week
    month_outcomes = resolved_outcomes[-20:] if resolved_outcomes else []

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weekly_metrics": compute_rolling_metrics(week_outcomes, window=5),
        "monthly_metrics": compute_rolling_metrics(month_outcomes, window=20),
        "total_predictions_all_time": len(resolved_outcomes),
        "model_health": _assess_health(month_outcomes),
        "recommendations": _generate_recommendations(month_outcomes, performance_history),
    }

    report_path = WEEKLY_REPORT_PATH / f"weekly_{date.today().isoformat()}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def _assess_health(outcomes: list[dict]) -> str:
    """Assess overall model health."""
    if not outcomes:
        return "unknown"

    predicted_wins = [o for o in outcomes if o.get("predicted_win")]
    if not predicted_wins:
        return "no_predictions"

    correct = sum(1 for o in predicted_wins if o.get("actual_win"))
    win_rate = correct / len(predicted_wins) * 100

    if win_rate >= 55:
        return "healthy"
    elif win_rate >= 45:
        return "warning"
    else:
        return "degraded"


def _generate_recommendations(
    outcomes: list[dict],
    history: list[dict],
) -> list[str]:
    """Generate actionable recommendations."""
    recommendations = []

    if not outcomes:
        recommendations.append("Chưa có đủ dữ liệu prediction outcomes. Tiếp tục thu thập.")
        return recommendations

    predicted_wins = [o for o in outcomes if o.get("predicted_win")]
    if not predicted_wins:
        recommendations.append("Model không chọn trade nào trong 20 phiên gần nhất. Kiểm tra threshold.")
        return recommendations

    correct = sum(1 for o in predicted_wins if o.get("actual_win"))
    win_rate = correct / len(predicted_wins) * 100

    if win_rate < 45:
        recommendations.append("Win rate dưới 45%. Nên chạy full retrain + HPO cuối tuần.")
    if win_rate < 35:
        recommendations.append("Win rate nghiêm trọng thấp. Tạm disable ML override, chỉ dùng advisory.")

    all_returns = [o.get("net_return_pct", 0) for o in predicted_wins]
    avg_return = np.mean(all_returns) if all_returns else 0
    if avg_return < -0.5:
        recommendations.append(f"Avg return {avg_return:.2f}% < -0.5%. Xem xét tăng probability_threshold.")

    selection_rate = len(predicted_wins) / len(outcomes) * 100 if outcomes else 0
    if selection_rate > 40:
        recommendations.append(f"Selection rate {selection_rate:.0f}% quá cao. Nên tăng threshold để chọn ít hơn, chính xác hơn.")
    elif selection_rate < 5:
        recommendations.append(f"Selection rate {selection_rate:.0f}% quá thấp. Xem xét giảm threshold.")

        recommendations.append("Model hoạt động tốt. Tiếp tục monitoring.")

    return recommendations


MANIFEST_LOG_PATH = Path("data/pipeline/experiment_manifests.jsonl")
MANIFEST_DIR = Path("data/pipeline/manifests")


def log_experiment_manifest(manifest: Any) -> Path:
    """Log an experiment manifest to the central log and save it as an individual JSON file."""
    MANIFEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    clean_ts = manifest.timestamp.replace(":", "").replace("+", "Z")
    individual_path = MANIFEST_DIR / f"manifest_{clean_ts}.json"

    from ..schemas import to_plain_dict
    manifest_dict = to_plain_dict(manifest)

    with individual_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, ensure_ascii=False, indent=2)

    with MANIFEST_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(manifest_dict, ensure_ascii=False) + "\n")

    logger.info(f"Saved experiment manifest to {individual_path}")
    return individual_path


def create_and_log_manifest(
    command: str,
    rules: dict,
    data_start: str,
    data_end: str,
    symbols: list[str],
    report_links: dict[str, str],
    symbols_excluded: list[str] | None = None,
    data_hash: str = "",
    data_hashes: dict[str, str] | None = None,
) -> Path:
    """Create an ExperimentManifest and log it."""
    from ..schemas import ExperimentManifest
    from ..config import compute_rules_hash

    manifest = ExperimentManifest(
        command=command,
        timestamp=datetime.now(timezone.utc).isoformat(),
        rules_hash=compute_rules_hash(rules),
        data_start=data_start,
        data_end=data_end,
        symbols=symbols,
        symbols_excluded=symbols_excluded or [],
        report_links=report_links,
        data_hash=data_hash,
        data_hashes=data_hashes or {},
    )
    return log_experiment_manifest(manifest)

