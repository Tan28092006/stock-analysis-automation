"""Label Resolver — Automatically label past predictions with actual T+2 outcomes.

After T+2 trading days, we can check whether the prediction was correct by
comparing the actual exit price to the entry price.  This module resolves
pending predictions and appends them to the training dataset.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from ..constants import PRICE_CACHE_DIR
from ..data.exchange_calendar import add_trading_days, next_trading_day
from ..data.validation import normalize_ohlcv
from ..features.backtest import BacktestConfig, enforce_price_limit, round_trip_cost_pct

logger = logging.getLogger(__name__)

PENDING_PREDICTIONS_PATH = Path("data/pipeline/pending_predictions.jsonl")
RESOLVED_LABELS_PATH = Path("data/pipeline/resolved_labels.jsonl")


def log_prediction(
    symbol: str,
    signal_date: date,
    decision: str,
    ml_probability: float | None,
    ml_passed: bool | None,
    score: float,
    entry_reference: float,
) -> None:
    """Log a prediction for future label resolution."""
    PENDING_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "symbol": symbol,
        "signal_date": str(signal_date),
        "decision": decision,
        "ml_probability": ml_probability,
        "ml_passed": ml_passed,
        "score": score,
        "entry_reference": entry_reference,
        "status": "pending",
    }
    with PENDING_PREDICTIONS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_pending_labels(
    rules: dict,
    price_dir: Path = PRICE_CACHE_DIR,
    today: date | None = None,
) -> dict[str, Any]:
    """Resolve pending predictions whose T+2 outcome is now known.

    Returns summary of how many were resolved.
    """
    today = today or date.today()
    cost_config = BacktestConfig.from_rules(rules)
    cost_pct = round_trip_cost_pct(cost_config)

    if not PENDING_PREDICTIONS_PATH.exists():
        return {"status": "no_pending", "resolved": 0}

    # Load pending
    pending: list[dict] = []
    with PENDING_PREDICTIONS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pending.append(json.loads(line))

    if not pending:
        return {"status": "no_pending", "resolved": 0}

    still_pending: list[dict] = []
    resolved_count = 0
    RESOLVED_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)

    for record in pending:
        if record.get("status") != "pending":
            continue

        signal_date = date.fromisoformat(record["signal_date"])
        entry_date = next_trading_day(signal_date)
        exit_date = add_trading_days(entry_date, cost_config.holding_days)

        # Check if exit_date has passed
        if exit_date > today:
            still_pending.append(record)
            continue

        # Load price data
        symbol = record["symbol"]
        csv_path = price_dir / f"{symbol}.csv"
        if not csv_path.exists():
            still_pending.append(record)
            continue

        try:
            df = normalize_ohlcv(pd.read_csv(csv_path))
            date_to_idx = {d: i for i, d in enumerate(df["date"])}

            entry_idx = date_to_idx.get(entry_date)
            exit_idx = date_to_idx.get(exit_date)

            if entry_idx is None or exit_idx is None:
                still_pending.append(record)
                continue

            entry_reference = record["entry_reference"]
            entry_price = enforce_price_limit(
                float(df.loc[entry_idx, "open"]), entry_reference, cost_config.price_limit_pct
            )
            exit_reference_idx = max(exit_idx - 1, 0)
            exit_reference = float(df.loc[exit_reference_idx, "close"])
            exit_price = enforce_price_limit(
                float(df.loc[exit_idx, "close"]), exit_reference, cost_config.price_limit_pct
            )

            gross_return = (exit_price - entry_price) / entry_price * 100.0 if entry_price else 0.0
            net_return = gross_return - cost_pct

            resolved = {
                **record,
                "status": "resolved",
                "entry_date": str(entry_date),
                "exit_date": str(exit_date),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "gross_return_pct": round(gross_return, 4),
                "net_return_pct": round(net_return, 4),
                "actual_win": int(net_return > 0),
                "predicted_win": bool(record.get("ml_passed", False)),
            }

            with RESOLVED_LABELS_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(resolved, ensure_ascii=False) + "\n")

            resolved_count += 1
            logger.info(f"Resolved {symbol} {signal_date}: net_return={net_return:.2f}%")

        except Exception as exc:
            logger.warning(f"Failed to resolve {symbol} {signal_date}: {exc}")
            still_pending.append(record)

    # Rewrite pending file with only unresolved items
    with PENDING_PREDICTIONS_PATH.open("w", encoding="utf-8") as f:
        for record in still_pending:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "status": "ok",
        "resolved": resolved_count,
        "still_pending": len(still_pending),
    }


def load_resolved_labels() -> pd.DataFrame:
    """Load all resolved labels as a DataFrame."""
    if not RESOLVED_LABELS_PATH.exists():
        return pd.DataFrame()

    records = []
    with RESOLVED_LABELS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "signal_date" in df.columns:
        df["signal_date"] = pd.to_datetime(df["signal_date"]).dt.date
    return df
