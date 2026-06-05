"""Daily Runner — Automated pipeline that runs every trading day.

Pipeline: fetch → label → retrain → predict → scan → report

Usage
-----
CLI:  python -m stock_agent.cli daily-run
Task: schtasks /create /tn "VN30_DailyPipeline" /tr "python -m stock_agent.cli daily-run" /sc daily /st 16:00
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..config import compute_rules_hash, load_rules, load_universe
from ..constants import PRICE_CACHE_DIR
from ..data.exchange_calendar import is_trading_day
from ..features.calibration import feature_columns
from ..features.ensemble_model import EnsembleConfig, EnsembleTrainer
from ..features.feature_engineering_v2 import get_v2_feature_columns
from .label_resolver import resolve_pending_labels, load_resolved_labels
from .performance_tracker import (
    compute_rolling_metrics,
    log_daily_performance,
    generate_weekly_report,
    load_performance_history,
)

logger = logging.getLogger(__name__)

PIPELINE_DIR = Path("data/pipeline")
DAILY_LOG_PATH = PIPELINE_DIR / "daily_runs.jsonl"


class DailyRunner:
    """Orchestrates the daily ML pipeline."""

    def __init__(
        self,
        rules: dict | None = None,
        demo: bool = False,
        force: bool = False,
        on_progress: Callable[[str, str], None] | None = None,
    ):
        self.rules = rules or load_rules()
        self.demo = demo
        self.force = force
        self.on_progress = on_progress or (lambda stage, msg: logger.info(f"[{stage}] {msg}"))
        self.today = date.today()

    def run(self) -> dict[str, Any]:
        """Execute the full daily pipeline."""
        start_time = datetime.now(timezone.utc)
        PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

        if not self.force and not self.demo and not is_trading_day(self.today):
            return {"status": "skipped", "reason": "not a trading day", "date": str(self.today)}

        result: dict[str, Any] = {
            "date": str(self.today),
            "started_at": start_time.isoformat(),
            "stages": {},
        }

        try:
            # Stage 1: Resolve pending labels from T-2
            self.on_progress("LABEL", "Resolving T+2 outcomes for past predictions...")
            label_result = self._resolve_labels()
            result["stages"]["labels"] = label_result

            # Stage 2: Build/expand training dataset
            self.on_progress("DATASET", "Building training dataset from all VN30 symbols...")
            dataset, v1_cols = self._build_dataset()
            result["stages"]["dataset"] = {
                "rows": len(dataset),
                "symbols": sorted(dataset["symbol"].unique().tolist()) if not dataset.empty else [],
            }

            if dataset.empty or len(dataset) < 200:
                result["status"] = "insufficient_data"
                result["stages"]["model"] = {"status": "skipped", "reason": f"only {len(dataset)} rows"}
                self._log_run(result, start_time)
                return result

            # Stage 3: Train or incrementally update ensemble
            self.on_progress("MODEL", "Training/updating ensemble model...")
            model_result = self._train_or_update(dataset, v1_cols)
            result["stages"]["model"] = model_result

            # Stage 4: Run scan with updated model
            self.on_progress("SCAN", "Running scan with updated ML model...")
            scan_result = self._run_scan()
            result["stages"]["scan"] = {
                "scan_id": scan_result.scan_id if hasattr(scan_result, "scan_id") else "unknown",
                "buy_count": sum(1 for c in scan_result.candidates if c.decision == "BUY_SETUP"),
                "watch_count": sum(1 for c in scan_result.candidates if c.decision == "WATCH"),
                "reject_count": scan_result.rejected_count,
            }

            # Stage 5: Log predictions for future label resolution
            self.on_progress("LOG", "Logging predictions for T+2 resolution...")
            self._log_predictions(scan_result)

            # Stage 6: Performance tracking
            self.on_progress("TRACK", "Computing performance metrics...")
            perf_result = self._track_performance()
            result["stages"]["performance"] = perf_result

            # Stage 7: Weekly report (if Friday)
            if self.today.weekday() == 4:  # Friday
                self.on_progress("WEEKLY", "Generating weekly performance report...")
                weekly = self._weekly_report()
                result["stages"]["weekly_report"] = weekly

            result["status"] = "completed"

        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
            logger.error(f"Daily pipeline error: {exc}", exc_info=True)

        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        result["duration_seconds"] = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()

        self._log_run(result, start_time)
        return result

    # -- Stage implementations -------------------------------------------

    def _resolve_labels(self) -> dict[str, Any]:
        return resolve_pending_labels(self.rules, price_dir=PRICE_CACHE_DIR, today=self.today)

    def _build_dataset(self) -> tuple[pd.DataFrame, list[str]]:
        from ..agents.parallel_engine import build_labeled_dataset_fast

        universe = load_universe()
        symbols = [s.upper() for s in universe["symbols"]]

        dataset = build_labeled_dataset_fast(
            symbols=symbols,
            rules=self.rules,
            price_dir=PRICE_CACHE_DIR,
        )

        v1_cols = feature_columns(dataset) if not dataset.empty else []
        return dataset, v1_cols

    def _train_or_update(self, dataset: pd.DataFrame, v1_cols: list[str]) -> dict[str, Any]:
        config = EnsembleConfig.from_rules(self.rules)
        all_feature_cols = v1_cols  # Start with v1 features; v2 will be added later

        try:
            # Try to load existing ensemble for incremental update
            trainer = EnsembleTrainer.load()
            self.on_progress("MODEL", "Incremental update with warm-start...")
            result = trainer.daily_update(dataset, label_col="net_t2_win")
            trainer.save()
            return result
        except FileNotFoundError:
            # No existing model — full train
            self.on_progress("MODEL", "No existing model found. Full training...")
            trainer = EnsembleTrainer(config=config)
            result = trainer.train(dataset, feature_cols=all_feature_cols, label_col="net_t2_win")
            if result.get("status") == "trained":
                trainer.save()
            return result

    def _run_scan(self):
        from ..agents.orchestrator import run_scan
        return run_scan(demo=self.demo, persist=True)

    def _log_predictions(self, scan_result) -> None:
        from .label_resolver import log_prediction

        for candidate in scan_result.candidates:
            if candidate.decision in {"BUY_SETUP", "WATCH"}:
                ml_prob = None
                ml_passed = None
                if candidate.model_signal:
                    ml_prob = candidate.model_signal.probability
                    ml_passed = candidate.model_signal.passed

                log_prediction(
                    symbol=candidate.symbol,
                    signal_date=self.today,
                    decision=candidate.decision,
                    ml_probability=ml_prob,
                    ml_passed=ml_passed,
                    score=candidate.score,
                    entry_reference=candidate.latest_close,
                )

    def _track_performance(self) -> dict[str, Any]:
        resolved = load_resolved_labels()
        if resolved.empty:
            return {"status": "no_resolved_data"}

        outcomes = resolved.to_dict("records")
        metrics = compute_rolling_metrics(outcomes, window=20)

        # Log daily performance
        log_daily_performance(
            scan_date=self.today,
            total_symbols=len(load_universe()["symbols"]),
            ml_buy_count=0,  # Will be updated from scan
            ml_watch_count=0,
            ml_reject_count=0,
            avg_probability=0.0,
            ensemble_agreement_avg=0.0,
            drift_status=metrics.get("status", "unknown"),
            model_vintage="",
            metrics=metrics,
        )

        return metrics

    def _weekly_report(self) -> dict[str, Any]:
        resolved = load_resolved_labels()
        outcomes = resolved.to_dict("records") if not resolved.empty else []
        history = load_performance_history()
        return generate_weekly_report(outcomes, history)

    def _log_run(self, result: dict, start_time: datetime) -> None:
        DAILY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DAILY_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
