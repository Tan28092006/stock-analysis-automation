import copy
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from stock_agent.ai.analyst import _analyze_single_stock
from stock_agent.agents.orchestrator import run_scan
from stock_agent.cli import build_parser
from stock_agent.config import compute_rules_hash, load_rules
from stock_agent.data.exchange_calendar import trading_days_between
from stock_agent.data.sample_data import make_demo_ohlcv
from stock_agent.features.backtest import BacktestConfig, run_portfolio_backtest
from stock_agent.features.indicators import add_indicators
from stock_agent.features import ml_models
from stock_agent.features.ml_models import predict_model_signal, train_model_suite_from_dataset
from stock_agent.features.robustness import run_robustness
from stock_agent.features.signal_engine import score_symbol
from stock_agent.schemas import RuleEvidence


class DeepResearchIndicatorTests(unittest.TestCase):
    def test_research_indicators_and_lags_are_available(self):
        df = make_demo_ohlcv("FPT", date(2026, 5, 27), rows=180)
        out = add_indicators(df)
        latest = out.iloc[-1]
        for col in ["sma20", "vwap20", "obv", "adl", "adx14", "close_lag_1", "rolling_std_20"]:
            self.assertIn(col, out.columns)
            self.assertTrue(np.isfinite(float(latest[col])))


class DeepResearchModelTests(unittest.TestCase):
    def _dataset(self, rows=90):
        dates = pd.bdate_range(start="2026-01-01", periods=rows).date
        records = []
        rules_version = compute_rules_hash(load_rules())
        for idx, day in enumerate(dates):
            alpha = int(idx % 4 in {0, 1})
            records.append(
                {
                    "symbol": "AAA" if idx % 2 == 0 else "BBB",
                    "signal_date": day,
                    "decision": "BUY_SETUP" if alpha else "WATCH",
                    "rules_version": rules_version,
                    "net_t2_win": alpha,
                    "net_t2_return_pct": 0.9 if alpha else -0.5,
                    "rule_alpha": alpha,
                    "feature_close": 100 + idx,
                    "feature_return_1d": alpha * 0.01,
                    "feature_rsi14": 55 + alpha,
                    "feature_volume_ratio_20": 1.1 + alpha * 0.3,
                    "feature_atr_pct": 0.02,
                    "feature_bb_width": 0.04,
                    "feature_score": 60 + alpha * 20,
                }
            )
        return pd.DataFrame(records)

    def test_train_and_predict_logistic_model_without_future_dependency(self):
        rules = copy.deepcopy(load_rules())
        rules["ml"]["min_train_rows"] = 20
        rules["ml"]["model_family"] = "logistic"
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = ml_models.MODEL_DIR
            old_registry = ml_models.MODEL_REGISTRY_PATH
            ml_models.MODEL_DIR = Path(tmp)
            ml_models.MODEL_REGISTRY_PATH = Path(tmp) / "registry.json"
            try:
                payload = train_model_suite_from_dataset(self._dataset(), rules, families=["logistic", "lstm"])
                self.assertEqual(payload["models"]["logistic"]["status"], "trained")
                self.assertIn(payload["models"]["lstm"]["status"], {"skipped", "failed"})

                signal = SimpleNamespace(
                    evidence=[RuleEvidence("AAA:alpha", "alpha", True, 1, "ok")],
                    features={
                        "close": 120.0,
                        "return_1d": 0.01,
                        "rsi14": 56.0,
                        "volume_ratio_20": 1.4,
                        "atr_pct": 0.02,
                        "bb_width": 0.04,
                        "score": 80.0,
                    },
                )
                model_signal = predict_model_signal("AAA", signal, rules)
                self.assertEqual(model_signal.status, "available")
                self.assertIsNotNone(model_signal.probability)
            finally:
                ml_models.MODEL_DIR = old_dir
                ml_models.MODEL_REGISTRY_PATH = old_registry


class DeepResearchBacktestTests(unittest.TestCase):
    def test_portfolio_backtest_emits_research_metrics_and_lot_sized_trades(self):
        rules = copy.deepcopy(load_rules())
        rules["min_history_rows"] = 60
        frames = {
            "FPT": make_demo_ohlcv("FPT", date(2026, 5, 27), rows=150),
            "HPG": make_demo_ohlcv("HPG", date(2026, 5, 27), rows=150),
        }
        result = run_portfolio_backtest(frames, rules, BacktestConfig.from_rules(rules), start=date(2026, 2, 1))
        self.assertGreaterEqual(result.total_trades, 0)
        self.assertIsNotNone(result.expectancy_pct)
        self.assertIsNotNone(result.sortino_ratio)
        for trade in result.trades:
            self.assertEqual(trade.quantity % rules["backtest"]["lot_size"], 0)

    def test_robustness_report_is_seeded_and_structured(self):
        rules = copy.deepcopy(load_rules())
        rules["min_history_rows"] = 60
        dates = trading_days_between(date(2026, 1, 1), date(2026, 5, 27))
        df = make_demo_ohlcv("FPT", date(2026, 5, 27), rows=len(dates))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            df.to_csv(root / "FPT.csv", index=False)
            first = run_robustness(["FPT"], rules, price_dir=root, seed=7, monte_carlo_runs=5)
            second = run_robustness(["FPT"], rules, price_dir=root, seed=7, monte_carlo_runs=5)
        self.assertEqual(first["monte_carlo"], second["monte_carlo"])
        self.assertIn("parameter_sensitivity", first)
        self.assertIn("stress_scenarios", first)


class DeepResearchAiGuardTests(unittest.TestCase):
    def test_ai_attempted_decision_change_and_fake_evidence_are_sanitized(self):
        class FakeClient:
            def chat_json(self, *_args, **_kwargs):
                return {
                    "symbol": "FPT",
                    "decision": "BUY_SETUP",
                    "summary": "ok",
                    "key_evidence": ["FPT:ema_trend", "FPT:fake"],
                    "risk_notes": "risk",
                    "confidence_note": "confidence",
                }

        candidate = {
            "symbol": "FPT",
            "decision": "WATCH",
            "score": 50,
            "confidence": 0.7,
            "latest_close": 100,
            "latest_date": "2026-05-27",
            "data_quality": {"status": "passed", "cross_check_status": "matched"},
            "risk_plan": None,
            "warnings": [],
            "evidence": [
                {"evidence_id": "FPT:ema_trend", "name": "EMA trend", "passed": True, "points": 1, "detail": "ok"}
            ],
        }
        analysis = _analyze_single_stock(candidate, FakeClient())
        self.assertEqual(analysis.decision, "WATCH")
        self.assertEqual(analysis.key_evidence, ["FPT:ema_trend"])
        self.assertTrue(any("attempted to change decision" in item for item in analysis.contradictions))


class TechnicalOnlyModeTests(unittest.TestCase):
    def test_scan_cli_has_no_news_flag_and_scan_has_no_news_warning(self):
        parser = build_parser()
        help_text = parser.format_help()
        self.assertNotIn("--news", help_text)

        result = run_scan(demo=True, symbols=["FPT"], persist=False)
        payload = [warning.lower() for candidate in result.candidates for warning in candidate.warnings]
        self.assertFalse(any("news" in warning or "rss" in warning for warning in payload))


if __name__ == "__main__":
    unittest.main()
