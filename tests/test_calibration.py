import json
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from stock_agent.config import compute_rules_hash, load_rules
from stock_agent.data.exchange_calendar import next_trading_day, trading_days_between
from stock_agent.features.backtest import BacktestConfig
from stock_agent.features.calibration import (
    audit_indicator_correlation,
    build_labeled_t2_dataset,
    optimize_probability_threshold,
    run_calibration,
)


def _make_ohlcv(dates, base=100.0):
    close = np.array([base + idx * 0.2 + np.sin(idx / 5) for idx, _ in enumerate(dates)], dtype=float)
    open_ = close * (1 + np.array([np.sin(idx) * 0.002 for idx, _ in enumerate(dates)]))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = np.full(len(dates), 500_000, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _write_price_csv(root: Path, symbol: str, df: pd.DataFrame):
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(root / f"{symbol}.csv", index=False)


def _fixture_rules():
    rules = load_rules().copy()
    rules["min_history_rows"] = 90
    rules["time_horizon_days"] = 2
    return rules


class CalibrationLabelBuilderTests(unittest.TestCase):
    def test_label_builder_does_not_use_future_feature_values(self):
        rules = _fixture_rules()
        dates = trading_days_between(date(2025, 9, 1), date(2026, 2, 20))
        signal_date = dates[95]
        df = _make_ohlcv(dates)
        future_idx = df.index[df["date"] == dates[96]][0]
        df.loc[future_idx:, "close"] *= 1.10
        df["high"] = df[["high", "open", "close"]].max(axis=1) * 1.01
        df["low"] = df[["low", "open", "close"]].min(axis=1) * 0.99
        with tempfile.TemporaryDirectory() as tmp:
            price_dir = Path(tmp)
            _write_price_csv(price_dir, "AAA", df)
            dataset = build_labeled_t2_dataset(
                ["AAA"],
                signal_date,
                signal_date,
                rules,
                BacktestConfig(),
                price_dir=price_dir,
            )
        self.assertEqual(len(dataset), 1)
        expected_close = float(df.loc[df["date"] == signal_date, "close"].iloc[0])
        self.assertAlmostEqual(dataset.loc[0, "feature_close"], expected_close)

    def test_label_builder_skips_missing_next_open(self):
        rules = _fixture_rules()
        dates = trading_days_between(date(2025, 9, 1), date(2026, 2, 20))
        signal_date = dates[95]
        missing_entry = next_trading_day(signal_date)
        df = _make_ohlcv(dates)
        df = df[df["date"] != missing_entry].reset_index(drop=True)
        with tempfile.TemporaryDirectory() as tmp:
            price_dir = Path(tmp)
            _write_price_csv(price_dir, "AAA", df)
            dataset = build_labeled_t2_dataset(
                ["AAA"],
                signal_date,
                signal_date,
                rules,
                BacktestConfig(),
                price_dir=price_dir,
            )
        self.assertTrue(dataset.empty)

    def test_label_builder_t2_skips_weekend(self):
        rules = _fixture_rules()
        signal_date = date(2026, 5, 27)
        df = _make_ohlcv(trading_days_between(date(2026, 1, 1), date(2026, 6, 5)))
        with tempfile.TemporaryDirectory() as tmp:
            price_dir = Path(tmp)
            _write_price_csv(price_dir, "AAA", df)
            dataset = build_labeled_t2_dataset(
                ["AAA"],
                signal_date,
                signal_date,
                rules,
                BacktestConfig(),
                price_dir=price_dir,
            )
        self.assertEqual(str(dataset.loc[0, "entry_date"]), "2026-05-28")
        self.assertEqual(str(dataset.loc[0, "exit_date"]), "2026-06-01")

    def test_label_builder_excludes_possible_corporate_action(self):
        rules = _fixture_rules()
        dates = trading_days_between(date(2025, 9, 1), date(2026, 2, 20))
        signal_date = dates[95]
        df = _make_ohlcv(dates)
        jump_idx = 92
        df.loc[jump_idx:, ["open", "high", "low", "close"]] *= 1.25
        with tempfile.TemporaryDirectory() as tmp:
            price_dir = Path(tmp)
            _write_price_csv(price_dir, "AAA", df)
            dataset = build_labeled_t2_dataset(
                ["AAA"],
                signal_date,
                signal_date,
                rules,
                BacktestConfig(),
                price_dir=price_dir,
            )
        self.assertTrue(dataset.empty)


class CalibrationModelTests(unittest.TestCase):
    def _model_dataset(self, rows=120):
        dates = pd.bdate_range(start="2026-01-01", periods=rows).date
        records = []
        for idx, day in enumerate(dates):
            alpha = int(idx % 4 in {0, 1})
            beta = int(idx % 3 == 0)
            win = alpha
            records.append(
                {
                    "symbol": "AAA" if idx % 2 == 0 else "BBB",
                    "signal_date": day,
                    "decision": "BUY_SETUP" if alpha else "WATCH",
                    "rules_version": compute_rules_hash(load_rules()),
                    "net_t2_win": win,
                    "net_t2_return_pct": 0.8 if win else -0.4,
                    "rule_alpha": alpha,
                    "rule_beta": beta,
                    "feature_momentum": float(alpha) + idx * 0.001,
                    "feature_noise": float(beta),
                }
            )
        return pd.DataFrame(records)

    def test_run_calibration_is_deterministic(self):
        dataset = self._model_dataset()
        first = run_calibration(dataset, min_trades=5)
        second = run_calibration(dataset, min_trades=5)
        self.assertEqual(first.suggested_probability_threshold, second.suggested_probability_threshold)
        self.assertEqual(first.suggested_rule_weights, second.suggested_rule_weights)

    def test_threshold_optimizer_enforces_constraints(self):
        frame = pd.DataFrame(
            {
                "net_t2_win": [1] + [1] * 19,
                "net_t2_return_pct": [-1.0] + [1.0] * 19,
            }
        )
        probabilities = np.array([0.95] + [0.50] * 19)
        threshold, metrics, warnings = optimize_probability_threshold(frame, probabilities, min_trades=10)
        self.assertLessEqual(threshold, 0.50)
        self.assertGreaterEqual(metrics["avg_net_return_pct"], 0)
        self.assertEqual(warnings, [])

    def test_correlation_audit_flags_redundant_indicators(self):
        dataset = pd.DataFrame(
            {
                "rule_a": [0, 1, 0, 1, 0, 1],
                "feature_dup": [0, 1, 0, 1, 0, 1],
                "feature_other": [1, 1, 0, 0, 1, 0],
            }
        )
        flags = audit_indicator_correlation(dataset, threshold=0.8)
        pairs = {(item["feature_a"], item["feature_b"]) for item in flags}
        self.assertIn(("rule_a", "feature_dup"), pairs)


class CalibrationCliTests(unittest.TestCase):
    def test_calibrate_cli_smoke_outputs_json(self):
        cmd = [
            sys.executable,
            "-m",
            "stock_agent.cli",
            "calibrate",
            "--symbols",
            "ACB,FPT",
            "--start",
            "2026-01-01",
            "--end",
            "2026-05-25",
            "--min-trades",
            "5",
        ]
        result = subprocess.run(cmd, cwd="D:\\Chungkhoan", capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertIn(payload["status"], {"passed", "insufficient_data", "unstable"})
        self.assertEqual(payload["objective"], "win_rate")


if __name__ == "__main__":
    unittest.main()
