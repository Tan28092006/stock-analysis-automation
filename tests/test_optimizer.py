import copy
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from stock_agent.config import load_rules
from stock_agent.data.sample_data import make_demo_ohlcv
from stock_agent.features.backtest import BacktestConfig, run_portfolio_backtest
from stock_agent.features.indicators import add_indicators
from stock_agent.features.optimizer import (
    OptimizerConfig,
    _passes_risk_guard,
    allowed_search_paths,
    apply_best_rules,
    run_optimization,
    sample_rules,
    walk_forward_split,
)


class FakeTrial:
    number = 0

    def suggest_int(self, _name, low, high):
        return int((low + high) // 2)

    def suggest_float(self, _name, low, high):
        return float((low + high) / 2)


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        import tests.test_optimizer as test_mod
        self.original_load_rules = test_mod.load_rules
        def patched_load_rules():
            rules = self.original_load_rules()
            if "ml" in rules:
                rules["ml"]["enabled"] = False
                rules["ml"]["override_enabled"] = False
            return rules
        test_mod.load_rules = patched_load_rules

    def tearDown(self):
        import tests.test_optimizer as test_mod
        test_mod.load_rules = self.original_load_rules

    def test_search_space_sampling_writes_only_allowed_paths_and_no_news(self):
        rules = copy.deepcopy(load_rules())
        rules["news"] = {"rss_sources": ["bad"]}
        tuned = sample_rules(FakeTrial(), rules, include_ml=True)
        self.assertNotIn("news", tuned)
        params = {path for path in allowed_search_paths(include_ml=True)}
        self.assertIn("signal_thresholds.buy_setup", params)
        self.assertIn("ml.probability_threshold", params)

    def test_walk_forward_split_preserves_chronological_order(self):
        frames = {
            "FPT": make_demo_ohlcv("FPT", date(2026, 5, 27), rows=120),
            "HPG": make_demo_ohlcv("HPG", date(2026, 5, 27), rows=120),
        }
        split = walk_forward_split(frames)
        self.assertLess(split.train_start, split.train_end)
        self.assertLess(split.train_end, split.validation_start)
        self.assertLess(split.validation_end, split.test_start)
        self.assertLess(split.test_start, split.test_end)

    def test_optimizer_rejects_too_few_trades_and_high_drawdown(self):
        cfg = OptimizerConfig(min_final_trades=30, max_drawdown_pct=20)
        self.assertFalse(
            _passes_risk_guard(
                {"total_trades": 2, "max_drawdown_pct": 5, "expectancy_pct": 1},
                cfg,
            )
        )
        self.assertFalse(
            _passes_risk_guard(
                {"total_trades": 50, "max_drawdown_pct": 30, "expectancy_pct": 1},
                cfg,
            )
        )

    def test_apply_creates_backup_before_modifying_rules(self):
        rules = copy.deepcopy(load_rules())
        rules.pop("news", None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rules_t2.json"
            path.write_text(json.dumps(rules), encoding="utf-8")
            report = {
                "config": {"min_final_trades": 10, "max_drawdown_pct": 20},
                "baseline": {"final_test": {"total_return_pct": -1, "expectancy_pct": -0.1}},
                "selected_candidate": {
                    "params": {"signal_thresholds.buy_setup": 55},
                    "final_test": {
                        "total_return_pct": 2,
                        "expectancy_pct": 0.2,
                        "max_drawdown_pct": 5,
                        "total_trades": 10,
                    },
                },
            }
            result = apply_best_rules(report, rules_path=path)
            self.assertEqual(result["status"], "applied")
            self.assertTrue(Path(result["backup_path"]).exists())
            updated = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(updated["signal_thresholds"]["buy_setup"], 55)
            self.assertNotIn("news", updated)

    def test_fast_backtest_matches_precomputed_fixture(self):
        rules = copy.deepcopy(load_rules())
        rules["min_history_rows"] = 60
        frames = {
            "FPT": make_demo_ohlcv("FPT", date(2026, 5, 27), rows=150),
            "HPG": make_demo_ohlcv("HPG", date(2026, 5, 27), rows=150),
        }
        precomputed = {symbol: add_indicators(frame, atr_period=rules["risk"]["atr_period"]) for symbol, frame in frames.items()}
        raw_result = run_portfolio_backtest(frames, rules, BacktestConfig.from_rules(rules))
        fast_result = run_portfolio_backtest(precomputed, rules, BacktestConfig.from_rules(rules))
        self.assertEqual(raw_result.total_trades, fast_result.total_trades)
        self.assertAlmostEqual(raw_result.total_return_pct, fast_result.total_return_pct, places=4)

    def test_optimize_smoke_without_mutating_config(self):
        rules = copy.deepcopy(load_rules())
        rules["min_history_rows"] = 60
        if "ml" in rules:
            rules["ml"]["enabled"] = False
            rules["ml"]["override_enabled"] = False
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for symbol in ("FPT", "HPG"):
                make_demo_ohlcv(symbol, date(2026, 5, 27), rows=150).to_csv(root / f"{symbol}.csv", index=False)
            report = run_optimization(
                ["FPT", "HPG"],
                rules=rules,
                config=OptimizerConfig(years=0, n_trials=3, min_final_trades=5),
                price_dir=root,
                apply_if_valid=False,
            )
        self.assertIn(report["status"], {"target_reached", "target_not_reached", "no_valid_candidate"})
        self.assertEqual(report["trials_completed"], 3)
        self.assertIsNone(report["apply_result"])


if __name__ == "__main__":
    unittest.main()
