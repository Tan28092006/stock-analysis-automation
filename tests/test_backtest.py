import unittest
from datetime import date

from stock_agent.config import load_rules
from stock_agent.data.exchange_calendar import add_trading_days
from stock_agent.data.sample_data import make_demo_ohlcv
from stock_agent.features.backtest import (
    BacktestConfig,
    enforce_price_limit,
    round_trip_cost_pct,
    run_backtest,
)


class BacktestTests(unittest.TestCase):
    def setUp(self):
        import tests.test_backtest as test_mod
        self.original_load_rules = test_mod.load_rules
        def patched_load_rules():
            rules = self.original_load_rules()
            if "ml" in rules:
                rules["ml"]["enabled"] = False
                rules["ml"]["override_enabled"] = False
            return rules
        test_mod.load_rules = patched_load_rules

    def tearDown(self):
        import tests.test_backtest as test_mod
        test_mod.load_rules = self.original_load_rules

    def test_cost_calculation(self):
        self.assertAlmostEqual(round_trip_cost_pct(BacktestConfig()), 0.60)

    def test_price_limit_enforcement(self):
        self.assertAlmostEqual(enforce_price_limit(120.0, 100.0, 6.9), 106.9)
        self.assertAlmostEqual(enforce_price_limit(80.0, 100.0, 6.9), 93.1)
        self.assertAlmostEqual(enforce_price_limit(104.0, 100.0, 6.9), 104.0)

    def test_holiday_skipping(self):
        self.assertEqual(add_trading_days(date(2026, 4, 29), 1), date(2026, 5, 4))

    def test_thursday_entry_t2_is_monday(self):
        self.assertEqual(add_trading_days(date(2026, 5, 28), 2), date(2026, 6, 1))

    def test_backtest_smoke(self):
        df = make_demo_ohlcv("HPG", date(2026, 5, 26), rows=180)
        result = run_backtest("HPG", df, load_rules(), start=date(2026, 1, 1), end=date(2026, 5, 26))
        self.assertEqual(result.symbol, "HPG")
        self.assertGreaterEqual(result.total_trades, 0)
        self.assertIn("cost_model", result.summary)


if __name__ == "__main__":
    unittest.main()
