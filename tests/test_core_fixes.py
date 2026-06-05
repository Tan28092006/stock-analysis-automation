import unittest
from datetime import date

import numpy as np
import pandas as pd

from stock_agent.agents.risk_guard import equal_score_weights, hrp_weights
from stock_agent.config import load_rules
from stock_agent.data.sample_data import make_demo_ohlcv
from stock_agent.features.signal_engine import score_symbol
from stock_agent.portfolio.pnl import calculate_pnl


class CoreFixTests(unittest.TestCase):
    def test_hrp_single_stock_capped(self):
        prices = pd.DataFrame({"ACB": np.linspace(100, 120, 40)})
        weights = hrp_weights(prices, max_weight=0.25)
        self.assertLessEqual(weights["ACB"], 0.25)

    def test_hrp_two_stocks_capped(self):
        prices = pd.DataFrame(
            {
                "ACB": np.linspace(100, 120, 40),
                "FPT": np.linspace(80, 92, 40),
            }
        )
        weights = hrp_weights(prices, max_weight=0.25)
        self.assertTrue(weights)
        self.assertTrue(all(weight <= 0.25 for weight in weights.values()))

    def test_equal_score_single_stock_capped(self):
        weights = equal_score_weights({"ACB": 80}, max_weight=0.25)
        self.assertEqual(weights["ACB"], 0.25)

    def test_tp1_not_circular(self):
        rules = load_rules()
        signal = score_symbol("HPG", make_demo_ohlcv("HPG", date(2026, 5, 26), rows=180), rules)
        risk = signal.risk_plan
        self.assertIsNotNone(risk)
        old_tp1 = risk.entry_reference + (risk.entry_reference - risk.stop_loss) * rules["signal_thresholds"]["min_rr"]
        self.assertNotAlmostEqual(risk.take_profit_1, old_tp1, places=3)

    def test_pnl_missing_price_no_distortion(self):
        positions = [
            {"symbol": "FPT", "buy_price": 100.0, "quantity": 10},
            {"symbol": "ZZZ", "buy_price": 100.0, "quantity": 10},
        ]
        result = calculate_pnl(positions, {"FPT": 112.0})
        self.assertEqual(result["total_unrealized_pnl"], 120.0)
        self.assertEqual(result["unpriced_positions"], 1)
        self.assertEqual(result["unpriced_cost"], 1000.0)
        self.assertEqual(result["pnl_status"], "partial")

    def test_pnl_all_missing_returns_zero(self):
        positions = [{"symbol": "ZZZ", "buy_price": 100.0, "quantity": 10}]
        result = calculate_pnl(positions, {})
        self.assertEqual(result["total_unrealized_pnl"], 0.0)
        self.assertEqual(result["total_unrealized_pnl_pct"], 0.0)
        self.assertEqual(result["pnl_status"], "no_prices")


if __name__ == "__main__":
    unittest.main()
