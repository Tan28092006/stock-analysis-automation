import unittest

from stock_agent.portfolio.pnl import calculate_pnl


class PortfolioTests(unittest.TestCase):
    def test_calculate_pnl(self):
        positions = [{"symbol": "FPT", "buy_price": 100.0, "quantity": 10}]
        result = calculate_pnl(positions, {"FPT": 112.0})
        self.assertEqual(result["total_cost"], 1000.0)
        self.assertEqual(result["total_market_value"], 1120.0)
        self.assertEqual(result["total_unrealized_pnl"], 120.0)
        self.assertEqual(result["rows"][0]["unrealized_pnl_pct"], 12.0)


if __name__ == "__main__":
    unittest.main()

