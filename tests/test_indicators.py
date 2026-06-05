import unittest
from datetime import date

from stock_agent.data.sample_data import make_demo_ohlcv
from stock_agent.features.indicators import add_indicators
from stock_agent.features.signal_engine import score_symbol
from stock_agent.config import load_rules


class IndicatorTests(unittest.TestCase):
    def test_indicators_have_latest_values(self):
        df = make_demo_ohlcv("FPT", date.today(), rows=160)
        out = add_indicators(df)
        latest = out.iloc[-1]
        self.assertGreater(latest["ema21"], 0)
        self.assertGreaterEqual(latest["rsi14"], 0)
        self.assertLessEqual(latest["rsi14"], 100)

    def test_signal_output_is_bounded(self):
        df = make_demo_ohlcv("HPG", date.today(), rows=180)
        signal = score_symbol("HPG", df, load_rules())
        self.assertIn(signal.decision, {"BUY_SETUP", "WATCH", "REJECT"})
        self.assertGreaterEqual(signal.score, -20)
        self.assertIsNotNone(signal.risk_plan)


if __name__ == "__main__":
    unittest.main()

