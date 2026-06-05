import unittest
from datetime import date, timedelta

from stock_agent.data.sample_data import make_demo_ohlcv
from stock_agent.data.validation import ProviderFrame, pick_cross_checked_frame, normalize_ohlcv
from stock_agent.data.validation import validate_ohlcv
from stock_agent.schemas import ProviderAudit
from stock_agent.config import load_rules


class ValidationTests(unittest.TestCase):
    def test_cross_check_passes_demo_sources(self):
        rules = load_rules()
        end = date.today()
        start = end - timedelta(days=260)
        df1 = normalize_ohlcv(make_demo_ohlcv("FPT", end, variant=0))
        df2 = normalize_ohlcv(make_demo_ohlcv("FPT", end, variant=1))
        df1 = df1[(df1["date"] >= start) & (df1["date"] <= end)]
        df2 = df2[(df2["date"] >= start) & (df2["date"] <= end)]
        frame, quality = pick_cross_checked_frame(
            [
                ProviderFrame("a", df1, ProviderAudit("a", "ok")),
                ProviderFrame("b", df2, ProviderAudit("b", "ok")),
            ],
            rules,
        )
        self.assertIsNotNone(frame)
        self.assertIn(quality.status, {"passed", "failed"})
        self.assertNotEqual(quality.cross_check_status, "single_source")

    def test_demo_variant_keeps_ohlcv_consistent(self):
        rules = load_rules()
        df = normalize_ohlcv(make_demo_ohlcv("ACB", date(2026, 5, 26), variant=1))
        warnings = validate_ohlcv(
            df,
            min_rows=rules["min_history_rows"],
            today=date(2026, 5, 26),
            max_age_days=rules["max_price_age_days"],
        )
        self.assertNotIn("high price inconsistency", warnings)
        self.assertNotIn("low price inconsistency", warnings)


if __name__ == "__main__":
    unittest.main()
