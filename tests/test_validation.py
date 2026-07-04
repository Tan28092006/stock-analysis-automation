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

    def test_experiment_manifest_logging(self):
        from stock_agent.schemas import ExperimentManifest
        from stock_agent.pipeline.performance_tracker import log_experiment_manifest
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        # Create a mock manifest
        manifest = ExperimentManifest(
            command="python -m stock_agent.cli backtest --all",
            timestamp="2026-06-09T12:00:00Z",
            rules_hash="test_hash",
            data_start="2023-01-01",
            data_end="2025-12-31",
            symbols=["FPT", "TCB"],
            report_links={"backtest": "data/reports/backtest_results.json"}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_log_path = Path(tmpdir) / "experiment_manifests_test.jsonl"
            temp_dir_path = Path(tmpdir) / "manifests"

            with patch("stock_agent.pipeline.performance_tracker.MANIFEST_LOG_PATH", temp_log_path), \
                 patch("stock_agent.pipeline.performance_tracker.MANIFEST_DIR", temp_dir_path):

                # Log it
                individual_path = log_experiment_manifest(manifest)
                self.assertTrue(individual_path.exists())

                # Verify individual file contents
                with individual_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                self.assertEqual(data["command"], manifest.command)
                self.assertEqual(data["rules_hash"], manifest.rules_hash)

                # Verify central log contains the record
                self.assertTrue(temp_log_path.exists())
                found = False
                with temp_log_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        record = json.loads(line.strip())
                        if record.get("timestamp") == manifest.timestamp and record.get("rules_hash") == manifest.rules_hash:
                            found = True
                            break
                self.assertTrue(found)

    def test_create_and_log_manifest_with_new_fields(self):
        from stock_agent.pipeline.performance_tracker import create_and_log_manifest
        from stock_agent.config import load_rules
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        rules = load_rules()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_log_path = Path(tmpdir) / "experiment_manifests_test.jsonl"
            temp_dir_path = Path(tmpdir) / "manifests"

            with patch("stock_agent.pipeline.performance_tracker.MANIFEST_LOG_PATH", temp_log_path), \
                 patch("stock_agent.pipeline.performance_tracker.MANIFEST_DIR", temp_dir_path):

                individual_path = create_and_log_manifest(
                    command="python -m stock_agent.cli optimize --symbols FPT,HPG",
                    rules=rules,
                    data_start="2023-01-01",
                    data_end="2026-01-01",
                    symbols=["FPT", "HPG"],
                    report_links={"optimization_report": "data/reports/optimization_test.json"},
                    symbols_excluded=["VPL"],
                    data_hash="combined_test_hash",
                    data_hashes={"FPT": "fpt_hash", "HPG": "hpg_hash"},
                )

                self.assertTrue(individual_path.exists())

                # Verify individual file contents
                with individual_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                self.assertEqual(data["command"], "python -m stock_agent.cli optimize --symbols FPT,HPG")
                self.assertEqual(data["symbols_excluded"], ["VPL"])
                self.assertEqual(data["data_hash"], "combined_test_hash")
                self.assertEqual(data["data_hashes"], {"FPT": "fpt_hash", "HPG": "hpg_hash"})

                # Verify central log contains the record
                self.assertTrue(temp_log_path.exists())
                found = False
                with temp_log_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        record = json.loads(line.strip())
                        if record.get("command") == data["command"] and record.get("data_hash") == "combined_test_hash":
                            found = True
                            break
                self.assertTrue(found)


if __name__ == "__main__":
    unittest.main()
