import unittest
from stock_agent.reports.contradiction_detector import detect_contradictions
from stock_agent.reports.evidence import (
    _extract_numbers,
    _build_allowed_numbers,
    validate_reason_content,
)


class ContradictionDetectorTests(unittest.TestCase):

    def test_high_score_low_rr(self):
        candidate = {
            "symbol": "ACB", "decision": "BUY_SETUP", "score": 90,
            "risk_plan": {"reward_risk": 1.2, "stop_loss_pct": 3.0,
                          "entry_reference": 24800, "take_profit_1": 25200},
            "data_quality": {"status": "passed", "cross_check_status": "matched"},
            "evidence": [], "warnings": [],
        }
        issues = detect_contradictions(candidate)
        self.assertTrue(any("R:R thấp" in i for i in issues))

    def test_buy_with_degraded_data(self):
        candidate = {
            "symbol": "FPT", "decision": "BUY_SETUP", "score": 70,
            "risk_plan": {"reward_risk": 2.0, "stop_loss_pct": 3.0,
                          "entry_reference": 75000, "take_profit_1": 80000},
            "data_quality": {"status": "degraded", "cross_check_status": "single_source"},
            "evidence": [], "warnings": [],
        }
        issues = detect_contradictions(candidate)
        self.assertTrue(any("degraded" in i for i in issues))
        self.assertTrue(any("single source" in i for i in issues))

    def test_wide_stop_loss(self):
        candidate = {
            "symbol": "VHM", "decision": "WATCH", "score": 58,
            "risk_plan": {"reward_risk": 1.8, "stop_loss_pct": 5.5,
                          "entry_reference": 158700, "take_profit_1": 170000},
            "data_quality": {"status": "passed", "cross_check_status": "matched"},
            "evidence": [], "warnings": [],
        }
        issues = detect_contradictions(candidate)
        self.assertTrue(any("quá rộng" in i for i in issues))

    def test_tp1_too_close(self):
        candidate = {
            "symbol": "TCB", "decision": "BUY_SETUP", "score": 65,
            "risk_plan": {"reward_risk": 1.5, "stop_loss_pct": 2.0,
                          "entry_reference": 32750, "take_profit_1": 32900},
            "data_quality": {"status": "passed", "cross_check_status": "matched"},
            "evidence": [], "warnings": [],
        }
        issues = detect_contradictions(candidate)
        self.assertTrue(any("quá gần" in i for i in issues))

    def test_no_contradictions(self):
        candidate = {
            "symbol": "SSB", "decision": "BUY_SETUP", "score": 75,
            "risk_plan": {"reward_risk": 2.0, "stop_loss_pct": 2.5,
                          "entry_reference": 17450, "take_profit_1": 18500},
            "data_quality": {"status": "passed", "cross_check_status": "matched"},
            "evidence": [{"passed": True}, {"passed": True}, {"passed": False}],
            "warnings": [],
        }
        issues = detect_contradictions(candidate)
        self.assertEqual(len(issues), 0)

    def test_reject_has_no_contradictions(self):
        candidate = {
            "symbol": "MWG", "decision": "REJECT", "score": 20,
            "risk_plan": {"reward_risk": 0.5, "stop_loss_pct": 6.0,
                          "entry_reference": 78500, "take_profit_1": 79000},
            "data_quality": {"status": "passed", "cross_check_status": "matched"},
            "evidence": [], "warnings": [],
        }
        # REJECT stocks shouldn't trigger BUY-specific contradictions
        issues = detect_contradictions(candidate)
        # wide SL only triggers for BUY/WATCH
        self.assertFalse(any("quá rộng" in i for i in issues))


class FabricatedNumberTests(unittest.TestCase):

    def test_extract_numbers(self):
        nums = _extract_numbers("Giá 24800 VND, RSI 55.3, volume tăng 1.25x")
        self.assertIn(24800.0, nums)
        self.assertIn(55.3, nums)
        self.assertIn(1.25, nums)

    def test_build_allowed_numbers(self):
        candidate = {
            "score": 98, "confidence": 0.92, "latest_close": 24800,
            "risk_plan": {"entry_reference": 24800, "stop_loss": 24100,
                          "take_profit_1": 25500, "reward_risk": 1.8},
            "evidence": [{"value": 55.3}, {"value": {"rr": 1.8, "sl": 2.5}}],
        }
        allowed = _build_allowed_numbers(candidate)
        self.assertIn(24800.0, allowed)
        self.assertIn(98.0, allowed)
        self.assertIn(55.3, allowed)
        self.assertIn(1.8, allowed)

    def test_valid_reason_passes(self):
        analysis = {
            "symbol": "ACB",
            "summary": "Giá 24800, RSI 55.3, score 98 điểm.",
            "risk_notes": "SL tại 24100, TP1 tại 25500.",
            "confidence_note": "Confidence 0.92",
        }
        candidate = {
            "score": 98, "confidence": 0.92, "latest_close": 24800,
            "risk_plan": {"entry_reference": 24800, "stop_loss": 24100,
                          "take_profit_1": 25500, "reward_risk": 1.8},
            "evidence": [{"value": 55.3}],
        }
        errors = validate_reason_content(analysis, candidate)
        self.assertEqual(len(errors), 0)

    def test_fabricated_number_flagged(self):
        analysis = {
            "symbol": "ACB",
            "summary": "Giá đã tăng lên 30000 VND theo dự đoán.",
            "risk_notes": "",
            "confidence_note": "",
        }
        candidate = {
            "score": 98, "latest_close": 24800,
            "risk_plan": {"entry_reference": 24800},
            "evidence": [],
        }
        errors = validate_reason_content(analysis, candidate)
        self.assertTrue(any("30000" in e for e in errors))


class OllamaClientConfigTests(unittest.TestCase):

    def test_default_config(self):
        from stock_agent.ai.ollama_client import OllamaConfig
        config = OllamaConfig()
        self.assertEqual(config.base_url, "http://localhost:11434")
        self.assertEqual(config.model, "qwen2.5:3b")
        self.assertEqual(config.max_retries, 3)


if __name__ == "__main__":
    unittest.main()
