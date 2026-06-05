import unittest

from stock_agent.reports.evidence import build_evidence_bundle, validate_ai_summary


class EvidenceGuardTests(unittest.TestCase):
    def test_rejects_changed_decision_and_unknown_evidence(self):
        scan = {
            "scan_id": "scan-1",
            "candidates": [
                {
                    "symbol": "FPT",
                    "decision": "WATCH",
                    "score": 50,
                    "confidence": 0.7,
                    "latest_close": 1,
                    "latest_date": "2026-05-26",
                    "evidence": [
                        {
                            "evidence_id": "FPT:ema_trend",
                            "name": "EMA trend",
                            "passed": True,
                            "detail": "ok",
                        }
                    ],
                }
            ],
        }
        bundle = build_evidence_bundle(scan)
        errors = validate_ai_summary(
            {
                "scan_id": "scan-1",
                "candidates": [
                    {"symbol": "FPT", "decision": "BUY_SETUP", "evidence_ids": ["FPT:fake"]}
                ],
            },
            bundle,
        )
        self.assertEqual(len(errors), 2)


if __name__ == "__main__":
    unittest.main()

