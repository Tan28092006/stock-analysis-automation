import unittest

from stock_agent.ai.nvidia_client import _parse_json_content


class NvidiaClientTests(unittest.TestCase):
    def test_parse_fenced_json(self):
        payload = _parse_json_content('```json\n{"scan_id":"x"}\n```')
        self.assertEqual(payload["scan_id"], "x")


if __name__ == "__main__":
    unittest.main()

