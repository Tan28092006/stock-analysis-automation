import os
import tempfile
import unittest
from pathlib import Path

from stock_agent.ai.env import load_dotenv


class EnvTests(unittest.TestCase):
    def test_load_dotenv_does_not_override_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("NVIDIA_API_KEY=from_file\nNVIDIA_MODEL=test-model\n", encoding="utf-8")
            os.environ["NVIDIA_API_KEY"] = "existing"
            old_model = os.environ.pop("NVIDIA_MODEL", None)
            try:
                load_dotenv(path)
                self.assertEqual(os.environ["NVIDIA_API_KEY"], "existing")
                self.assertEqual(os.environ["NVIDIA_MODEL"], "test-model")
            finally:
                if old_model is not None:
                    os.environ["NVIDIA_MODEL"] = old_model
                os.environ.pop("NVIDIA_API_KEY", None)


if __name__ == "__main__":
    unittest.main()

