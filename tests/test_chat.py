import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd

from stock_agent.ai.chat import (
    extract_symbol_from_message,
    get_symbol_context,
    get_chat_client,
    handle_chat
)
from stock_agent.ai.ollama_client import OllamaChatClient
from stock_agent.ai.nvidia_client import NvidiaChatClient


class TestChatbot(unittest.TestCase):

    def test_extract_symbol_from_message(self):
        # Test exact match in universe
        self.assertEqual(extract_symbol_from_message("Phân tích mã FPT cho tôi"), "FPT")
        self.assertEqual(extract_symbol_from_message("Mua TCB giá nào được?"), "TCB")
        
        # Test case-insensitivity
        self.assertEqual(extract_symbol_from_message("Mã hpg thế nào?"), "HPG")
        
        # Test invalid symbol
        self.assertIsNone(extract_symbol_from_message("Mua con mã này thế nào?"))

    @patch("stock_agent.ai.chat.read_json")
    @patch("stock_agent.ai.chat.PRICE_CACHE_DIR")
    def test_get_symbol_context(self, mock_price_dir, mock_read_json):
        # Mock scan candidates
        mock_read_json.return_value = {
            "candidates": [
                {
                    "symbol": "FPT",
                    "decision": "BUY_SETUP",
                    "score": 65,
                    "latest_close": 99000,
                    "latest_date": "2026-05-27",
                    "risk_plan": {
                        "entry_reference": 99000,
                        "stop_loss": 96000
                    }
                }
            ]
        }
        
        # Mock CSV file presence and content
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_price_dir.__truediv__.return_value = mock_file
        
        df_mock = pd.DataFrame([
            {"date": "2026-05-27", "open": 98000, "high": 100000, "low": 97000, "close": 99000, "volume": 5000000}
        ])
        
        with patch("pandas.read_csv", return_value=df_mock):
            context = get_symbol_context("FPT")
            
            self.assertIsNotNone(context["candidate"])
            self.assertEqual(context["candidate"]["symbol"], "FPT")
            self.assertEqual(context["candidate"]["decision"], "BUY_SETUP")
            self.assertEqual(len(context["recent_prices"]), 1)
            self.assertEqual(context["recent_prices"][0]["close"], 99000)

    @patch("stock_agent.ai.chat.load_dotenv")
    @patch.dict("os.environ", {}, clear=True)
    def test_get_chat_client_default(self, mock_load):
        # Without NVIDIA_API_KEY, should return OllamaChatClient
        client = get_chat_client()
        self.assertIsInstance(client, OllamaChatClient)

    @patch("stock_agent.ai.chat.load_dotenv")
    @patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-testkey"}, clear=True)
    def test_get_chat_client_nvidia(self, mock_load):
        # With NVIDIA_API_KEY, should return NvidiaChatClient
        client = get_chat_client()
        self.assertIsInstance(client, NvidiaChatClient)

    @patch("stock_agent.ai.chat.get_chat_client")
    @patch("stock_agent.ai.chat.get_symbol_context")
    def test_handle_chat_compiles_context(self, mock_get_context, mock_get_client):
        # Mock context data
        mock_get_context.return_value = {
            "candidate": {
                "symbol": "FPT",
                "decision": "BUY_SETUP",
                "score": 60,
                "latest_close": 99000,
                "latest_date": "2026-05-27"
            },
            "recent_prices": [
                {"date": "2026-05-27", "open": 98000, "high": 100000, "low": 97000, "close": 99000, "volume": 5000000}
            ]
        }
        
        # Mock client chat
        mock_client = MagicMock()
        mock_client.chat_text.return_value = "Mocked AI Response about FPT"
        mock_client.is_available.return_value = True
        mock_get_client.return_value = mock_client
        
        response = handle_chat("Phân tích FPT", symbol="FPT")
        self.assertEqual(response, "Mocked AI Response about FPT")
        
        # Verify mock client chat_text was called with context containing FPT technical data
        args, kwargs = mock_client.chat_text.call_args
        system_prompt, user_payload = args
        self.assertIn("FPT", user_payload)
        self.assertIn("BUY_SETUP", user_payload)
        self.assertIn("99,000", user_payload) # formatted volume/close
        self.assertIn("THÔNG TIN BỐI CẢNH CỦA MÃ CHỨNG KHOÁN: FPT", user_payload)


if __name__ == "__main__":
    unittest.main()
