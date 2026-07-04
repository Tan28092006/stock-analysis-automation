import unittest
from unittest.mock import patch, MagicMock
import os
import json

from stock_agent.ai.nvidia_client import NvidiaChatClient, NvidiaConfig, _parse_json_content


class NvidiaClientTests(unittest.TestCase):
    def test_parse_fenced_json(self):
        payload = _parse_json_content('```json\n{"scan_id":"x"}\n```')
        self.assertEqual(payload["scan_id"], "x")

    @patch("stock_agent.ai.nvidia_client.load_dotenv")
    @patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-testkey"}, clear=True)
    def test_config_from_env(self, mock_load):
        config = NvidiaConfig.from_env()
        self.assertEqual(config.api_key, "nvapi-testkey")
        self.assertEqual(config.model, "google/gemma-2-2b-it")

    @patch("stock_agent.ai.nvidia_client.load_dotenv")
    @patch.dict("os.environ", {}, clear=True)
    def test_config_from_env_missing(self, mock_load):
        with self.assertRaises(RuntimeError):
            NvidiaConfig.from_env()

    @patch("stock_agent.ai.nvidia_client.load_dotenv")
    @patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-testkey"}, clear=True)
    def test_is_available(self, mock_load):
        client = NvidiaChatClient()
        self.assertTrue(client.is_available())

    @patch("stock_agent.ai.nvidia_client.urlopen")
    @patch("stock_agent.ai.nvidia_client.load_dotenv")
    @patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-testkey"}, clear=True)
    def test_chat_text(self, mock_load, mock_urlopen):
        # Mock response from urlopen
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{
                "message": {
                    "content": "Hello world"
                }
            }]
        }).encode("utf-8")
        
        # Enter context manager mock
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        client = NvidiaChatClient()
        response = client.chat_text("System", "User")
        self.assertEqual(response, "Hello world")

    @patch("stock_agent.ai.nvidia_client.urlopen")
    @patch("stock_agent.ai.nvidia_client.load_dotenv")
    @patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-testkey"}, clear=True)
    def test_chat_json(self, mock_load, mock_urlopen):
        # Mock response from urlopen returning fenced json
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{
                "message": {
                    "content": "```json\n{\"result\": \"ok\"}\n```"
                }
            }]
        }).encode("utf-8")
        
        # Enter context manager mock
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        client = NvidiaChatClient()
        response = client.chat_json("System", "User")
        self.assertEqual(response, {"result": "ok"})


if __name__ == "__main__":
    unittest.main()
