from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .env import load_dotenv


@dataclass(frozen=True)
class NvidiaConfig:
    api_key: str
    base_url: str = "https://integrate.api.nvidia.com/v1"
    model: str = "google/gemma-2-2b-it"
    timeout_seconds: int = 60

    @classmethod
    def from_env(cls) -> "NvidiaConfig":
        load_dotenv()
        api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY is missing. Put it in .env or set it as an environment variable.")
        return cls(
            api_key=api_key,
            base_url=os.environ.get("NVIDIA_BASE_URL", cls.base_url).rstrip("/"),
            model=os.environ.get("NVIDIA_MODEL", cls.model).strip() or cls.model,
        )


class NvidiaChatClient:
    def __init__(self, config: NvidiaConfig | None = None):
        self.config = config or NvidiaConfig.from_env()

    def is_available(self) -> bool:
        """Check if NVIDIA API key is configured."""
        return bool(self.config.api_key)

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 1800,
    ) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "System instructions:\n"
                        f"{system}\n\n"
                        "User payload:\n"
                        f"{user}"
                    ),
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = self._post("/chat/completions", payload)
        content = response["choices"][0]["message"]["content"]
        return _parse_json_content(content)

    def chat_text(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 1800,
    ) -> str:
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "System instructions:\n"
                        f"{system}\n\n"
                        "User payload:\n"
                        f"{user}"
                    ),
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = self._post("/chat/completions", payload)
        return response["choices"][0]["message"]["content"]

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"NVIDIA API HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"NVIDIA API connection failed: {exc}") from exc


def _parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise
