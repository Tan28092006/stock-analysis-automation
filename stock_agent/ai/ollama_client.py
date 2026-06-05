from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import re
import os

from .env import load_dotenv


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:3b"
    timeout_seconds: int = 120
    max_retries: int = 3
    retry_base_delay: float = 2.0

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        load_dotenv()
        return cls(
            base_url=os.environ.get("OLLAMA_BASE_URL", cls.base_url).rstrip("/"),
            model=os.environ.get("OLLAMA_MODEL", cls.model).strip() or cls.model,
        )


class OllamaChatClient:
    """Local Ollama LLM client with retry/backoff."""

    def __init__(self, config: OllamaConfig | None = None):
        self.config = config or OllamaConfig.from_env()

    # ── public API ──────────────────────────────────────────────

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Send prompt, parse response as JSON."""
        content = self._chat(system, user, temperature)
        return _parse_json_content(content)

    def chat_text(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
    ) -> str:
        """Send prompt, return raw text."""
        return self._chat(system, user, temperature)

    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            req = Request(f"{self.config.base_url}/api/tags")
            with urlopen(req, timeout=5) as resp:  # noqa: S310
                return resp.status == 200
        except Exception:
            return False

    # ── internals ───────────────────────────────────────────────

    def _chat(self, system: str, user: str, temperature: float) -> str:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 2048,
            },
        }
        response = self._post_with_retry("/api/chat", payload)
        return response.get("message", {}).get("content", "")

    def _post_with_retry(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                request = Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=self.config.timeout_seconds) as resp:  # noqa: S310
                    return json.loads(resp.read().decode("utf-8"))
            except HTTPError as exc:
                last_exc = exc
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429 or exc.code >= 500:
                    delay = self.config.retry_base_delay * (2 ** attempt)
                    print(f"  ⚠ Ollama HTTP {exc.code}, retry in {delay:.0f}s...")
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Ollama API HTTP {exc.code}: {detail}") from exc
            except (URLError, OSError) as exc:
                last_exc = exc
                delay = self.config.retry_base_delay * (2 ** attempt)
                print(f"  ⚠ Ollama connection error, retry in {delay:.0f}s...")
                time.sleep(delay)
                continue

        raise RuntimeError(
            f"Ollama API failed after {self.config.max_retries} retries: {last_exc}"
        )


def _parse_json_content(content: str) -> dict[str, Any]:
    """Extract JSON from LLM response, handling markdown fences."""
    text = content.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return {"items": parsed}
        return parsed
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        # Try to find JSON array
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                result = json.loads(text[start : end + 1])
                if isinstance(result, list):
                    return {"items": result}
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse JSON from Ollama response:\n{content[:500]}")
