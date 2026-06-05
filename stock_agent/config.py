from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import CONFIG_DIR


@dataclass(frozen=True)
class AppConfig:
    universe_path: Path
    rules_path: Path

    @classmethod
    def default(cls) -> "AppConfig":
        return cls(
            universe_path=CONFIG_DIR / "universe_vn30.json",
            rules_path=CONFIG_DIR / "rules_t2.json",
        )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_universe(path: Path | None = None) -> dict[str, Any]:
    return load_json(path or AppConfig.default().universe_path)


def load_rules(path: Path | None = None) -> dict[str, Any]:
    return load_json(path or AppConfig.default().rules_path)


def compute_rules_hash(rules: dict[str, Any]) -> str:
    raw = json.dumps(rules, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
