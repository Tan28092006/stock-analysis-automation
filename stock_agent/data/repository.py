from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

def clean_nan_inf(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, list):
        return [clean_nan_inf(item) for item in obj]
    if isinstance(obj, dict):
        return {key: clean_nan_inf(value) for key, value in obj.items()}
    return obj


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(clean_nan_inf(payload), f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(clean_nan_inf(payload), ensure_ascii=False) + "\n")

