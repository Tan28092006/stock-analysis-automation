"""Desk configuration, hashes and filesystem layout."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

DEFAULT_CONFIG = Path("configs/swing_desk/desk_v1.json")
UNIVERSE = Path("configs/universe_vn30.json")


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=str).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_config(path: Path | None = None) -> dict:
    path = Path(path or DEFAULT_CONFIG)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["params_hash"] = sha256_bytes(canonical({k: v for k, v in cfg.items() if k != "params_hash"}))[:16]
    return cfg


def code_hash() -> str:
    """Hash of the desk package source: identifies the deterministic rule code in log rows."""
    here = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(here.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class DeskPaths:
    """Where the desk reads (repo configs, data root) and writes (desk dir).

    `data_root` holds `data/paper/snapshots` and `data/foreign_flows_v2`; it is read-only
    for the desk. `desk` is the git-ignored-runtime workspace `swing_desk/`.
    """

    repo: Path
    data_root: Path
    desk: Path

    @classmethod
    def default(cls, repo: Path | None = None, data_root: Path | None = None,
                desk: Path | None = None) -> "DeskPaths":
        repo = Path(repo or Path.cwd())
        return cls(repo=repo, data_root=Path(data_root or repo), desk=Path(desk or repo / "swing_desk"))

    @property
    def paper_snapshots(self) -> Path:
        return self.data_root / "data" / "paper" / "snapshots"

    @property
    def desk_snapshots(self) -> Path:
        # paper_runner --refresh --output-dir swing_desk/snapshots writes <output>/snapshots/<UTC>/
        return self.desk / "snapshots" / "snapshots"

    @property
    def flows_root(self) -> Path:
        return self.data_root / "data" / "foreign_flows_v2"

    @property
    def portfolio(self) -> Path:
        return self.desk / "portfolio.txt"

    @property
    def state(self) -> Path:
        return self.desk / "state"

    @property
    def checks_log(self) -> Path:
        return self.desk / "checks" / "checks.jsonl"

    @property
    def rec_log(self) -> Path:
        return self.desk / "recommendations" / "recommendations.jsonl"

    @property
    def scores(self) -> Path:
        return self.desk / "recommendations" / "scores"

    @property
    def events(self) -> Path:
        return self.desk / "events" / "events.json"

    @property
    def universe(self) -> Path:
        return self.repo / UNIVERSE

    @property
    def config(self) -> Path:
        return self.repo / DEFAULT_CONFIG


def load_universe_symbols(paths: DeskPaths) -> list[str]:
    data = json.loads(paths.universe.read_text(encoding="utf-8"))
    return [s.upper() for s in data["symbols"]]
