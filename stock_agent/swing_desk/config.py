"""Desk configuration, hashes and filesystem layout."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

DEFAULT_CONFIG = Path("configs/swing_desk/desk_v1.json")
BF_CONFIG = Path("configs/swing_desk/bottomfish_v1.json")
BF_PROTOCOL = Path("configs/swing_desk/bottomfish_prospective_v1.json")
UNIVERSE = Path("configs/universe_vn30.json")          # the paper snapshot basket (VN30) and legacy universe
UNIVERSE_REGISTRY = Path("configs/swing_desk/universe.json")
LEGACY_UNIVERSE = "vn30"
_SYMBOL_RE = re.compile(r"[A-Z0-9]{2,12}")


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


def load_bf_config(path: Path | None = None) -> dict:
    """Bottom-fish thresholds (separate file so desk_v1 and its params_hash stay unchanged)."""
    path = Path(path or BF_CONFIG)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["bf_params_hash"] = sha256_bytes(canonical({k: v for k, v in cfg.items() if k != "bf_params_hash"}))[:16]
    return cfg


def code_hash() -> str:
    """Hash of the desk package source: identifies the deterministic rule code in log rows.

    Line endings are normalized so a CRLF (Windows autocrlf) and an LF checkout of the same
    commit give the same hash.
    """
    here = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(here.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class DeskPaths:
    """Where the desk reads (repo configs, data root) and writes (desk dir).

    `data_root` holds `data/paper/snapshots`, `data/foreign_flows_v2` and `data/history`; it is
    read-only for the desk (only the explicit `refresh --with-flows` / `refresh --history` triggers
    run the owning collectors). `desk` is the git-ignored-runtime workspace `swing_desk/`.
    `history` is "on" (long windows from the verified 10-year store when consistent) or "off"
    (snapshot only: the pre-history behaviour). `universe_name` selects the active universe from
    `configs/swing_desk/universe.json` (None = its default, VN100); a repo without that registry is the
    legacy VN30 desk.
    """

    repo: Path
    data_root: Path
    desk: Path
    history: str = "on"
    universe_name: str | None = None

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
    def history_root(self) -> Path:
        """Verified long price history stores (`stock_agent.data.history_store`), read-only here."""
        return self.data_root / "data" / "history"

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
    def bf_news(self) -> Path:
        return self.desk / "events" / "bf_news.json"

    @property
    def extra_snapshots(self) -> Path:
        """Isolated snapshots for held symbols outside the active universe (`refresh --held`)."""
        return self.desk / "snapshots" / "extra"

    @property
    def universe_snapshots(self) -> Path:
        """Desk-owned daily snapshots for active-universe symbols outside the paper VN30 basket."""
        return self.desk / "snapshots" / "universe"

    @property
    def bf_config(self) -> Path:
        return self.repo / BF_CONFIG

    @property
    def bf_protocol(self) -> Path:
        return self.repo / BF_PROTOCOL

    @property
    def universe(self) -> Path:
        """Symbols file of the active universe (configs/universe_vn30.json in the legacy desk)."""
        return universe_info(self, symbols=False)["path"]

    @property
    def primary_universe(self) -> Path:
        """The paper snapshot basket (VN30): the daily verified snapshot covers exactly these + VNINDEX."""
        return self.repo / UNIVERSE

    @property
    def config(self) -> Path:
        return self.repo / DEFAULT_CONFIG


def file_sha256(path: Path) -> str:
    """sha256 of the LF-normalized bytes (a CRLF and an LF checkout of one commit agree)."""
    return sha256_bytes(Path(path).read_bytes().replace(b"\r\n", b"\n"))


def _registry(repo: Path) -> dict | None:
    path = Path(repo) / UNIVERSE_REGISTRY
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _symbols(path: Path) -> tuple[dict, list[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    symbols = [str(s).strip().upper() for s in data.get("symbols") or []]
    if not symbols:
        raise ValueError(f"universe file {path} has an empty symbol list")
    bad = [s for s in symbols if not _SYMBOL_RE.fullmatch(s)]
    if bad:
        raise ValueError(f"universe file {path} has invalid symbols {bad}")
    dup = sorted({s for s in symbols if symbols.count(s) > 1})
    if dup:
        raise ValueError(f"universe file {path} has duplicate symbols {dup}")
    if "member_count" in data and int(data["member_count"]) != len(symbols):
        raise ValueError(f"universe file {path}: member_count {data['member_count']} != {len(symbols)} symbols")
    return data, symbols


def universe_info(paths: DeskPaths, *, symbols: bool = True) -> dict:
    """Active universe: name, label, symbols file, its LF sha256, symbols and optional sector map path.

    Name = `paths.universe_name` or the registry default. Without a registry only "vn30" exists
    (the legacy desk: configs/universe_vn30.json, no sector map).
    """
    reg = _registry(paths.repo)
    name = (paths.universe_name or (reg or {}).get("default") or LEGACY_UNIVERSE).lower()
    if reg is None:
        if name != LEGACY_UNIVERSE:
            raise ValueError(f"unknown universe {name!r}: no {UNIVERSE_REGISTRY.as_posix()} in {paths.repo} (only vn30)")
        entry = {"symbols_file": UNIVERSE.as_posix(), "sector_map": None}
    else:
        entry = (reg.get("universes") or {}).get(name)
        if entry is None:
            raise ValueError(f"unknown universe {name!r}; choose from {sorted(reg.get('universes') or {})}")
    path = paths.repo / entry["symbols_file"]
    out = {"name": name, "path": path,
           "sector_map": (paths.repo / entry["sector_map"]) if entry.get("sector_map") else None}
    if symbols:
        data, syms = _symbols(path)
        out.update(label=str(data.get("name") or name.upper()), symbols=syms, sha256=file_sha256(path),
                   effective_from=data.get("effective_from"))
    return out


def load_universe_symbols(paths: DeskPaths) -> list[str]:
    """Symbols of the active universe (G1 / BF-G1 membership, scans, state)."""
    return universe_info(paths)["symbols"]


def load_primary_symbols(paths: DeskPaths) -> list[str]:
    """The paper snapshot basket (VN30); the market regime/breadth and S2 stay defined on it."""
    return _symbols(paths.primary_universe)[1]


def universe_fields(paths: DeskPaths) -> dict:
    """Provenance stamped on checks.jsonl rows, frozen recommendation rows and the state manifest."""
    info = universe_info(paths)
    return {"universe": info["name"], "universe_source_sha": info["sha256"]}


def load_sector_map(paths: DeskPaths) -> dict | None:
    """Sector map of the active universe (None for VN30: legacy bank/vingroup caps only)."""
    path = universe_info(paths, symbols=False)["sector_map"]
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["source_file_sha256"] = file_sha256(path)
    return data


def concentration(paths: DeskPaths, cfg: dict) -> dict | None:
    """G9 concentration inputs for the active universe, or None (legacy desk_v1 groups only).

    groups: desk_v1 `sectors` (bank, vingroup) unioned with the sector map's `groups` (e.g. NAB, OCB as
    banks), with desk_v1's limits; plus a per-sector cap over the cited level-1 sector of each symbol.
    """
    smap = load_sector_map(paths)
    if smap is None:
        return None
    groups = {k: sorted(set(v)) for k, v in cfg["sectors"].items()}
    for k, v in (smap.get("groups") or {}).items():
        groups[k] = sorted(set(groups.get(k, [])) | {s.upper() for s in v})
    sector_of = {s.upper(): (e.get("sector") if e.get("status") == "verified" else None)
                 for s, e in (smap.get("symbols") or {}).items()}
    return {"groups": groups, "max_per_sector": int(smap["max_positions_per_sector"]), "sector_of": sector_of,
            "sector_map_sha256": smap["source_file_sha256"]}
