"""Retention inventory for EOD runtime folders — report only, nothing deleted.

Lists paper snapshot dirs (``data/paper/snapshots/<UTC>``) and flow run dirs
(``data/foreign_flows_v2/runs/<UTC>_<id>``) older than N days with file counts
and sizes. Snapshots referenced by a paper record are flagged: they are
scoring/audit evidence. Any deletion policy is a separate decision.

    python -m stock_agent.pipeline.retention_report --days 30
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re

PAPER_DIR = Path("data/paper")
FLOW_ROOT = Path("data/foreign_flows_v2")
_STAMP = re.compile(r"^(\d{8}T\d{6})(\d{6})?Z")


def _created(path: Path) -> datetime:
    match = _STAMP.match(path.name)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)


def _size(path: Path) -> tuple[int, int]:
    files = size = 0
    for item in path.rglob("*"):
        if item.is_file():
            files += 1
            size += item.stat().st_size
    return files, size


def _referenced_snapshots(paper_dir: Path) -> set[str]:
    found = set()
    for record in (paper_dir / "runs").glob("*/paper.json"):
        try:
            manifest = json.loads(record.read_text(encoding="utf-8")).get("snapshot_manifest")
        except (OSError, ValueError, AttributeError):
            continue
        if manifest:
            found.add(str(Path(manifest).resolve().parent))
    return found


def _inventory(folder: Path, cutoff: datetime, now: datetime, extra) -> dict:
    dirs = sorted(p for p in folder.glob("*") if p.is_dir()) if folder.is_dir() else []
    older = []
    for path in dirs:
        created = _created(path)
        if created >= cutoff:
            continue
        files, size = _size(path)
        older.append({"name": path.name, "path": str(path.resolve()), "created_at": created.isoformat(),
                      "age_days": round((now - created).total_seconds() / 86400, 1),
                      "files": files, "bytes": size, **extra(path)})
    total = sum(item["bytes"] for item in older)
    return {"folder": str(folder.resolve()), "total_count": len(dirs), "older_count": len(older),
            "older_bytes": total, "older_mb": round(total / 1e6, 2), "older": older}


def retention_report(*, paper_dir: Path = PAPER_DIR, flow_root: Path = FLOW_ROOT, days: int = 30,
                     now: datetime | None = None) -> dict:
    if days < 0:
        raise ValueError("days must be non-negative")
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    paper_dir, flow_root = Path(paper_dir), Path(flow_root)
    referenced = _referenced_snapshots(paper_dir)
    snapshots = _inventory(paper_dir / "snapshots", cutoff, now,
                           lambda p: {"referenced_by_paper_record": str(p.resolve()) in referenced})
    flow_runs = _inventory(flow_root / "runs", cutoff, now,
                           lambda p: {"has_manifest": (p / "manifest.json").exists()})
    return {"generated_at": now.isoformat(), "days": days, "cutoff": cutoff.isoformat(),
            "snapshots": snapshots, "flow_runs": flow_runs,
            "note": ("report only; nothing deleted. Snapshots referenced by paper records, research "
                     "manifests or audits are evidence; flow runs feed health()/load_flows() history.")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--paper-dir", type=Path, default=PAPER_DIR)
    parser.add_argument("--flow-root", type=Path, default=FLOW_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(retention_report(paper_dir=args.paper_dir, flow_root=args.flow_root, days=args.days),
                     ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
