"""Create derived clean/quarantine ledgers and strict scores; never rewrite sources.

Usage: python scripts/audit_prediction_ledgers.py --output-dir docs/audits/ledger
Empty clean output means insufficient evidence, not a 0% win rate.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timezone
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_agent.pipeline import forward_test as ft
from stock_agent.pipeline.ledger_integrity import audit_rows
from stock_agent.data.exchange_calendar import VN_TIMEZONE, next_trading_day


def read_rows(path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("not an object")
            out.append(value)
        except (ValueError, TypeError):
            out.append({"_parse_error": True, "_raw_line": line})
    return out


def git_reconstruction():
    """Secondary evidence only: never silently promote reconstructed records."""
    ledger = "data/pipeline/forward_test.jsonl"
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT)
    commits = git("log", "--reverse", "--format=%H", "--", ledger).decode().splitlines()
    first = {}
    for commit in commits:
        stamp = git("show", "-s", "--format=%cI", commit).decode().strip()
        blob = git("show", f"{commit}:{ledger}").decode("utf-8")
        artifact = subprocess.run(["git", "show", f"{commit}:data/models/win_prob_mr.pkl"],
                                  cwd=ROOT, capture_output=True)
        sha = hashlib.sha256(artifact.stdout).hexdigest() if artifact.returncode == 0 else None
        for line in blob.splitlines():
            if line.strip():
                key = json.dumps(json.loads(line), sort_keys=True)
                first.setdefault(key, (stamp, sha))
    counts, models = Counter(), Counter()
    rows = read_rows(ROOT / ledger)
    for row in rows:
        hit = first.get(json.dumps(row, sort_keys=True))
        if not hit:
            counts["no_exact_history_match"] += 1
            continue
        stamp, sha = hit
        sd = date.fromisoformat(row["signal_date"])
        begin = datetime.combine(sd, time(16), VN_TIMEZONE)
        entry = datetime.combine(next_trading_day(sd), time(9), VN_TIMEZONE)
        key = "first_commit_within_ex_ante_window" if begin <= datetime.fromisoformat(stamp) < entry else "first_commit_outside_ex_ante_window"
        counts[key] += 1
        models[sha or "missing"] += 1
    return {"ledger_changing_commits": len(commits), "exact_record_count": len(rows),
            "timing_counts": dict(counts), "tracked_artifact_hashes": dict(models),
            "caveat": "Local Git committer time is reconstructive evidence, not authenticated append time or proof of the actually loaded model."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # Resolve source paths relative to this repository, not the caller's directory.
    ft.PRICES_DIR = ROOT / "data/raw/prices_hist"
    ft.LEDGER_PATH = ROOT / "data/pipeline/forward_test.jsonl"
    paths = {"legacy": ROOT / "data/pipeline/pending_predictions.jsonl", "forward": ft.LEDGER_PATH}
    output = args.output_dir.resolve()
    if any(output == p.parent or p.parent in output.parents for p in paths.values()):
        raise ValueError("audit outputs must be outside the original ledger directory")
    output.mkdir(parents=True, exist_ok=True)
    frames = {}

    def frame_for(symbol):
        if symbol not in frames:
            frames[symbol] = ft._load_frame(symbol)
        return frames[symbol]

    summary = {"created_at": datetime.now(timezone.utc).isoformat(),
               "scope": "Derived provenance screening, not certification of model quality or live fills.",
               "legacy_score": None,
               "legacy_score_note": "No reconstruction of unknown original horizons or fabricated clipped fills."}
    for kind, path in paths.items():
        source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        audit = audit_rows(read_rows(path), frame_for, legacy=kind == "legacy")
        clean, quarantine = [], []
        for classified in audit.pop("records"):
            record = classified.pop("record")
            classified.update(source=str(path.relative_to(ROOT)), source_sha256=source_sha,
                              symbol=record.get("symbol"), signal_date=record.get("signal_date"),
                              engine=record.get("engine", kind), kind=record.get("kind"),
                              record_sha256=hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest())
            if classified["eligible"]:
                clean.append({**classified, "record": record})
            else:
                quarantine.append(classified)
        for suffix, records in (("clean", clean), ("quarantine", quarantine)):
            (output / f"{kind}-{suffix}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source_sha, "source changed during audit"
        summary[kind] = {**audit, "source_sha256": source_sha}
    summary["forward_score"] = ft.score()
    summary["git_reconstruction"] = git_reconstruction()
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
