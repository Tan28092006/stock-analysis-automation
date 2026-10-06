"""R4: evidence bundle writer/validator. Validation never creates a pass.

Statuses (kept separate): FAIL / INCONCLUSIVE / BLOCKED / engineering_complete /
research_pass / prospective_ready. This validator can only return BLOCKED, FAIL or
``engineering_complete``; ``research_pass`` needs regression evidence and a human
reviewer recorded in decision.json, and is still not live approval.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

REQUIRED_FILES = ("experiment.json", "source_manifest.json", "environment.json", "dataset_manifest.json",
                  "splits.json", "metrics.json", "decision.json")
REQUIRED_DIRS = ("transforms", "predictions", "portfolio", "regression")
STATUSES = {"FAIL", "INCONCLUSIVE", "BLOCKED", "engineering_complete", "research_pass", "prospective_ready"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def init_bundle(root: Path, protocol: str, experiment_id: str, run_id: str) -> Path:
    """New directory per run; never overwrites an existing one."""
    run = Path(root) / protocol / experiment_id / run_id
    run.mkdir(parents=True, exist_ok=False)
    for d in REQUIRED_DIRS:
        (run / d).mkdir()
    return run


def write_json(path: Path, payload: dict) -> None:
    # LF on every OS: sealed bundles are hashed byte-for-byte.
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), newline="\n")


def seal(run: Path) -> dict:
    """Record hashes of every artifact file in experiment.json['artifact_hashes']."""
    exp = json.loads((run / "experiment.json").read_text())
    exp["artifact_hashes"] = {p.relative_to(run).as_posix(): sha256_file(p) for p in sorted(run.rglob("*"))
                              if p.is_file() and p.name not in ("experiment.json", "decision.json")}
    write_json(run / "experiment.json", exp)
    return exp["artifact_hashes"]


def _finite(obj, path="metrics") -> list[str]:
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            bad += _finite(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad += _finite(v, f"{path}[{i}]")
    elif isinstance(obj, float) and not math.isfinite(obj):
        bad.append(path)
    return bad


def validate_bundle(run: Path) -> dict:
    run = Path(run)
    problems, blocked = [], []
    for name in REQUIRED_FILES:
        if not (run / name).is_file():
            problems.append(f"missing:{name}")
    for name in REQUIRED_DIRS:
        if not (run / name).is_dir():
            problems.append(f"missing_dir:{name}")
    if problems:
        return {"status": "BLOCKED", "problems": problems}
    try:
        docs = {n: json.loads((run / n).read_text()) for n in REQUIRED_FILES}
    except json.JSONDecodeError as exc:
        return {"status": "FAIL", "problems": [f"invalid_json:{exc}"]}
    exp, metrics, decision = docs["experiment.json"], docs["metrics.json"], docs["decision.json"]
    # hash drift on any sealed artifact => FAIL
    sealed = exp.get("artifact_hashes")
    if not sealed:
        blocked.append("bundle_not_sealed")
    else:
        for rel, digest in sealed.items():
            p = run / rel
            if not p.is_file():
                problems.append(f"sealed_file_missing:{rel}")
            elif sha256_file(p) != digest:
                problems.append(f"hash_drift:{rel}")
        extra = {p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()} - set(sealed) - {
            "experiment.json", "decision.json"}
        if extra:
            problems.append(f"unsealed_files:{sorted(extra)[:5]}")
    if not any((run / "transforms").iterdir()) or not any((run / "predictions").iterdir()):
        blocked.append("no_transform_or_prediction_lineage")
    bad = _finite(metrics)
    if bad:
        problems.append(f"nonfinite_metrics:{bad[:5]}")
    floors = exp.get("sample_floors")
    support = metrics.get("sample_support", {})
    if not floors:
        blocked.append("sample_floors_not_frozen")
    else:
        for key, floor in floors.items():
            if support.get(key) is None:
                blocked.append(f"support_missing:{key}")
            elif support[key] < floor:
                blocked.append(f"below_floor:{key}")
    if not exp.get("activity_floor"):
        blocked.append("activity_floor_not_frozen")  # needs the user's agreed number
    splits = docs["splits.json"]
    for fold in splits.get("folds", []):
        if fold.get("problems"):
            blocked.append(f"fold_{fold.get('index')}:{','.join(fold['problems'])}")
    claimed = decision.get("status")
    if claimed not in STATUSES:
        problems.append("decision_status_invalid")
    if claimed in ("research_pass", "prospective_ready"):
        if not any((run / "regression").iterdir()):
            problems.append("research_pass_without_regression_evidence")
        if not decision.get("reviewer"):
            problems.append("research_pass_without_reviewer")
    if problems:
        status = "FAIL"
    elif blocked:
        status = "BLOCKED"
    else:
        status = "engineering_complete"
    return {"status": status, "problems": problems, "blocked": blocked}


if __name__ == "__main__":  # python -m stock_agent.research_ml.evidence <run-dir>
    result = validate_bundle(Path(sys.argv[1]))
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "engineering_complete" else 1)
