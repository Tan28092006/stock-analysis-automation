"""Immutable MR candidates: offline evaluation is not live approval.

The predeclared gates permit experimental paper use only. Provider-adjusted prices
and a current fixed universe do not establish unbiased historical performance.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import pickle
import subprocess
from uuid import uuid4

CANDIDATE_DIR = Path("data/models/candidates")
PRODUCTION_PATH = Path("data/models/win_prob_mr.pkl")
CANDIDATE_GATES = {"min_test_auc": .52, "max_test_ece": .10,
                   "max_brier_over_baseline": 0., "min_test_rows": 100,
                   "min_calibration_rows": 100, "decision_threshold": .65}
LIMITATIONS = ["point_in_time_universe_unverified", "price_adjustment_basis_unverified",
               "offline_metrics_not_execution_profit", "prospective_track_record_required"]


def evaluate_candidate(metrics: dict, *, snapshot_verified: bool) -> dict:
    failures = [] if snapshot_verified else ["snapshot_not_verified"]
    requirements = {"test_auc": lambda x: x >= CANDIDATE_GATES["min_test_auc"],
                    "test_ece": lambda x: x <= CANDIDATE_GATES["max_test_ece"],
                    "test_rows": lambda x: x >= CANDIDATE_GATES["min_test_rows"],
                    "calibration_rows": lambda x: x >= CANDIDATE_GATES["min_calibration_rows"]}
    for key, check in requirements.items():
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not check(value):
            failures.append(key)
    brier, baseline = metrics.get("test_brier"), metrics.get("baseline_brier")
    if (not isinstance(brier, (int, float)) or not isinstance(baseline, (int, float))
            or not math.isfinite(brier) or not math.isfinite(baseline)
            or brier > baseline + CANDIDATE_GATES["max_brier_over_baseline"]):
        failures.append("baseline_brier")
    return {"stage": "candidate", "paper_eligible": not failures,
            "shadow_eligible": bool(snapshot_verified), "live_approved": False,
            "criteria": dict(CANDIDATE_GATES), "failures": failures,
            "limitations": list(LIMITATIONS)}


def validate_candidate_target(artifact_path: Path | None) -> None:
    if artifact_path is not None and Path(artifact_path).resolve() == PRODUCTION_PATH.resolve():
        raise ValueError("Candidate training cannot replace the production artifact")
    if artifact_path is not None and Path(artifact_path).exists():
        raise FileExistsError(f"Immutable candidate already exists: {artifact_path}")


def verify_snapshot(manifest_path: Path, prices_dir: Path) -> dict:
    """Bind all consumed CSVs and provider responses to the immutable snapshot."""
    from ..data.reconciliation import verify_snapshot as verify_market_snapshot
    manifest_path = Path(manifest_path).resolve()
    verify_market_snapshot(manifest_path)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("kind") != "market_snapshot" or manifest.get("status") != "verified":
        raise ValueError("Training requires a verified market snapshot")
    files = manifest.get("files", {})
    actual = {p.stem for p in Path(prices_dir).glob("*.csv")}
    if not files or actual != set(files):
        raise ValueError("Snapshot file inventory differs from training prices")
    root = manifest_path.parent
    for symbol, info in files.items():
        for path_key, hash_key in (("path", "sha256"), ("raw_path", "raw_sha256")):
            path = (root / info[path_key]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Snapshot path invalid: {symbol}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != info[hash_key]:
                raise ValueError(f"Snapshot hash mismatch: {symbol}")
        if (root / info["path"]).resolve() != (Path(prices_dir) / f"{symbol}.csv").resolve():
            raise ValueError(f"Snapshot training path mismatch: {symbol}")
    return {"verified": True, "manifest_path": str(manifest_path),
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "as_of": manifest.get("as_of"), "source": manifest.get("source"),
            "adjustment_basis": manifest.get("adjustment_basis"),
            "universe_policy": manifest.get("universe_policy"),
            "files": {key: value["sha256"] for key, value in files.items()}}


def code_provenance() -> dict:
    root = Path(__file__).resolve().parents[2]
    # Use paths relative to this package so hashes do not depend on checkout location.
    package = Path(__file__).resolve().parents[1]
    source_paths = [package / "features" / name for name in
                    ("model_release.py", "win_probability.py", "indicators.py", "mr_exit.py", "temporal_validation.py")]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                 text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git_sha = None
    dependencies = {}
    for name in ("numpy", "pandas", "scikit-learn", "lightgbm"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = "unavailable"
    return {"git_sha": git_sha, "source_files": hashes,
            "source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "dependencies": dependencies}


def stage_candidate(artifact: dict, *, candidate_dir: Path = CANDIDATE_DIR,
                    artifact_path: Path | None = None, snapshot: dict | None = None) -> dict:
    validate_candidate_target(artifact_path)
    if artifact_path is None:
        run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:10]
        artifact_path = Path(candidate_dir) / run / "win_prob_mr.pkl"
    artifact_path = Path(artifact_path)
    report_path = artifact_path.with_suffix(".release.json")
    if report_path.exists():
        raise FileExistsError(f"Immutable candidate manifest already exists: {report_path}")
    payload = dict(artifact)
    payload["release"] = evaluate_candidate(payload.get("evaluation", {}),
                                             snapshot_verified=bool(snapshot and snapshot.get("verified")))
    payload["source_snapshot"] = snapshot or {"verified": False}
    payload["code_provenance"] = code_provenance()
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {key: payload.get(key) for key in ("trained_at", "training_metadata", "evaluation",
                "release", "source_snapshot", "code_provenance")}
    manifest["artifact_sha256"] = hashlib.sha256(raw).hexdigest()
    manifest_text = json.dumps(manifest, indent=2, allow_nan=False)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("xb") as handle:
        handle.write(raw)
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(manifest_text)
    return {"artifact": str(artifact_path), "artifact_sha256": manifest["artifact_sha256"],
            "release_manifest": str(report_path), "release": payload["release"]}


def load_candidate(artifact_path: Path, *, mode: str = "paper"):
    """Load only explicit locally trusted candidates; hashes are not a signature.

    This checks release eligibility, not historical availability. Consumers must
    separately call available_at for signal timestamps and log actual generation.
    Never load downloaded or otherwise untrusted pickle files.
    """
    from .win_probability import WinProbModel
    if mode not in {"paper", "shadow"}:
        raise ValueError("Candidate mode must be paper or shadow; live is not approved")
    artifact_path = Path(artifact_path)
    manifest = json.loads(artifact_path.with_suffix(".release.json").read_text(encoding="utf-8"))
    raw = artifact_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != manifest.get("artifact_sha256"):
        raise ValueError("Candidate artifact hash mismatch")
    eligible_key = f"{mode}_eligible"
    if not manifest.get("release", {}).get(eligible_key):
        raise ValueError(f"Candidate not {mode} eligible")
    snapshot = manifest.get("source_snapshot", {})
    if not snapshot.get("verified") or not snapshot.get("manifest_path"):
        raise ValueError("Candidate source snapshot not verified")
    source_manifest = Path(snapshot["manifest_path"])
    if verify_snapshot(source_manifest, source_manifest.parent / "prices") != snapshot:
        raise ValueError("Candidate source snapshot changed")
    recomputed = evaluate_candidate(manifest.get("evaluation", {}), snapshot_verified=True)
    if not recomputed[eligible_key]:
        raise ValueError(f"Candidate no longer {mode} eligible under current gates")
    payload = pickle.loads(raw)
    if payload.get("release") != manifest["release"] or payload.get("source_snapshot") != snapshot:
        raise ValueError("Candidate release metadata mismatch")
    model = WinProbModel(payload)
    model.meta["model_version"] = digest
    model.meta["candidate_mode"] = mode
    return model
