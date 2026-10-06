"""Prospective arms: rule-only baseline, random control, analyst arms (desk / single), scoring.

Arms (configs/swing_desk/prospective_v1.json):
  baseline  S1 or S2 tickers with structural gate PASS, lowest RSI(2) first, top 3.
  control   3 random tickers whose structural gate (ignoring the G11 setup rule) is PASS,
            seeded by the target session date.
  single    one analyst rollout (a1, fixed in advance) before VeriHarness verification.
  desk      the repaired, adjudicated VeriHarness deliverable.
Structural gate = G0-G8 + G11 (G9 portfolio and G10 events are identical for every arm).
All arms share one entry/stop/target plan: limit = close x 1.005 on the next session,
stop = close x 0.97, target = SMA20 (with setup) or 20-session high, 8-session time stop.
"""
from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import random

import pandas as pd

from . import data as dl
from . import reclog
from .config import DeskPaths, code_hash, load_config
from .scorer import score_row, summarize
from .service import load_state, state_freshness

PROMPTS = ("ANALYST.md", "OUTPUT_CONTRACT.md", "SENTIMENT_RUBRIC.md")


def prompt_hash(paths: DeskPaths) -> str | None:
    digest = hashlib.sha256()
    found = False
    for name in PROMPTS:
        path = paths.desk / "prompts" / name
        if path.is_file():
            found = True
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16] if found else None


def _plan_row(t: dict, state: dict, cfg: dict, gate_key: str) -> dict | None:
    ind = t["indicators"]
    close = ind["close"]
    limit = round(close * 1.005, 2)
    stop = round(close * (1 - cfg["stops"]["user_stop_pct"] / 100), 2)
    target = t.get("target")
    if not target or target <= limit:
        return None
    m = state["manifest"]
    plan = t["plan_default_nav"]
    return {"ticker": t["ticker"], "action": "BUY", "setup": t["setups"], "gate": t[gate_key],
            "gate_failed": [r["id"] for r in t["rules"] if r["status"] == "FAIL"],
            "entry": {"type": "limit", "limit_price": limit, "valid_session": m["trade_session"]},
            "entry_ref": close, "stop": stop, "stop_basis": "user_3pct", "target": round(target, 2),
            "max_hold_sessions": cfg["setups"]["generic_max_hold_sessions"], "earliest_sell": t["earliest_sell"],
            "size": {"qty": plan["qty"], "risk_vnd_effective": plan["vnd_at_risk_effective"],
                     "risk_pct_nav": cfg["money"]["risk_per_trade_pct"],
                     "nav_source": plan.get("nav_source")},
            "rsi2": ind.get("rsi2"), "data_as_of": m["as_of"], "snapshot_sha256": m["snapshot_manifest_sha256"]}


def candidate_rows(state: dict, source: str, cfg: dict | None = None) -> list[dict]:
    cfg = cfg or load_config()
    n_base, n_ctrl = cfg["evaluation"]["baseline_top_n"], cfg["evaluation"]["control_n"]
    tickers = state["tickers"].values()
    if source == "baseline":
        pool = [t for t in tickers if t["setups"] and t["structural_gate"] == "PASS"]
        pool.sort(key=lambda t: (t["indicators"].get("rsi2") if t["indicators"].get("rsi2") is not None else 999, t["ticker"]))
        rows = [_plan_row(t, state, cfg, "structural_gate") for t in pool]
        return [r for r in rows if r][:n_base]
    if source == "control":
        pool = sorted((t for t in tickers if t["structural_gate_ex_setup"] == "PASS"), key=lambda t: t["ticker"])
        rows = [r for r in (_plan_row(t, state, cfg, "structural_gate_ex_setup") for t in pool) if r]
        seed = int(state["manifest"]["trade_session"].replace("-", ""))
        return sorted(random.Random(seed).sample(rows, min(n_ctrl, len(rows))), key=lambda r: r["ticker"])
    raise ValueError("code-generated arms are baseline and control")


def _fresh_state(paths: DeskPaths, now: datetime) -> dict:
    fresh = state_freshness(paths, now)
    if not fresh["ok"]:
        raise ValueError(f"desk state not fresh: {fresh['reason']}")
    return load_state(paths)


def _meta(paths: DeskPaths, state: dict, cfg: dict, *, model_id: str, n_rollouts: int,
          adjudication: dict | None) -> dict:
    return {"snapshot_manifest_sha256": state["manifest"]["snapshot_manifest_sha256"], "desk_code_hash": code_hash(),
            "params_hash": cfg["params_hash"], "prompt_hash": prompt_hash(paths), "model_id": model_id,
            "n_rollouts": n_rollouts, "adjudication": adjudication, "protocol": "swing_desk_prospective_v1"}


def _already(paths: DeskPaths, source: str, session: str) -> bool:
    return any(r.get("source") == source and r.get("target_session") == session for r in reclog.read_rows(paths.rec_log))


def freeze_code_arms(paths: DeskPaths, *, now: datetime) -> dict:
    cfg = load_config(paths.config)
    state = _fresh_state(paths, now)
    session = state["manifest"]["trade_session"]
    out = {}
    for source in ("baseline", "control"):
        if _already(paths, source, session):
            raise ValueError(f"{source} already frozen for {session}")
    for source in ("baseline", "control"):
        rows = candidate_rows(state, source, cfg)
        out[source] = reclog.freeze(paths.rec_log, rows, source=source, target_session=date.fromisoformat(session),
                                    session_as_of=state["manifest"]["as_of"], now=now,
                                    meta=_meta(paths, state, cfg, model_id="rules", n_rollouts=0, adjudication=None))
    return out


def freeze_analyst(paths: DeskPaths, rows: list[dict], *, source: str, now: datetime, model_id: str,
                   n_rollouts: int, adjudication: dict | None = None) -> list[dict]:
    if source not in reclog.ANALYST_SOURCES:
        raise ValueError("analyst arms are desk and single")
    cfg = load_config(paths.config)
    state = _fresh_state(paths, now)
    session = state["manifest"]["trade_session"]
    if _already(paths, source, session):
        raise ValueError(f"{source} already frozen for {session}")
    for row in rows:
        t = state["tickers"].get(str(row.get("ticker", "")).upper())
        if row.get("action") == "BUY" and (t is None or t["structural_gate"] == "BLOCK"):
            raise ValueError(f"{row.get('ticker')}: not in VN30 state or structural gate BLOCK")
        if row.get("snapshot_sha256") not in (None, state["manifest"]["snapshot_manifest_sha256"]):
            raise ValueError(f"{row.get('ticker')}: snapshot_sha256 does not match the desk state")
    return reclog.freeze(paths.rec_log, rows, source=source, target_session=date.fromisoformat(session),
                         session_as_of=state["manifest"]["as_of"], now=now,
                         meta=_meta(paths, state, cfg, model_id=model_id, n_rollouts=n_rollouts,
                                    adjudication=adjudication))


def score_log(paths: DeskPaths, *, now: datetime) -> dict:
    cfg = load_config(paths.config)
    chain = reclog.verify_chain(paths.rec_log)
    if not chain["ok"]:
        raise ValueError(f"recommendation log chain broken: {chain['errors']}")
    manifest = dl.newest_verified_manifest(paths)
    if manifest is None:
        raise ValueError("no verified snapshot to score against")
    frames = dl.load_frames(manifest)
    rows = [score_row(r, frames.get(r.get("ticker"))) for r in reclog.read_rows(paths.rec_log)]
    checks = reclog.read_rows(paths.checks_log)
    by_verdict = {}
    for c in checks:
        by_verdict[c.get("verdict")] = by_verdict.get(c.get("verdict"), 0) + 1
    as_of = json.loads(Path(manifest).read_text(encoding="utf-8"))["as_of"]
    result = {"as_of": as_of, "scored_at": now.isoformat(), "snapshot": str(manifest),
              "snapshot_sha256": dl.manifest_sha256(manifest), "chain": chain,
              "checks_chain": reclog.verify_chain(paths.checks_log), "user_checks_by_verdict": by_verdict,
              "rows": rows, "summary": summarize(rows, activity_floor=cfg["evaluation"]["activity_floor_desk_ideas_per_month"])}
    paths.scores.mkdir(parents=True, exist_ok=True)
    reclog._atomic_write(paths.scores / f"{as_of}.json", json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return result
