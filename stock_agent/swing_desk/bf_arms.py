"""Bottom-fish prospective arms (configs/swing_desk/bottomfish_prospective_v1.json).

Code arms, frozen before 09:00 VN of the trade session and before any analyst row:
  (VN30 = the active universe since the 2026-10-08 amendment: VN100 by default; rows carry `universe`.)
  bf_baseline  MARKET-type VN30 tickers (price-only proxy) with BF-G3/G4/G7 PASS, P1 plan.
  bf_knife     same MARKET-type episodes with a floor close: entry at the open after the first
               floor close of the episode (counterfactual "catch the knife" control).
  bf_random    3 random VN30 tickers with ret_10d <= -10%, seeded by the target session date.
Analyst arm bf_desk: <= 1 type-(a) idea per session. user_bf: checks.jsonl rows with mode=bottomfish.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
import json
import random
import statistics

from ..data.exchange_calendar import next_trading_day
from . import bf_facts, reclog
from .bf_service import scan_rows
from .config import DeskPaths, code_hash, load_bf_config, load_config, universe_fields
from .history import provenance_fields

CODE_ARMS = ("bf_baseline", "bf_knife", "bf_random")
MIN_EPISODES, MIN_MARKET_EPISODES, MIN_MONTHS = 30, 3, 12


def load_protocol(paths: DeskPaths) -> dict:
    try:
        proto = json.loads(paths.bf_protocol.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError("bottomfish_prospective_v1.json is not registered: no BF row may be frozen") from None
    if proto.get("status") != "registered_before_first_log_row" or proto.get("protocol") != "bottomfish_prospective_v1":
        raise ValueError("bottom-fish protocol is not in registered state")
    return proto


def _meta(paths, state, bf, cfg, *, model_id, n_rollouts):
    from .baseline import prompt_hash
    return {"snapshot_manifest_sha256": state["manifest"]["snapshot_manifest_sha256"], "desk_code_hash": code_hash(),
            "params_hash": cfg["params_hash"], "bf_params_hash": bf["bf_params_hash"], "prompt_hash": prompt_hash(paths),
            "model_id": model_id, "n_rollouts": n_rollouts, "adjudication": None,
            "protocol": "bottomfish_prospective_v1", "mode": "bottomfish", **universe_fields(paths)}


def _row(r: dict, scan: dict, bf: dict, *, bf_type: str, entry: dict, invalidation, news: dict | None,
         events_sha: str | None) -> dict:
    p1 = bf["profiles"]["P1"]
    f, plan = r["fields"], r["plan_p1"]
    return {"ticker": r["ticker"], "action": "BUY", "mode": "bottomfish", "bf_type": bf_type, "profile": "P1",
            "episode_id": r["episode_id"] or f"{r['ticker']}-none", "attempt_no": 1, "entry": entry,
            "entry_ref": f["close"], "panic_low": f.get("panic_low"), "invalidation": invalidation,
            "catastrophe_stop_pct": p1["catastrophe_stop_pct"], "time_stop_sessions": p1["time_stop_sessions"],
            "exit_signal": p1["exit_signal"], "effective_loss_pct": plan["effective_loss_pct"],
            "earliest_sell": plan["earliest_sell"],
            "size": {"qty": plan["qty"], "risk_vnd": plan["risk_vnd"], "nav_source": scan.get("nav_source")},
            "stab_state": f["stab_state"], "fall_type_auto": r["fall_type_auto"],
            "structural_verdict": r["structural_verdict"], "structural_failed": r["structural_failed"],
            "price_only_classification": news is None, "bf_news_sha": (news or {}).get("source_sha"),
            "events_sha": events_sha, "data_as_of": scan["as_of"],
            "snapshot_sha256": scan["snapshot_manifest_sha256"], **provenance_fields(r.get("history"))}


def _first_floor_entry(df, f: dict, as_of: str, trade_session: str, floor_pct: float):
    """(entry session, causal panic low before it) for the knife arm, or None."""
    start = f.get("episode_start_date")
    if not start:
        return None
    days = [str(d)[:10] for d in df["date"]]
    c, lo = df["close"].astype(float).tolist(), df["low"].astype(float).tolist()
    for k in range(max(1, days.index(start) if start in days else 1), len(days)):
        if days[k] > as_of:
            break
        if (c[k] / c[k - 1] - 1) * 100 <= floor_pct:
            first = days.index(start)
            low_before = min(lo[first:k + 1])
            return (days[k + 1] if k + 1 < len(days) and days[k + 1] <= as_of else trade_session), low_before
    return None


def candidate_bf_rows(paths: DeskPaths, scan: dict, source: str, bf: dict | None = None) -> list[dict]:
    bf = bf or load_bf_config(paths.bf_config)
    store = bf_facts.load_bf_news(paths)
    events_sha = bf_facts.file_sha(paths.events)
    T = scan["trade_session"]
    rows = []
    if source in ("bf_baseline", "bf_knife"):
        for r in sorted(scan["rows"], key=lambda r: r["ticker"]):
            if r["fall_type_auto"] != "MARKET":
                continue
            news = bf_facts.latest_news(store, r["ticker"])
            if news and str(news.get("checked_session")) >= scan["as_of"] and news.get("class") == "fact_negative":
                continue
            news = news if news and str(news.get("checked_session")) >= scan["as_of"] else None
            f = r["fields"]
            if source == "bf_baseline":
                if r["structural_verdict"] != "PASS" or f["stab_state"] != "STAB2":
                    continue
                entry = {"type": "limit", "limit_price": round(f["close"] * bf["profiles"]["P1"]["entry_limit_mult"], 2),
                         "valid_session": T}
                rows.append(_row(r, scan, bf, bf_type="a", entry=entry, invalidation=f.get("panic_low"),
                                 news=news, events_sha=events_sha))
            else:
                knife = _first_floor_entry(scan["frames"][r["ticker"]], f, scan["as_of"], T,
                                           bf["limit"]["floor_proxy_pct"])
                if knife:
                    rows.append(_row(r, scan, bf, bf_type="a", entry={"type": "ato", "session": knife[0]},
                                     invalidation=knife[1], news=news, events_sha=events_sha))
        return rows
    if source == "bf_random":
        ev = bf["evaluation"]
        pool = [r for r in sorted(scan["rows"], key=lambda r: r["ticker"])
                if isinstance(r["fields"].get("ret_10d"), float) and r["fields"]["ret_10d"] <= ev["random_min_ret_10d_pct"]]
        pick = random.Random(int(T.replace("-", ""))).sample(pool, min(ev["random_n"], len(pool)))
        for r in sorted(pick, key=lambda r: r["ticker"]):
            entry = {"type": "limit", "limit_price": round(r["fields"]["close"] * bf["profiles"]["P1"]["entry_limit_mult"], 2),
                     "valid_session": T}
            rows.append(_row(r, scan, bf, bf_type="unclassified", entry=entry, invalidation=r["fields"].get("panic_low"),
                             news=None, events_sha=events_sha))
        return rows
    raise ValueError(f"code-generated bottom-fish arms are {CODE_ARMS}")


def _fresh(paths, now):
    from .baseline import _fresh_state
    return _fresh_state(paths, now)


def freeze_bf_arms(paths: DeskPaths, *, now: datetime) -> dict:
    load_protocol(paths)
    cfg, bf = load_config(paths.config), load_bf_config(paths.bf_config)
    state = _fresh(paths, now)
    session = state["manifest"]["trade_session"]
    logged = reclog.read_rows(paths.rec_log)
    if any(r.get("source") == "bf_desk" and r.get("target_session") == session for r in logged):
        raise ValueError(f"an analyst bf_desk row already exists for {session}: code arms must be frozen first")
    for source in CODE_ARMS:
        if any(r.get("source") == source and r.get("target_session") == session for r in logged):
            raise ValueError(f"{source} already frozen for {session}")
    scan = scan_rows(paths, now)
    if scan["as_of"] != state["manifest"]["as_of"]:
        raise ValueError("snapshot and desk state disagree; run `refresh`")
    meta = _meta(paths, state, bf, cfg, model_id="rules", n_rollouts=0)
    out = {}
    for source in CODE_ARMS:
        rows = candidate_bf_rows(paths, scan, source, bf)
        out[source] = reclog.freeze(paths.rec_log, rows, source=source, target_session=date.fromisoformat(session),
                                    session_as_of=state["manifest"]["as_of"], now=now, meta=meta)
    return out


def freeze_bf_desk(paths: DeskPaths, rows: list[dict], *, now: datetime, model_id: str, n_rollouts: int) -> list[dict]:
    load_protocol(paths)
    cfg, bf = load_config(paths.config), load_bf_config(paths.bf_config)
    state = _fresh(paths, now)
    session = state["manifest"]["trade_session"]
    if len(rows) > 1:
        raise ValueError("bf_desk allows at most 1 bottom-fish idea per session")
    if any(r.get("source") == "bf_desk" and r.get("target_session") == session for r in reclog.read_rows(paths.rec_log)):
        raise ValueError(f"bf_desk already frozen for {session}")
    for row in rows:
        t = state["tickers"].get(str(row.get("ticker", "")).upper())
        if t is None or not t.get("input_sha256"):
            label = state["manifest"].get("universe_label") or "VN30"
            raise ValueError(f"{row.get('ticker')}: bf_desk ideas must be {label} tickers with fresh verified daily "
                             "data in the desk state")
    from .baseline import with_history
    return reclog.freeze(paths.rec_log, with_history(rows, state), source="bf_desk",
                         target_session=date.fromisoformat(session),
                         session_as_of=state["manifest"]["as_of"], now=now,
                         meta=_meta(paths, state, bf, cfg, model_id=model_id, n_rollouts=n_rollouts))


# ------------------------------------------------------------------ summary

def _q(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return values[min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))]


def bf_summary(scored: list[dict], checks: list[dict]) -> dict:
    arms = defaultdict(list)
    for r in scored:
        arms[r["source"]].append(r)
    out = {"arms": {}, "paired": {}, "activity": {}, "user_bf_by_verdict": {}}
    for arm, rows in sorted(arms.items()):
        res = [r for r in rows if r["status"] == "resolved"]
        nf = [r for r in rows if r["status"] == "no_fill"]
        nets = [r["net_pct"] for r in res]
        out["arms"][arm] = {
            "rows": len(rows), "resolved": len(res), "pending": sum(r["status"] == "pending" for r in rows),
            "no_fill": len(nf), "episodes": len({r.get("episode_id") for r in res}),
            "mean_net_pct": statistics.mean(nets) if nets else None,
            "median_net_pct": statistics.median(nets) if nets else None,
            "mean_net_pct_stress": statistics.mean(r["net_pct_stress"] for r in res) if res else None,
            "win_rate": sum(r["win"] for r in res) / len(res) * 100 if res else None,
            "net_p10_pct": _q(nets, 0.1), "mae_p10_pct": _q([r["mae_pct"] for r in res], 0.1),
            "floor_lock_days_mean": statistics.mean(r["floor_lock_days"] for r in res) if res else None}
        months = defaultdict(set)
        for r in rows:
            months[r["target_session"][:7]].add(r.get("episode_id"))
        out["activity"][arm] = {m: {"new_episodes": len(v)} for m, v in sorted(months.items())}

    def by(arm, key):
        groups = defaultdict(list)
        for r in arms.get(arm, []):
            if r["status"] == "resolved":
                groups[r[key]].append(r["net_pct"])
        return {k: statistics.mean(v) for k, v in groups.items()}

    for a, b, key in (("bf_baseline", "bf_random", "target_session"), ("bf_baseline", "bf_knife", "episode_id")):
        pa, pb = by(a, key), by(b, key)
        common = sorted(set(pa) & set(pb))
        if common:
            out["paired"][f"{a}_minus_{b}"] = {"pairs": len(common),
                                              "mean_diff_pct": statistics.mean(pa[k] - pb[k] for k in common)}
    for c in checks:
        if c.get("mode") == "bottomfish":
            out["user_bf_by_verdict"][c.get("verdict")] = out["user_bf_by_verdict"].get(c.get("verdict"), 0) + 1
    base = out["arms"].get("bf_baseline", {})
    out["conclusion"] = ("NOT PROVEN: below the registered floor (>= 30 resolved episodes per arm, >= 3 market episodes, "
                         ">= 12 months); describe only" if base.get("episodes", 0) < MIN_EPISODES else
                         "Floor may be reached: run the registered episode block bootstrap before any claim")
    out["activity_floor"] = "none for bottom-fish (zero-episode months are expected)"
    return out
