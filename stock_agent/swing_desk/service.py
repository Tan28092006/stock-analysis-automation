"""Desk services: pre-trade check, state build, scan and status (deterministic, offline)."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import uuid

from ..data.exchange_calendar import completed_session_date, next_trading_day
from . import data as dl
from .behaviour import behaviour_block, evaluate_behaviour
from .calendar_events import (SELL_FROM, earliest_sell, holiday_table_end, horizon_warning,
                              next_vn30f_expiry, trade_session)
from .config import DeskPaths, code_hash, load_config, load_universe_symbols
from .gate import FAIL, evaluate_gate, verdict
from .indicators import compute_ticker, flow_fields, jsonable
from .journal import load_journal
from . import reclog
from .sizing import plan as size_plan

STRUCTURAL_EXCLUDE = {"G9", "G10"}   # portfolio- and agent-dependent; identical for every arm


def structural_verdict(rules: list[dict], exclude=STRUCTURAL_EXCLUDE) -> str:
    return verdict([r for r in rules if r["id"] not in exclude])


def load_events(paths: DeskPaths) -> dict | None:
    if not paths.events.is_file():
        return None
    try:
        return json.loads(paths.events.read_text(encoding="utf-8"))
    except ValueError:
        return None


def mark_events(paths: DeskPaths, ticker: str, session: str, events: list[dict], source: str) -> dict:
    """Record that the event calendar was checked for `ticker` (agent/user supplied, cited)."""
    data = load_events(paths) or {"checked": {}, "events": []}
    data["checked"][ticker.upper()] = session
    data["events"] = [e for e in data["events"] if e.get("ticker") != ticker.upper() or e.get("date", "") < session]
    for ev in events:
        data["events"].append({"ticker": ticker.upper(), "source": source, **ev})
    paths.events.parent.mkdir(parents=True, exist_ok=True)
    reclog._atomic_write(paths.events, json.dumps(data, ensure_ascii=False, indent=2))
    return data


def read_state(paths: DeskPaths) -> tuple[Path | None, dict | None]:
    pointer = paths.state / "LATEST"
    if not pointer.is_file():
        return None, None
    folder = paths.state / pointer.read_text(encoding="utf-8").strip()
    try:
        return folder, json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None


def state_freshness(paths: DeskPaths, now: datetime) -> dict:
    """G0 for state readers: state must be built from the snapshot that verifies now."""
    fresh = dl.freshness(paths, now)
    folder, manifest = read_state(paths)
    if not fresh["ok"]:
        return {**fresh, "state": str(folder) if folder else None}
    if manifest is None:
        return {**fresh, "ok": False, "reason": "no desk state; run `refresh`"}
    if manifest.get("snapshot_manifest_sha256") != fresh["manifest_sha256"] or manifest.get("as_of") != fresh["as_of"]:
        return {**fresh, "ok": False, "reason": "desk state was built from another snapshot; run `refresh`"}
    for path in sorted((folder / "tickers").glob("*.json")):
        t = json.loads(path.read_text(encoding="utf-8"))
        if t.get("input_sha256") != fresh["file_sha256"].get(t.get("ticker")):
            return {**fresh, "ok": False, "reason": f"state ticker {path.stem} input hash mismatch"}
    return {**fresh, "state": str(folder)}


def _portfolio(journal, nav_arg, cfg):
    if nav_arg:
        nav, source = float(nav_arg), "--nav"
    elif journal is not None and journal.nav():
        nav, source = journal.nav(), "journal"
    else:
        nav, source = float(cfg["money"]["account_nav_default"]), "default (no NAV/CASH line in journal)"
    if journal is None:
        return nav, source, None, 0.0
    exposure = sum(p["cost_value"] for p in journal.positions.values())
    return nav, source, {"positions": journal.positions, "nav": nav}, exposure


def check(paths: DeskPaths, ticker: str, *, now: datetime, price: float | None = None,
          amount: float | None = None, reason: str | None = None, emotion: str | None = None,
          ack: bool = False, nav: float | None = None, record: bool = True,
          s2_override: list[str] | None = None) -> dict:
    cfg = load_config(paths.config)
    ticker = ticker.upper()
    tday = trade_session(now)
    fresh = dl.freshness(paths, now)
    universe = load_universe_symbols(paths)
    checks = reclog.read_rows(paths.checks_log)
    out = {"ticker": ticker, "checked_at": now.astimezone(timezone.utc).isoformat(), "trade_session": str(tday),
           "expected_session": fresh["expected_session"], "data_as_of": fresh["as_of"],
           "snapshot_manifest_sha256": fresh["manifest_sha256"], "price": price, "amount": amount,
           "reason": reason, "emotion": emotion, "params_hash": cfg["params_hash"], "desk_code_hash": code_hash(),
           "earliest_sell": f"{earliest_sell(tday)} {SELL_FROM}", "horizon_warning": horizon_warning(tday),
           "eod_note": f"EOD data as of {fresh['as_of'] or 'n/a'}"}
    ind = market = plan = None
    rules = []
    if fresh["ok"]:
        frames = dl.load_frames(Path(fresh["manifest"]))
        lookup = dl.close_lookup(frames)
        journal = load_journal(paths.portfolio, price_lookup=lookup, checks=checks)
        nav_v, nav_source, portfolio, exposure = _portfolio(journal, nav, cfg)
        out["nav_source"] = nav_source
        out["journal_errors"] = journal.errors if journal else None
        market = dl.market_summary(Path(fresh["manifest"]), frames, universe, cfg)
        if ticker in universe and ticker in frames:
            ind = compute_ticker(frames[ticker], limit_proxy_pct=cfg["gate"]["limit_proxy_pct"],
                                 excursion_lookback=cfg["excursion"]["lookback_entries"])
            entry = price or ind["close"]
            plan = size_plan(entry, nav_v, cfg, current_exposure=exposure, amount=amount)
        if s2_override is not None:
            s2_list, s2_status = s2_override, "override"
        elif ticker in universe:
            s2_list, s2_status = dl.s2_symbols(Path(fresh["manifest"]), fresh["as_of"])
        else:
            s2_list, s2_status = [], "NA"
        out["s2_status"] = s2_status
        g = evaluate_gate(ticker=ticker, ind=ind, market=market, freshness=fresh, universe=universe, cfg=cfg,
                          trade_day=tday, plan=plan, price=price, s2=ticker in s2_list, portfolio=portfolio,
                          events=load_events(paths))
        rules = g["rules"]
        if ind is not None and not g["out_of_scope"]:
            rules += evaluate_behaviour(ticker=ticker, ind=ind, journal=journal, trade_day=tday, cfg=cfg, plan=plan,
                                        price=price, reason=reason, setups=g["setups"], checks=checks)
        out.update(setups=g["setups"], target=g.get("target"), reward_risk=g.get("reward_risk"),
                   out_of_scope=g["out_of_scope"])
        thesis = [e for e in journal.theses(ticker) if e.date >= fresh["as_of"]] if journal else []
        out["thesis_present"] = bool(thesis)
    else:
        g = evaluate_gate(ticker=ticker, ind=None, market={}, freshness=fresh, universe=universe, cfg=cfg,
                          trade_day=tday, plan=None)
        rules = g["rules"]
        out.update(setups=[], out_of_scope=False, thesis_present=False)
    out["verdict"] = verdict(rules)
    out["rules"] = rules
    out["failed"] = [r["id"] for r in rules if r["status"] == FAIL]
    out["unverified"] = [r["id"] for r in rules if r["status"] == "UNVERIFIED"]
    out["behaviour_block"] = behaviour_block(rules)
    out["recheck_from"] = f"{next_trading_day(tday)} 09:00 (ATO)" if out["behaviour_block"] else None
    cooling = cfg["behaviour"].get("cooling_off_enabled", True)
    out["thesis_required"] = bool(cooling and out["verdict"] == "WARN")
    out["acknowledged"] = bool(ack and out["verdict"] == "WARN" and out["thesis_present"])
    out["ack_refused"] = bool(ack and not out["acknowledged"])
    out["indicators"] = ind
    out["market"] = market
    out["plan"] = plan
    if ind is not None:
        s1 = "S1" in out["setups"]
        out["exit_plan"] = ("S1: bán khi đóng cửa > SMA5 hoặc sau 8 phiên; stop như trên" if s1 else
                            "Chung: target / stop / tối đa 8 phiên")
    state_folder, state_manifest = read_state(paths)
    if state_folder and state_manifest and state_manifest.get("as_of") == fresh.get("as_of") \
            and (state_folder / "tickers" / f"{ticker}.json").is_file():
        out["flows"] = json.loads((state_folder / "tickers" / f"{ticker}.json").read_text(encoding="utf-8")).get("flows")
    else:
        out["flows"] = {"flow_status": "UNVERIFIED", "reason": "no desk state for this session (run refresh)"}
    out = jsonable(out)
    if record:
        row = {k: out[k] for k in ("ticker", "trade_session", "checked_at", "verdict", "failed", "unverified",
                                   "behaviour_block", "price", "amount", "reason", "emotion", "data_as_of",
                                   "snapshot_manifest_sha256", "params_hash", "desk_code_hash", "thesis_present",
                                   "acknowledged", "setups")}
        row["source"] = "user_check"
        row["qty_suggested"] = (out["plan"] or {}).get("qty")
        row["entry_ref"] = (out["plan"] or {}).get("entry_ref")
        row["stop_user"] = (out["plan"] or {}).get("stop_user")
        reclog.append(paths.checks_log, row)
    return out


# ------------------------------------------------------------------ state build

def compute_state(paths: DeskPaths, now: datetime, *, s2_override: list[str] | None = None,
                  with_flows_data: bool = True) -> dict:
    cfg = load_config(paths.config)
    fresh = dl.freshness(paths, now)
    if not fresh["ok"]:
        raise ValueError(f"G0: {fresh['reason']}")
    manifest = Path(fresh["manifest"])
    universe = load_universe_symbols(paths)
    frames = dl.load_frames(manifest)
    market = dl.market_summary(manifest, frames, universe, cfg)
    tday = next_trading_day(completed_session_date(now))
    market.update(next_vn30f_expiry=str(next_vn30f_expiry(tday)), holiday_table_ends=str(holiday_table_end()),
                  horizon_warning=horizon_warning(tday), trade_session=str(tday))
    try:
        market["universe_effective_from"] = json.loads(paths.universe.read_text(encoding="utf-8")).get("effective_from")
    except ValueError:
        market["universe_effective_from"] = None
    if s2_override is not None:
        s2_list, s2_status = s2_override, "override"
    else:
        s2_list, s2_status = dl.s2_symbols(manifest, fresh["as_of"])
    health, flows = (dl.flows_snapshot(paths, universe, now, fresh.get("fetched_at"))
                     if with_flows_data else ({"status": "UNVERIFIED", "reason": "skipped"}, None))
    events = load_events(paths)
    journal = load_journal(paths.portfolio, price_lookup=dl.close_lookup(frames))
    nav_v, nav_source, _, _ = _portfolio(journal, None, cfg)
    tickers = {}
    for sym in universe:
        ind = compute_ticker(frames[sym], limit_proxy_pct=cfg["gate"]["limit_proxy_pct"],
                             excursion_lookback=cfg["excursion"]["lookback_entries"])
        plan = size_plan(ind["close"], nav_v, cfg)
        g = evaluate_gate(ticker=sym, ind=ind, market=market, freshness=fresh, universe=universe, cfg=cfg,
                          trade_day=tday, plan=plan, s2=sym in s2_list, portfolio={}, events=events)
        fl = flow_fields(flows, sym, fresh["as_of"], ind["avg_value_20_bn"])
        if health.get("status") != "ready" and fl["flow_status"] == "OK":
            fl["flow_status"] = "UNVERIFIED"
        tickers[sym] = jsonable({
            "ticker": sym, "as_of": fresh["as_of"], "input_sha256": fresh["file_sha256"][sym],
            "indicators": ind, "flows": fl, "setups": g["setups"], "gate": g["verdict"],
            "structural_gate": structural_verdict(g["rules"]),
            "structural_gate_ex_setup": structural_verdict(g["rules"], STRUCTURAL_EXCLUDE | {"G11"}),
            "rules": g["rules"], "failed": g["failed"], "unverified": g["unverified"],
            "target": g.get("target"), "reward_risk": g.get("reward_risk"),
            "plan_default_nav": {**{k: plan[k] for k in ("entry_ref", "stop_user", "stop_validated", "qty",
                                                         "vnd_at_risk_nominal", "vnd_at_risk_effective", "nav")},
                                 "nav_source": nav_source},
            "earliest_sell": f"{earliest_sell(tday)} {SELL_FROM}"})
    state_manifest = {
        "session": fresh["as_of"], "as_of": fresh["as_of"], "trade_session": str(tday),
        "snapshot_manifest": fresh["manifest"], "snapshot_manifest_sha256": fresh["manifest_sha256"],
        "file_sha256": fresh["file_sha256"], "flows_health": health, "s2_status": s2_status,
        "s2_symbols": s2_list, "desk_code_hash": code_hash(), "params_hash": cfg["params_hash"],
        "nav_source": nav_source,
        "created_at": now.astimezone(timezone.utc).isoformat(),
        "limitations": ["Advisory only; no orders.", "Provider-adjusted prices; may differ from board prices.",
                        "Current fixed VN30 universe (survivorship).",
                        "G9 (portfolio) and G10 (events) are UNVERIFIED in state; `check` evaluates them."]}
    return {"manifest": jsonable(state_manifest), "market": jsonable(market), "tickers": tickers}


def gate_table(state: dict) -> str:
    m, mk = state["manifest"], state["market"]
    lines = [f"# Gate table — dữ liệu EOD tới {m['as_of']} (snapshot sha {m['snapshot_manifest_sha256'][:12]})", "",
             f"Regime: {mk['regime']} | VNINDEX {mk.get('index_close')} vs EMA50 {mk.get('index_ema50')} | "
             f"breadth {mk.get('breadth_pct')}% | phiên giao dịch kế: {m['trade_session']} | "
             f"đáo hạn VN30F kế: {mk.get('next_vn30f_expiry')}", "",
             "Structural = G0-G8 + G11 (G9 portfolio, G10 events đánh giá trong `check`).", "",
             "| Mã | Structural | Setup | Close | RSI2 | ATR% | SMA50 | R:R | Luật trượt |",
             "|---|---|---|---|---|---|---|---|---|"]
    order = sorted(state["tickers"].values(), key=lambda t: ({"PASS": 0, "WARN": 1, "BLOCK": 2}[t["structural_gate"]],
                                                              t["indicators"].get("rsi2") or 999))
    for t in order:
        ind = t["indicators"]
        failed = [r["id"] for r in t["rules"] if r["status"] == "FAIL" and r["id"] not in STRUCTURAL_EXCLUDE]
        fmt = lambda v, f: (f.format(v) if v is not None else "n/a")  # noqa: E731
        lines.append(f"| {t['ticker']} | {t['structural_gate']} | {'+'.join(t['setups']) or '-'} | "
                     f"{fmt(ind.get('close'), '{:,.0f}')} | {fmt(ind.get('rsi2'), '{:.1f}')} | "
                     f"{fmt(ind.get('atr_pct'), '{:.1f}')} | {fmt(ind.get('sma50'), '{:,.0f}')} | "
                     f"{fmt(t.get('reward_risk'), '{:.2f}')} | {', '.join(failed) or '-'} |")
    lines += ["", "Advisory only. PASS = không có cờ cấu trúc và có setup đã biết, không phải 'sẽ thắng'."]
    return "\n".join(lines) + "\n"


def write_state(paths: DeskPaths, state: dict, *, extra_files: dict[str, str] | None = None) -> Path:
    """Atomic: build in a temp dir, swap into state/<session>/, then move the LATEST pointer."""
    paths.state.mkdir(parents=True, exist_ok=True)
    session = state["manifest"]["session"]
    tmp = paths.state / f".tmp-{uuid.uuid4().hex}"
    tmp.mkdir()
    try:
        (tmp / "tickers").mkdir()
        dump = lambda p, d: p.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False),  # noqa: E731
                                         encoding="utf-8")
        dump(tmp / "manifest.json", state["manifest"])
        dump(tmp / "market.json", state["market"])
        for sym, t in state["tickers"].items():
            dump(tmp / "tickers" / f"{sym}.json", t)
        (tmp / "gate_table.md").write_text(gate_table(state), encoding="utf-8")
        for name, text in (extra_files or {}).items():
            (tmp / name).write_text(text, encoding="utf-8")
        final = paths.state / session
        old = None
        if final.exists():
            old = paths.state / f".old-{uuid.uuid4().hex}"
            os.replace(final, old)
        os.replace(tmp, final)
        reclog._atomic_write(paths.state / "LATEST", session)
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)
        return final
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def load_state(paths: DeskPaths) -> dict | None:
    folder, manifest = read_state(paths)
    if folder is None:
        return None
    tickers = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((folder / "tickers").glob("*.json"))}
    return {"folder": str(folder), "manifest": manifest,
            "market": json.loads((folder / "market.json").read_text(encoding="utf-8")), "tickers": tickers}


def status(paths: DeskPaths, now: datetime, *, with_flows: bool = True) -> dict:
    fresh = dl.freshness(paths, now)
    folder, manifest = read_state(paths)
    out = {"now_utc": now.astimezone(timezone.utc).isoformat(),
           "completed_session": str(completed_session_date(now)),
           "trade_session": str(trade_session(now)),
           "snapshot": {k: fresh.get(k) for k in ("ok", "as_of", "manifest", "manifest_sha256", "reason")},
           "desk_state": {"session": manifest.get("session") if manifest else None,
                          "folder": str(folder) if folder else None,
                          "matches_snapshot": bool(manifest and manifest.get("snapshot_manifest_sha256") == fresh["manifest_sha256"])},
           "holiday_table_ends": str(holiday_table_end()),
           "horizon_warning": horizon_warning(trade_session(now)),
           "legacy_prices_hist": dl.legacy_prices_freshness(paths, now),
           "orders": "advisory only; the desk never places orders"}
    if with_flows:
        out["foreign_flows"], _ = dl.flows_snapshot(paths, load_universe_symbols(paths), now, None)
    return out
