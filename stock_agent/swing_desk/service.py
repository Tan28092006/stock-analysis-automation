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
from .config import (DeskPaths, code_hash, concentration, load_config, load_primary_symbols, load_universe_symbols,
                     universe_fields, universe_info)
from .gate import FAIL, evaluate_gate, verdict
from .history import HistoryResolver, provenance_fields, store_line
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


def mark_events(paths: DeskPaths, ticker: str, session: str, events: list[dict], source: str, *,
                pending_dilution: bool = False) -> dict:
    """Record that the event calendar was checked for `ticker` (agent/user supplied, cited).

    `pending_dilution` marks the recorded events as an unresolved dilution vote/issue (bottom-fish
    BF-G5) until `bf_facts.mark_resolved` records the result.
    """
    data = load_events(paths) or {"checked": {}, "events": []}
    data["checked"][ticker.upper()] = session
    data["events"] = [e for e in data["events"] if e.get("ticker") != ticker.upper() or e.get("date", "") < session]
    for ev in events:
        data["events"].append({"ticker": ticker.upper(), "source": source, **ev,
                               **({"pending_dilution": True} if pending_dilution else {})})
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
    active = universe_fields(paths)
    if (manifest.get("universe") or "vn30") != active["universe"] or \
            manifest.get("universe_source_sha", active["universe_source_sha"]) != active["universe_source_sha"]:
        return {**fresh, "ok": False, "reason": f"desk state was built for universe {manifest.get('universe') or 'vn30'}, "
                                                f"active is {active['universe']}; run `refresh`"}
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
          s2_override: list[str] | None = None, mode: str = "swing", profile: str | None = None,
          tranche: int | None = None, why: str | None = None) -> dict:
    if mode == "bottomfish":
        from .bf_service import check_bottomfish
        return check_bottomfish(paths, ticker, now=now, price=price, amount=amount, reason=reason, emotion=emotion,
                                ack=ack, nav=nav, record=record, profile=profile, tranche=tranche, why=why)
    if mode != "swing":
        raise ValueError("mode must be swing or bottomfish")
    cfg = load_config(paths.config)
    ticker = ticker.upper()
    tday = trade_session(now)
    fresh = dl.freshness(paths, now)
    uinfo = universe_info(paths)
    universe, primary = uinfo["symbols"], load_primary_symbols(paths)
    conc = concentration(paths, cfg)
    checks = reclog.read_rows(paths.checks_log)
    out = {"ticker": ticker, "checked_at": now.astimezone(timezone.utc).isoformat(), "trade_session": str(tday),
           "expected_session": fresh["expected_session"], "data_as_of": fresh["as_of"],
           "snapshot_manifest_sha256": fresh["manifest_sha256"], "price": price, "amount": amount,
           "reason": reason, "emotion": emotion, "params_hash": cfg["params_hash"], "desk_code_hash": code_hash(),
           "earliest_sell": f"{earliest_sell(tday)} {SELL_FROM}", "horizon_warning": horizon_warning(tday),
           "eod_note": f"EOD data as of {fresh['as_of'] or 'n/a'}"}
    ind = market = plan = hist = None
    rules = []
    if fresh["ok"]:
        frames = dl.load_desk_frames(fresh)
        lookup = dl.close_lookup(frames)
        journal = load_journal(paths.portfolio, price_lookup=lookup, checks=checks)
        nav_v, nav_source, portfolio, exposure = _portfolio(journal, nav, cfg)
        out["nav_source"] = nav_source
        out["journal_errors"] = journal.errors if journal else None
        market = dl.market_summary(Path(fresh["manifest"]), dl.primary_frames(frames, primary), primary, cfg)
        if ticker in universe and ticker in frames:
            long_df, hist = HistoryResolver(paths, fresh["as_of"]).series(ticker, frames[ticker])
            ind = compute_ticker(frames[ticker], limit_proxy_pct=cfg["gate"]["limit_proxy_pct"],
                                 excursion_lookback=cfg["excursion"]["lookback_entries"], long_df=long_df)
            entry = price or ind["close"]
            plan = size_plan(entry, nav_v, cfg, current_exposure=exposure, amount=amount)
        if ticker in universe and ticker not in primary:
            s2_list, s2_status = [], "NA (S2 = VN30 MR scan on the paper snapshot; not defined for this ticker)"
        elif s2_override is not None:
            s2_list, s2_status = s2_override, "override"
        elif ticker in universe:
            s2_list, s2_status = dl.s2_symbols(Path(fresh["manifest"]), fresh["as_of"])
        else:
            s2_list, s2_status = [], "NA"
        out["s2_status"] = s2_status
        g = evaluate_gate(ticker=ticker, ind=ind, market=market, freshness=fresh, universe=universe, cfg=cfg,
                          trade_day=tday, plan=plan, price=price, s2=ticker in s2_list, portfolio=portfolio,
                          events=load_events(paths), universe_label=uinfo["label"], concentration=conc)
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
                          trade_day=tday, plan=None, universe_label=uinfo["label"])
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
    out["history"] = hist   # which series the long-window fields used (None: no indicators computed)
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
        row.update(provenance_fields(out["history"]))
        row.update(universe_fields(paths))
        row.update(ticker_data_fields(fresh, ticker))
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
    uinfo = universe_info(paths)
    universe, primary = uinfo["symbols"], load_primary_symbols(paths)
    conc = concentration(paths, cfg)
    frames = dl.load_desk_frames(fresh)
    market = dl.market_summary(manifest, dl.primary_frames(frames, primary), primary, cfg)
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
    s2_list = [s for s in s2_list if s in primary]     # S2 (VN30 MR scan) is defined on the paper basket only
    health, flows = (dl.flows_snapshot(paths, primary, now, fresh.get("fetched_at"))
                     if with_flows_data else ({"status": "UNVERIFIED", "reason": "skipped"}, None))
    events = load_events(paths)
    journal = load_journal(paths.portfolio, price_lookup=dl.close_lookup(frames))
    nav_v, nav_source, _, _ = _portfolio(journal, None, cfg)
    resolver = HistoryResolver(paths, fresh["as_of"])
    tickers, provs = {}, {}
    for sym in universe:
        if sym not in frames:   # extended universe: no fresh verified daily data -> G0 UNVERIFIED (WARN)
            g = evaluate_gate(ticker=sym, ind=None, market=market, freshness=fresh, universe=universe, cfg=cfg,
                              trade_day=tday, plan=None, universe_label=uinfo["label"])
            tickers[sym] = jsonable({
                "ticker": sym, "as_of": None, "input_sha256": None, "indicators": {},
                "flows": {"flow_status": "UNVERIFIED", "reason": "no fresh daily data"}, "setups": [],
                "gate": g["verdict"], "structural_gate": structural_verdict(g["rules"]),
                "structural_gate_ex_setup": structural_verdict(g["rules"], STRUCTURAL_EXCLUDE | {"G11"}),
                "rules": g["rules"], "failed": g["failed"], "unverified": g["unverified"], "target": None,
                "reward_risk": None, "plan_default_nav": None, "earliest_sell": f"{earliest_sell(tday)} {SELL_FROM}",
                "data_reason": (fresh.get("tickers") or {}).get(sym, {}).get("reason"), **provenance_fields(None)})
            continue
        long_df, provs[sym] = resolver.series(sym, frames[sym])
        ind = compute_ticker(frames[sym], limit_proxy_pct=cfg["gate"]["limit_proxy_pct"],
                             excursion_lookback=cfg["excursion"]["lookback_entries"], long_df=long_df)
        plan = size_plan(ind["close"], nav_v, cfg)
        g = evaluate_gate(ticker=sym, ind=ind, market=market, freshness=fresh, universe=universe, cfg=cfg,
                          trade_day=tday, plan=plan, s2=sym in s2_list, portfolio={}, events=events,
                          universe_label=uinfo["label"], concentration=conc)
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
            "earliest_sell": f"{earliest_sell(tday)} {SELL_FROM}",
            **provenance_fields(provs[sym]),
            **{k: provs[sym][k] for k in ("history_data_issues", "history_notes", "history_mismatch_detail")
               if k in provs[sym]}})
    state_manifest = {
        "session": fresh["as_of"], "as_of": fresh["as_of"], "trade_session": str(tday),
        "snapshot_manifest": fresh["manifest"], "snapshot_manifest_sha256": fresh["manifest_sha256"],
        "file_sha256": fresh["file_sha256"], "flows_health": health, "s2_status": s2_status,
        "s2_symbols": s2_list, "desk_code_hash": code_hash(), "params_hash": cfg["params_hash"],
        "nav_source": nav_source, "history": resolver.summary(provs),
        "created_at": now.astimezone(timezone.utc).isoformat(),
        **universe_fields(paths), "universe_label": uinfo["label"], "universe_count": len(universe),
        "limitations": ["Advisory only; no orders.", "Provider-adjusted prices; may differ from board prices.",
                        f"Current fixed {uinfo['label']} universe (survivorship).",
                        "G9 (portfolio) and G10 (events) are UNVERIFIED in state; `check` evaluates them."]}
    if fresh.get("tickers"):
        state_manifest["universe_snapshots"] = fresh.get("universe_manifests") or {}
        state_manifest["ticker_data"] = {
            "primary": sorted(primary), "fresh_count": sum(1 for s in universe if s in frames),
            "unverified": sorted(s for s in universe if s not in frames),
            "reasons": {s: t["reason"] for s, t in sorted(fresh["tickers"].items()) if not t["ok"]},
            "regime_breadth_s2_basis": "paper VN30 snapshot (unchanged definitions)"}
    return {"manifest": jsonable(state_manifest), "market": jsonable(market), "tickers": tickers}


def ticker_data_fields(fresh: dict, ticker: str) -> dict:
    """Which verified daily snapshot a ticker's data came from (row provenance)."""
    t = (fresh.get("tickers") or {}).get(ticker.upper())
    if t is not None:
        return {"ticker_data_source": t["source"] if t["ok"] else "UNVERIFIED",
                "ticker_snapshot_manifest_sha256": t.get("manifest_sha256") if t["ok"] else None}
    ok = ticker.upper() in (fresh.get("file_sha256") or {})
    return {"ticker_data_source": "primary" if ok else None,
            "ticker_snapshot_manifest_sha256": fresh.get("manifest_sha256") if ok else None}


def history_lines(summary: dict | None) -> list[str]:
    """Gate/scan table lines naming the long-window source (none with --history off: old tables)."""
    if not summary or summary.get("history_mode") != "on":
        return []
    stores = ", ".join(summary.get("stores") or {}) or "none"
    lines = [f"Long windows (SMA200, high_250, MAE/MFE, episode): history_source {summary.get('history_source')} "
             f"(store {stores}); snapshot_650d fallback: {', '.join(summary.get('fallbacks') or []) or 'none'}"]
    if summary.get("mismatches"):
        lines.append(f"WARN history_mismatch (store != snapshot, not stitched): {', '.join(summary['mismatches'])}")
    return lines + [""]


def universe_lines(m: dict) -> list[str]:
    """Scan header line for an extended universe (none for VN30: old tables)."""
    td = m.get("ticker_data")
    if not td:
        return []
    bad = td.get("unverified") or []
    return [f"Universe: {m.get('universe_label')} ({m.get('universe_count')} mã; dữ liệu ngày tươi {td['fresh_count']}; "
            f"UNVERIFIED (G0, WARN): {', '.join(bad) or 'none'}). Regime/breadth/S2 vẫn tính trên rổ VN30 của "
            "snapshot paper.", ""]


def gate_table(state: dict) -> str:
    m, mk = state["manifest"], state["market"]
    lines = [f"# Gate table — dữ liệu EOD tới {m['as_of']} (snapshot sha {m['snapshot_manifest_sha256'][:12]})", "",
             f"Regime: {mk['regime']} | VNINDEX {mk.get('index_close')} vs EMA50 {mk.get('index_ema50')} | "
             f"breadth {mk.get('breadth_pct')}% | phiên giao dịch kế: {m['trade_session']} | "
             f"đáo hạn VN30F kế: {mk.get('next_vn30f_expiry')}", "",
             "Structural = G0-G8 + G11 (G9 portfolio, G10 events đánh giá trong `check`).", "",
             *universe_lines(m),
             *history_lines(m.get("history")),
             "| Mã | Structural | Setup | Close | RSI2 | ATR% | SMA50 | R:R | Luật trượt |",
             "|---|---|---|---|---|---|---|---|---|"]
    order = sorted(state["tickers"].values(), key=lambda t: ({"PASS": 0, "WARN": 1, "BLOCK": 2}[t["structural_gate"]],
                                                              t["indicators"].get("rsi2") or 999))
    for t in order:
        ind = t["indicators"]
        failed = [r["id"] for r in t["rules"] if r["status"] == "FAIL" and r["id"] not in STRUCTURAL_EXCLUDE]
        failed += [f"{r['id']} UNVERIFIED" for r in t["rules"] if r["id"] == "G0" and r["status"] == "UNVERIFIED"]
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
           "history_store": store_line(paths, now, fresh.get("as_of")),
           "orders": "advisory only; the desk never places orders"}
    info = universe_info(paths)
    out["universe"] = {"name": info["name"], "label": info["label"], "count": len(info["symbols"]),
                       "source_sha": info["sha256"], "effective_from": info.get("effective_from")}
    if fresh.get("tickers"):
        bad = sorted(s for s, t in fresh["tickers"].items() if not t["ok"])
        out["universe"].update(fresh_daily=len(fresh["tickers"]) - len(bad), unverified_daily=bad)
        if bad:
            out["universe"]["hint"] = "run `refresh` to fetch the daily snapshot for the names outside VN30"
    if with_flows:
        out["foreign_flows"], _ = dl.flows_snapshot(paths, load_primary_symbols(paths), now, None)
    return out
