"""`check --mode bottomfish` and `scan --mode bottomfish` (deterministic, offline, advisory only).

Swing mode is untouched: `service.check` dispatches here only for mode="bottomfish".
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path

from ..data.exchange_calendar import next_trading_day
from . import bf_facts, reclog
from . import data as dl
from .behaviour import behaviour_block, evaluate_behaviour
from .bf_indicators import compute_bottomfish_fields, episode_id
from .bottomfish import (CAVEAT, P3_NOTE, bf_plan, classify_fall, evaluate_bf_behaviour, evaluate_bf_gate,
                         held_facts, _episode_buys)
from .calendar_events import SELL_FROM, earliest_sell, horizon_warning, trade_session
from .config import DeskPaths, code_hash, load_bf_config, load_config, load_universe_symbols
from .gate import FAIL, PASS, verdict
from .indicators import compute_ticker, jsonable
from .journal import load_journal
from .sizing import effective_loss_pct

CHECKLIST = Path("swing_desk/prompts/BOTTOMFISH_CHECKLIST.md")
DEFAULT_NOTE = "Khuyến nghị mặc định: P0 — không bắt đáy."


def _sector_mean(ticker: str, frames: dict, cfg: dict):
    for members in cfg.get("sectors", {}).values():
        if ticker in members:
            rets = []
            for s in members:
                df = frames.get(s)
                if df is not None and len(df) > 10:
                    c = df["close"].astype(float).to_numpy()
                    rets.append((c[-1] / c[-11] - 1) * 100)
            return sum(rets) / len(rets) if rets else None
    return None


def _recheck(failed: list[str], fall: dict, profile: str, behaviour: bool, tday) -> list[str]:
    out = []
    if behaviour:
        out.append(f"cờ hành vi hard: chỉ kiểm lại từ {next_trading_day(tday)} 09:00 (ATO)")
    if "BF-G1" in failed:
        out.append("ngoài VN30: không có lệnh bắt đáy mới (chỉ HOLD/SELL/AVOID cho vị thế đang giữ)")
    if {"BF-G3", "BF-G4"} & set(failed):
        out.append(">= 2 phiên không sàn, đáy sau cao hơn đáy trước, không có đáy mới của đợt giảm")
    if "BF-G5" in failed:
        out.append("sau khi kết quả sự kiện + điều khoản phát hành được ghi (events --resolved) và ngoài cửa sổ 10 phiên")
    if fall.get("fall_type") == "b" or "BF-G6" in failed:
        out.append("loại (b): chỉ hồ sơ P3 (<= 1% NAV, 'no evidence of edge; measurement only'), >= 10 phiên sau tin FACT cuối")
    if "BF-G8" in failed:
        out.append("ghi trạng thái margin/cảnh báo đã kiểm (bf-news --margin ...)")
    if "BF-G12" in failed and profile != "P0":
        out.append("viết dòng THESIS (bf_type=, profile=, invalid=, time_stop=, max_attempts=, checklist=done)")
    if profile == "P0":
        out.append("hồ sơ P0 (mặc định) không bắt đáy; chỉ đổi hồ sơ khi bạn chủ động chọn P1/P2/P3")
    return out


def check_bottomfish(paths: DeskPaths, ticker: str, *, now: datetime, price: float | None = None,
                     amount: float | None = None, reason: str | None = None, emotion: str | None = None,
                     ack: bool = False, nav: float | None = None, record: bool = True, profile: str | None = None,
                     tranche: int | None = None, why: str | None = None) -> dict:
    from .service import _portfolio, load_events
    cfg = load_config(paths.config)
    bf = load_bf_config(paths.bf_config)
    ticker = ticker.upper()
    profile = (profile or bf["default_profile"]).upper()
    if profile not in bf["profiles"]:
        raise ValueError(f"profile must be one of {sorted(bf['profiles'])}")
    tday = trade_session(now)
    fresh = dl.freshness(paths, now)
    universe = load_universe_symbols(paths)
    in_scope = ticker in universe or ticker in bf["scope"].get("extra_universe", [])
    checks = reclog.read_rows(paths.checks_log)
    frames = dl.load_frames(Path(fresh["manifest"])) if fresh["ok"] else {}
    market = dl.market_summary(Path(fresh["manifest"]), frames, universe, cfg) if fresh["ok"] else {}
    extra, df, idx = {}, None, None
    data_as_of, snap_sha = fresh["as_of"], fresh["manifest_sha256"]
    if ticker in universe:
        snap = {"ok": fresh["ok"] and ticker in frames, "reason": fresh["reason"], "source": "vn30"}
        if snap["ok"]:
            df, idx = frames[ticker], frames.get("VNINDEX")
    else:
        ex = dl.extra_freshness(paths, now, ticker)
        snap = {"ok": ex["ok"], "reason": ex["reason"], "source": "extra", "manifest": ex["manifest"]}
        snap_sha = ex["manifest_sha256"]
        if ex["ok"]:
            extra = dl.load_frames(Path(ex["manifest"]))
            df, idx, data_as_of = extra[ticker], extra.get("VNINDEX"), ex["as_of"]
            if fresh["ok"] and fresh["as_of"] != ex["as_of"]:
                snap = {**snap, "ok": False, "reason": "extra snapshot and VN30 snapshot dates differ"}
                df = None
    all_frames = {**frames, **extra}
    journal = load_journal(paths.portfolio, price_lookup=dl.close_lookup(all_frames), checks=checks)
    nav_v, nav_source, _, _ = _portfolio(journal, nav, cfg)
    fields = (compute_bottomfish_fields(df, idx, cfg=bf, sector_mean_ret_10d=_sector_mean(ticker, frames, cfg),
                                        symbol=ticker) if df is not None else None)
    ind = (compute_ticker(df, limit_proxy_pct=cfg["gate"]["limit_proxy_pct"],
                          excursion_lookback=cfg["excursion"]["lookback_entries"]) if df is not None else None)
    news = bf_facts.latest_news(bf_facts.load_bf_news(paths), ticker)
    events = load_events(paths)
    fall = classify_fall(fields, market, news, data_as_of=data_as_of, cfg=bf)
    entry = price or (fields or {}).get("close")
    plan = (bf_plan(entry, nav_v, profile=profile, cfg=bf, fall_type=fall["fall_type"], fields=fields,
                    trade_day=tday, lot=int(cfg["money"]["lot_size"]), amount=amount, tranche=tranche)
            if entry and profile != "P0" else None)
    held = bool(journal and ticker in journal.positions)
    rules = evaluate_bf_gate(ticker=ticker, profile=profile, fields=fields, market=market, fall=fall, snapshot=snap,
                             in_scope=in_scope, held=held, journal=journal, news=news, events=events, plan=plan,
                             price=price, tranche=tranche, trade_day=tday, data_as_of=data_as_of, cfg=bf,
                             swing_effective_loss_pct=effective_loss_pct(cfg))
    structural_ok = all(r["status"] == PASS for r in rules if r["id"] in ("BF-G3", "BF-G4"))
    if ind is not None:
        swing_plan = plan or {"position_value": 0.0}
        rules += [r for r in evaluate_behaviour(ticker=ticker, ind=ind, journal=journal, trade_day=tday, cfg=cfg,
                                                plan=swing_plan, price=price, reason=reason, setups=[], checks=checks)
                  if r["id"] != "B4"]          # BF-B4 replaces swing B4 in this mode
    last_close = {s: float(f["close"].iloc[-1]) for s, f in all_frames.items() if len(f)}
    rules += evaluate_bf_behaviour(ticker=ticker, fields=fields, journal=journal, plan=plan, price=price, why=why,
                                   reason=reason, fall_type=fall["fall_type"], profile=profile, tranche=tranche,
                                   structural_ok=structural_ok, trade_day=tday, cfg=bf, close_lookup=last_close.get)
    pcfg = bf["profiles"][profile]
    out = {"mode": "bottomfish", "ticker": ticker, "profile": profile, "profile_label": pcfg.get("label"),
           "default_note": DEFAULT_NOTE, "edge_note": pcfg.get("edge_note"), "caveat": CAVEAT,
           "checked_at": now.astimezone(timezone.utc).isoformat(), "trade_session": str(tday),
           "expected_session": fresh["expected_session"], "data_as_of": data_as_of, "snapshot": snap,
           "snapshot_manifest_sha256": snap_sha, "price": price, "amount": amount, "reason": reason, "why": why,
           "emotion": emotion, "tranche": tranche, "params_hash": cfg["params_hash"],
           "bf_params_hash": bf["bf_params_hash"], "desk_code_hash": code_hash(),
           "earliest_sell": f"{earliest_sell(tday)} {SELL_FROM}", "horizon_warning": horizon_warning(tday),
           "eod_note": f"EOD data as of {data_as_of or 'n/a'}", "nav_source": nav_source,
           "journal_errors": journal.errors if journal else None, "in_scope": in_scope, "held": held_facts(journal, ticker, tday),
           "fall": fall, "fields": fields, "market": market, "plan": plan, "news_class": (news or {}).get("class"),
           "bf_news_sha": (news or {}).get("source_sha"), "events_sha": bf_facts.file_sha(paths.events),
           "episode_id": episode_id(ticker, fields) if fields else None,
           "checklist_path": str(CHECKLIST).replace("\\", "/"), "setups": []}
    if journal is not None and fields is not None:
        out["attempt_no"] = len(_episode_buys(journal, ticker, out["episode_id"], fields.get("episode_start_date"))) + 1
    else:
        out["attempt_no"] = None
    out["verdict"] = verdict(rules)
    out["rules"] = rules
    out["failed"] = [r["id"] for r in rules if r["status"] == FAIL]
    out["unverified"] = [r["id"] for r in rules if r["status"] == "UNVERIFIED"]
    out["behaviour_block"] = behaviour_block(rules)
    out["recheck_from"] = f"{next_trading_day(tday)} 09:00 (ATO)" if out["behaviour_block"] else None
    out["recheck_conditions"] = _recheck(out["failed"], fall, profile, out["behaviour_block"], tday)
    g12 = next((r for r in rules if r["id"] == "BF-G12"), None)
    out["thesis_present"] = bool(g12 and g12["status"] == PASS)
    out["thesis_required"] = out["verdict"] != "PASS"
    out["acknowledged"] = bool(ack and out["verdict"] == "WARN" and out["thesis_present"])
    out["ack_refused"] = bool(ack and not out["acknowledged"])
    out = jsonable(out)
    if record:
        row = {k: out[k] for k in ("ticker", "trade_session", "checked_at", "verdict", "failed", "unverified",
                                   "behaviour_block", "price", "amount", "reason", "emotion", "data_as_of",
                                   "snapshot_manifest_sha256", "params_hash", "desk_code_hash", "thesis_present",
                                   "acknowledged", "setups", "mode", "profile", "why", "tranche", "episode_id",
                                   "attempt_no", "bf_news_sha", "events_sha", "bf_params_hash")}
        p = out["plan"] or {}
        row.update(source="user_check", bf_type=fall["fall_type"], qty_suggested=p.get("order_qty"),
                   entry_ref=p.get("entry_ref"), invalidation=p.get("invalidation"),
                   catastrophe_stop=p.get("catastrophe_stop"), time_stop_session=p.get("time_stop_session"),
                   panic_low=(fields or {}).get("panic_low"))
        reclog.append(paths.checks_log, jsonable(row))
    return out


# ------------------------------------------------------------------ scan

def fall_type_auto(f: dict, market: dict, cfg: dict) -> str:
    """Price-only proxy for the scan (no news): MARKET / IDIO / NONE."""
    c = cfg["classification"]
    idio = f.get("idio_ret_10d")
    is_market = (isinstance(f.get("idx_ret_5d"), float) and f["idx_ret_5d"] <= c["market_idx_ret_5d_pct"]) or \
        (market.get("state") == "PANIC_BEAR" and (market.get("breadth_pct") or 100) < c["market_breadth_max_pct"])
    if idio is None or f.get("stab_state") == "NONE":
        return "NONE"
    if is_market and idio > c["idio_market_max_pct"]:
        return "MARKET"
    if idio <= c["idio_market_max_pct"]:
        return "IDIO"
    return "NONE"


def scan_rows(paths: DeskPaths, now: datetime) -> dict:
    cfg = load_config(paths.config)
    bf = load_bf_config(paths.bf_config)
    fresh = dl.freshness(paths, now)
    if not fresh["ok"]:
        raise ValueError(f"BF-G0: {fresh['reason']}")
    universe = load_universe_symbols(paths)
    manifest = Path(fresh["manifest"])
    frames = dl.load_frames(manifest)
    market = dl.market_summary(manifest, frames, universe, cfg)
    tday = trade_session(now)
    from .service import _portfolio
    journal = load_journal(paths.portfolio, price_lookup=dl.close_lookup(frames))
    nav_v, nav_source, _, _ = _portfolio(journal, None, cfg)
    rows = []
    for sym in universe:
        f = compute_bottomfish_fields(frames[sym], frames.get("VNINDEX"), cfg=bf,
                                      sector_mean_ret_10d=_sector_mean(sym, frames, cfg), symbol=sym)
        auto = fall_type_auto(f, market, bf)
        plan = bf_plan(f["close"], nav_v, profile="P1", cfg=bf,
                       fall_type="a", fields=f, trade_day=tday, lot=int(cfg["money"]["lot_size"]))
        rules = evaluate_bf_gate(ticker=sym, profile="P1", fields=f, market=market, fall={"fall_type": "a"},
                                 snapshot={"ok": True}, in_scope=True, held=False, journal=None, news=None,
                                 events=None, plan=plan, price=None, tranche=None, trade_day=tday,
                                 data_as_of=fresh["as_of"], cfg=bf)
        structural = [r for r in rules if r["id"] in ("BF-G3", "BF-G4", "BF-G7")]
        rows.append(jsonable({"ticker": sym, "fall_type_auto": auto, "fields": f, "episode_id": episode_id(sym, f),
                              "structural_verdict": verdict(structural),
                              "structural_failed": [r["id"] for r in structural if r["status"] == FAIL],
                              "plan_p1": plan}))
    return {"as_of": fresh["as_of"], "trade_session": str(tday), "market": jsonable(market),
            "snapshot_manifest_sha256": fresh["manifest_sha256"], "nav_source": nav_source,
            "bf_params_hash": bf["bf_params_hash"], "frames": frames, "rows": rows}


def scan_table(scan: dict) -> str:
    fmt = lambda v, f: (f.format(v) if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) else "n/a")  # noqa: E731
    mk = scan["market"]
    lines = [f"# Bảng bắt đáy (VN30) — dữ liệu EOD tới {scan['as_of']} (snapshot sha {scan['snapshot_manifest_sha256'][:12]})", "",
             f"Regime: {mk.get('regime')} | breadth {mk.get('breadth_pct')}% | phiên giao dịch kế: {scan['trade_session']}", "",
             "Chỉ giá: tin tức, lịch sự kiện và trạng thái margin KHÔNG có trong bảng này (BF-G2/G5/G6/G8 cần `check --mode bottomfish`).",
             "fall_type_auto là proxy giá (MARKET/IDIO/NONE), không phải phân loại (a)/(b)/(c). Structural = BF-G3/G4/G7.", "",
             "| Mã | fall_type_auto | stab_state | floor_streak | phiên từ sàn | idio_ret_10d | drawdown_60 | panic_low | capitulation | Structural |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    order = {"STAB2": 0, "STAB1": 1, "WEAK": 2, "FALLING": 3, "LOCKED": 4, "NONE": 5}
    for r in sorted(scan["rows"], key=lambda r: (order.get(r["fields"]["stab_state"], 9), r["ticker"])):
        f = r["fields"]
        lines.append(f"| {r['ticker']} | {r['fall_type_auto']} | {f['stab_state']} | {f['floor_streak']} | "
                     f"{f['sessions_since_floor'] if f['sessions_since_floor'] is not None else '-'} | "
                     f"{fmt(f.get('idio_ret_10d'), '{:.1f}')} | {fmt(f.get('drawdown_60_pct'), '{:.1f}')} | "
                     f"{fmt(f.get('panic_low'), '{:,.0f}')} | {'có' if f.get('capitulation_t') else ''} | "
                     f"{r['structural_verdict']}{' (' + ', '.join(r['structural_failed']) + ')' if r['structural_failed'] else ''} |")
    lines += ["", CAVEAT]
    return "\n".join(lines) + "\n"
