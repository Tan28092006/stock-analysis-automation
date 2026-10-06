"""Portfolio review of `swing_desk/portfolio.txt` (research doc §11). Descriptive only.

Privacy: the report never contains raw journal lines or notes, and shows absolute NAV/VND
amounts only with `show_nav=True` (percentages of NAV are shown).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
import statistics
from pathlib import Path

from . import data as dl
from .behaviour import review_flags
from .calendar_events import earliest_sell, session_distance, trade_session
from .config import DeskPaths, load_config, load_universe_symbols
from .gate import evaluate_gate
from .history import HistoryResolver
from .indicators import compute_ticker, jsonable
from .journal import load_journal
from . import reclog
from .render import pct, vnd
from .service import structural_verdict
from .sizing import effective_loss_pct

MIN_GROUP = 30


def _group(trades: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for t in trades:
        groups[t.get(key) or "unknown"].append(t)
    out = {}
    for name, items in sorted(groups.items()):
        nets = [t["net_pct"] for t in items]
        out[name] = {"n": len(items), "win_rate": sum(n > 0 for n in nets) / len(nets) * 100,
                     "mean_net_pct": statistics.mean(nets), "descriptive_only": len(items) < MIN_GROUP}
    return out


def _excursion(trade: dict, frames: dict) -> dict:
    df = frames.get(trade["ticker"])
    if df is None:
        return {}
    days = df["date"].astype(str)
    held = df[(days >= trade["entry_date"]) & (days <= trade["exit_date"])]
    if held.empty or str(held["date"].iloc[0]) != trade["entry_date"]:
        return {}
    e = trade["entry_avg"]
    return {"mae_pct": (held["low"].min() / e - 1) * 100, "mfe_pct": (held["high"].max() / e - 1) * 100}


def _month(d: str) -> str:
    return d[:7]


def _months(first: str, last: str) -> list[str]:
    y, m = int(first[:4]), int(first[5:7])
    out = []
    while f"{y:04d}-{m:02d}" <= last:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def activity(journal) -> dict:
    """Monthly flat-to-long entries, add-on buys, distinct entry days, zero-new-entry months."""
    held = defaultdict(int)
    stats = defaultdict(lambda: {"new_entries": 0, "add_on_buys": 0, "entry_days": set()})
    events = sorted([e for e in journal.events if e.action in ("BUY", "SELL")], key=lambda e: (e.date, e.line))
    for e in events:
        if e.action == "BUY":
            m = stats[_month(e.date)]
            if held[e.ticker] == 0:
                m["new_entries"] += 1
            else:
                m["add_on_buys"] += 1
            m["entry_days"].add(e.date)
            held[e.ticker] += e.qty
        else:
            held[e.ticker] -= e.qty
    if not events:
        return {}
    out = {}
    for month in _months(_month(events[0].date), _month(events[-1].date)):
        s = stats.get(month, {"new_entries": 0, "add_on_buys": 0, "entry_days": set()})
        out[month] = {"new_entries": s["new_entries"], "add_on_buys": s["add_on_buys"],
                      "distinct_entry_days": len(s["entry_days"]), "zero_new_entries": s["new_entries"] == 0}
    first_day = min(e.date for e in events if e.action == "BUY") if any(e.action == "BUY" for e in events) else None
    out["_first_session"] = {"date": first_day,
                             "entries": sum(1 for e in events if e.action == "BUY" and e.date == first_day)}
    return out


def build_review(paths: DeskPaths, *, now: datetime, show_nav: bool = False) -> dict:
    cfg = load_config(paths.config)
    fresh = dl.freshness(paths, now)
    manifest = Path(fresh["manifest"]) if fresh.get("manifest") else None
    frames = dl.load_frames(manifest) if manifest and fresh.get("as_of") else {}
    checks = reclog.read_rows(paths.checks_log)
    journal = load_journal(paths.portfolio, price_lookup=dl.close_lookup(frames) if frames else None, checks=checks)
    if journal is None:
        return {"markdown": "Chưa có swing_desk/portfolio.txt — xem swing_desk/portfolio.example.txt.\n", "empty": True}
    universe = load_universe_symbols(paths)
    nav = journal.nav()
    nav_known = nav is not None
    nav = nav or float(cfg["money"]["account_nav_default"])
    tday = trade_session(now)
    market = dl.market_summary(manifest, frames, universe, cfg) if frames else {"state": "UNKNOWN"}
    eff = effective_loss_pct(cfg) / 100
    stop_pct = cfg["stops"]["user_stop_pct"]

    open_rows = []
    for t, p in sorted(journal.positions.items()):
        row = {"ticker": t, "qty": p["qty"], "avg_cost": p["avg_cost"], "first_date": p["first_date"],
               "weight_pct": p["cost_value"] / nav * 100,
               "sellable": earliest_sell(date.fromisoformat(p["last_date"])) <= tday,
               "sellable_from": str(earliest_sell(date.fromisoformat(p["last_date"]))),
               "sessions_held": session_distance(date.fromisoformat(p["first_date"]),
                                                 date.fromisoformat(fresh["as_of"] or str(tday))),
               "actions": []}
        if t in frames:
            long_df, hist = HistoryResolver(paths, fresh["as_of"]).series(t, frames[t])
            ind = compute_ticker(frames[t], long_df=long_df)
            row["history_source"] = hist["history_source"]
            g = evaluate_gate(ticker=t, ind=ind, market=market, freshness=fresh, universe=universe, cfg=cfg,
                              trade_day=tday, plan=None, portfolio={}, events=None)
            stop = p["avg_cost"] * (1 - stop_pct / 100)
            row.update(close=ind["close"], unrealized_pct=(ind["close"] / p["avg_cost"] - 1) * 100,
                       today_gate=structural_verdict(g["rules"]), stop=stop)
            row["unrealized_R"] = row["unrealized_pct"] / stop_pct
            if ind["close"] <= stop:
                row["actions"].append("SELL: đóng cửa <= stop 3%")
            if (ind.get("sma50") and ind["close"] < ind["sma50"]) or market.get("state") == "PANIC_BEAR":
                row["actions"].append("luận điểm hỏng: C < SMA50 hoặc PANIC_BEAR")
            if row["sessions_held"] >= cfg["setups"]["s1_max_hold_sessions"]:
                row["actions"].append("vượt time-stop 8 phiên")
            if p.get("reason") == "S1" and ind.get("sma5") and ind["close"] > ind["sma5"]:
                row["actions"].append("S1: C > SMA5 -> bán phiên kế")
        else:
            row["actions"].append("không có giá trong snapshot (ngoài VN30?)")
        open_rows.append(row)

    exposure = sum(p["cost_value"] for p in journal.positions.values())
    risk = {"open_risk_pct_nav": exposure * eff / nav * 100, "max_open_risk_pct": cfg["money"]["max_open_risk_pct"],
            "exposure_pct_nav": exposure / nav * 100, "positions": len(journal.positions),
            "bank": len(set(journal.positions) & set(cfg["sectors"]["bank"])),
            "vingroup": len(set(journal.positions) & set(cfg["sectors"]["vingroup"]))}

    closed = journal.closed
    for t in closed:
        t.update(_excursion(t, frames))
        t["R"] = t["net_pct"] / stop_pct
    nets = [t["net_pct"] for t in closed]
    wins = [n for n in nets if n > 0]
    losses = [n for n in nets if n <= 0]
    summary = {"n": len(closed),
               "win_rate": len(wins) / len(nets) * 100 if nets else None,
               "avg_win_pct": statistics.mean(wins) if wins else None,
               "avg_loss_pct": statistics.mean(losses) if losses else None,
               "expectancy_pct": statistics.mean(nets) if nets else None,
               "expectancy_stress_pct": statistics.mean(t["net_pct_stress"] for t in closed) if closed else None,
               "mean_R": statistics.mean(t["R"] for t in closed) if closed else None,
               "pnl_vnd": sum(t["pnl_vnd"] for t in closed)}
    flags = review_flags(journal, frames, cfg)
    act = activity(journal)
    month_pnl = defaultdict(float)
    for t in closed:
        month_pnl[_month(t["exit_date"])] += t["pnl_vnd"]
    after_loss = {}
    for m, v in sorted(month_pnl.items()):
        if v < 0:
            nxt = _months(m, "9999-12")[1]
            after_loss[nxt] = (act.get(nxt) or {}).get("new_entries", 0)
    monthly_cost = defaultdict(float)
    for e in journal.events:
        if e.action in ("BUY", "SELL"):
            monthly_cost[_month(e.date)] += e.qty * e.price * (journal.buy_cost if e.action == "BUY" else journal.sell_cost)

    report = {"as_of": fresh.get("as_of"), "data_fresh": fresh["ok"], "nav_known": nav_known,
              "open_positions": open_rows, "risk": risk, "closed_summary": summary,
              "by_gate": _group(closed, "gate"), "by_emotion": _group(closed, "emotion"),
              "flags": flags, "activity": act, "entries_after_losing_month": after_loss,
              "monthly_cost_pct_nav": {m: v / nav * 100 for m, v in sorted(monthly_cost.items())},
              "journal_errors": [{"line": e["line"], "error": e["error"]} for e in journal.errors],
              "activity_floor_note": "Activity floor (>= 4/month) applies to desk ideas in the prospective log, not to your trades."}
    report["markdown"] = _markdown(report, nav if show_nav else None, nav_known, show_nav)
    if any(e.reason == "bottomfish" for e in journal.buys()):
        report["bottomfish"] = bf_review(paths, journal, frames, fresh, now)
        report["markdown"] += _bf_markdown(report["bottomfish"])
    return jsonable(report) | {"markdown": report["markdown"]}


def _invalid_level(text: str | None):
    """`invalid=close<24100` -> 24100.0 (other forms are kept as text only)."""
    if not text or "<" not in text:
        return None
    try:
        return float(text.split("<", 1)[1].strip().replace(",", "").replace("_", ""))
    except ValueError:
        return None


def bf_review(paths: DeskPaths, journal, frames: dict, fresh: dict, now: datetime) -> dict:
    """Bottom-fish section: episodes, attempts, BF-B10 (held past time stop / below invalidation), floor-lock days."""
    from .bottomfish import bf_buys
    as_of = fresh.get("as_of")
    episodes = defaultdict(list)
    for e in bf_buys(journal):
        episodes[(e.ticker, e.keys.get("episode", "").upper() or f"{e.ticker}-{e.date}")].append(e)
    ep_rows = [{"ticker": t, "episode": ep, "attempts": len(v), "first": min(x.date for x in v),
                "tranches": [x.keys.get("tranche") for x in v if x.keys.get("tranche")]} for (t, ep), v in sorted(episodes.items())]
    open_rows = []
    for t, p in sorted(journal.positions.items()):
        if p.get("reason") != "bottomfish":
            continue
        df = frames.get(t)
        if df is None and as_of:
            ex = dl.extra_freshness(paths, now, t)
            if ex["ok"]:
                df = dl.load_frames(Path(ex["manifest"])).get(t)
        theses = journal.theses(t)
        keys = theses[-1].keys if theses else {}
        inv, stop_day = _invalid_level(keys.get("invalid")), keys.get("time_stop")
        sell_from = str(earliest_sell(date.fromisoformat(p["last_date"])))
        row = {"ticker": t, "qty": p["qty"], "avg_cost": p["avg_cost"], "first_date": p["first_date"],
               "sellable_from": sell_from, "invalid": keys.get("invalid"), "time_stop": stop_day,
               "floor_lock_days": None, "b10": [], "price_data": df is not None}
        if df is not None:
            days = [str(d)[:10] for d in df["date"]]
            c = df["close"].astype(float).tolist()
            held = [k for k, d in enumerate(days) if d >= p["first_date"]]
            row["floor_lock_days"] = sum(1 for k in held if k and (c[k] / c[k - 1] - 1) * 100 <= -6.5)
            sellable = [days[k] for k in held if days[k] >= sell_from]
            if stop_day:
                late = [d for d in sellable if d > stop_day]
                if late and as_of and as_of > late[0]:
                    row["b10"].append(f"held past time stop {stop_day} (sellable since {late[0]})")
            if inv is not None:
                below = [days[k] for k in held if days[k] >= sell_from and c[k] < inv]
                if below and as_of and as_of > below[0]:
                    row["b10"].append(f"close {below[0]} below invalidation {inv:,.0f} and still held")
        open_rows.append(row)
    closed = [t for t in journal.closed if t.get("reason") == "bottomfish"]
    nets = [t["net_pct"] for t in closed]
    return {"episodes": ep_rows, "open": open_rows,
            "closed": {"n": len(closed), "win_rate": sum(n > 0 for n in nets) / len(nets) * 100 if nets else None,
                       "mean_net_pct": statistics.mean(nets) if nets else None,
                       "descriptive_only": len(closed) < MIN_GROUP},
            "b10_count": sum(1 for r in open_rows if r["b10"])}


def _bf_markdown(bf: dict) -> str:
    L = ["", "## Bắt đáy (BF)", "",
         "Chỉ tư vấn; bắt đáy chưa có bằng chứng edge tiến cứu; loại (b) có bằng chứng trôi giảm.", "",
         "| Mã | Episode | Số lần mua | Tầng | Lần đầu |", "|---|---|---|---|---|"]
    for e in bf["episodes"]:
        L.append(f"| {e['ticker']} | {e['episode']} | {e['attempts']} | {', '.join(e['tranches']) or '-'} | {e['first']} |")
    L.append("")
    for r in bf["open"]:
        L.append(f"- {r['ticker']}: {vnd(r['qty'])} cp, giá vốn {vnd(r['avg_cost'])}, bán được từ {r['sellable_from']}, "
                 f"invalid={r['invalid'] or 'chưa ghi'}, time_stop={r['time_stop'] or 'chưa ghi'}, "
                 f"phiên sàn khi giữ {r['floor_lock_days'] if r['floor_lock_days'] is not None else 'n/a'}"
                 + (f" — BF-B10: {'; '.join(r['b10'])}" if r["b10"] else ""))
    c = bf["closed"]
    L.append("")
    L.append(f"Lệnh BF đã đóng: {c['n']}" + (f"; win {pct(c['win_rate'])}; net TB {pct(c['mean_net_pct'], 2)} (mô tả)"
                                              if c["n"] else ""))
    L.append("Desk không khuyến nghị bán hay giữ; BF-B10 chỉ ghi nhận việc giữ quá kế hoạch đã viết.")
    return "\n".join(L) + "\n"


def _markdown(r: dict, nav, nav_known: bool, show_nav: bool) -> str:
    L = [f"# Review danh mục — dữ liệu tới {r['as_of'] or 'n/a'}" + ("" if r["data_fresh"] else " (CẢNH BÁO: dữ liệu không tươi)"), ""]
    if show_nav:
        L.append(f"NAV: {vnd(nav)} VND" + ("" if nav_known else " (mặc định, journal chưa có NAV/CASH)"))
    else:
        L.append("NAV ẩn (dùng --show-nav để hiện). Tỷ lệ % NAV vẫn hiển thị.")
    L += ["", "## Vị thế mở", ""]
    if not r["open_positions"]:
        L.append("Không có vị thế mở.")
    for p in r["open_positions"]:
        L.append(f"- {p['ticker']}: {vnd(p['qty'])} cp, giá vốn {vnd(p['avg_cost'])}, "
                 f"lãi/lỗ {pct(p.get('unrealized_pct'))} ({p.get('unrealized_R', 0):.2f}R), "
                 f"giữ {p['sessions_held']} phiên, {'đã bán được' if p['sellable'] else 'chưa bán được tới ' + p['sellable_from']}, "
                 f"gate hôm nay {p.get('today_gate', 'n/a')}, tỷ trọng {pct(p['weight_pct'])}"
                 + (f" — {'; '.join(p['actions'])}" if p["actions"] else ""))
    k = r["risk"]
    L += ["", "## Rủi ro danh mục", "",
          f"Rủi ro mở (lỗ hiệu dụng) {pct(k['open_risk_pct_nav'], 2)} NAV / trần {k['max_open_risk_pct']}%; "
          f"exposure {pct(k['exposure_pct_nav'])}; {k['positions']} vị thế; ngân hàng {k['bank']}; Vingroup {k['vingroup']}."]
    s = r["closed_summary"]
    L += ["", "## Lệnh đã đóng", ""]
    if s["n"]:
        L.append(f"{s['n']} lệnh; win {pct(s['win_rate'])}; lãi TB khi thắng {pct(s['avg_win_pct'], 2)}; lỗ TB khi thua "
                 f"{pct(s['avg_loss_pct'], 2)}; kỳ vọng {pct(s['expectancy_pct'], 2)}/lệnh (phí 0,4%), "
                 f"{pct(s['expectancy_stress_pct'], 2)} (phí 0,6%); R TB {s['mean_R']:.2f}"
                 + (f"; P&L {vnd(s['pnl_vnd'])} VND" if show_nav else ""))
    else:
        L.append("Chưa có lệnh đóng.")
    for title, key in (("Theo gate", "by_gate"), ("Theo cảm xúc", "by_emotion")):
        L += ["", f"## {title} (mô tả; cần >= {MIN_GROUP} lệnh/nhóm mới so sánh)", "", "| Nhóm | n | Win % | Net TB % |", "|---|---|---|---|"]
        for name, g in r[key].items():
            L.append(f"| {name} | {g['n']} | {g['win_rate']:.0f} | {g['mean_net_pct']:.2f} |")
    f = r["flags"]
    L += ["", "## Hành vi", "",
          f"- B8 không tôn trọng stop: {f['B8']['count']} lệnh",
          f"- B9 disposition: {f['B9']['flag'] if f['B9']['flag'] is not None else 'chưa đủ dữ liệu'} ({f['B9']['detail']})",
          f"- Lệnh mới tháng sau tháng lỗ: {r['entries_after_losing_month'] or 'không có'}",
          "- Phí giao dịch % NAV theo tháng: " + (", ".join(f"{m}: {v:.2f}%" for m, v in r["monthly_cost_pct_nav"].items()) or "n/a")]
    L += ["", "## Hoạt động (AGENTS.md)", "", "| Tháng | Lệnh mở mới | Mua thêm | Số ngày vào lệnh | Không có lệnh mới |", "|---|---|---|---|---|"]
    for m, a in r["activity"].items():
        if m.startswith("_"):
            continue
        L.append(f"| {m} | {a['new_entries']} | {a['add_on_buys']} | {a['distinct_entry_days']} | {'có' if a['zero_new_entries'] else ''} |")
    first = r["activity"].get("_first_session")
    if first and first["date"]:
        L.append(f"\nPhiên đầu tiên trong journal ({first['date']}): {first['entries']} lệnh mua (xây danh mục ban đầu).")
    if r["journal_errors"]:
        L += ["", "## Lỗi journal", ""] + [f"- dòng {e['line']}: {e['error']}" for e in r["journal_errors"]]
    L += ["", "Mô tả quá khứ, không phải kiểm định chiến lược. Desk chỉ tư vấn."]
    return "\n".join(L) + "\n"
