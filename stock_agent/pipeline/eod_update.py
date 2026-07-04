"""Daily EOD update job — designed to run unattended (Windows Task Scheduler, ~17:05).

    python -m stock_agent.pipeline.eod_update

Steps (each fail-soft so one bad provider never kills the run):
  1. Incremental price refresh: append missing recent sessions to every CSV in
     data/raw/prices_hist (VN100 + VNINDEX) via vnstock VCI. Only completed sessions
     are fetched (before 16:00 VN time the end date is yesterday).
  2. Foreign/prop flow harvest: Vietstock per-symbol chart endpoints give the last
     ~20 daily sessions; run daily, the JSONL accumulates gap-free history forward.
  3. Foreign room snapshot via vnstock price_board (adds room/ownership columns).
  4. Rebuild the dashboard mean-reversion scan cache so the web UI opens fresh.

All output goes to stdout; redirect to data/pipeline/eod_update.log in the task.
"""
from __future__ import annotations

import io
import json
import random
import re
import time
import urllib.parse
import urllib.request
import http.cookiejar
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta
from pathlib import Path

PRICES_DIR = Path("data/raw/prices_hist")
FOREIGN_DIR = Path("data/raw/foreign")

VIETSTOCK_REFERER = "https://finance.vietstock.vn/MBB/thong-ke-giao-dich.htm"
VIETSTOCK_ENDPOINTS = {
    "ndtnn_chart": ("https://finance.vietstock.vn/data/KQGDGiaoDichNDTNNChartByStock",
                    {"type": "1", "isRealTime": "false"}),
    "tudoanh_chart": ("https://finance.vietstock.vn/data/KQGDGiaoDichTuDoanhChartByStock",
                      {"type": "1"}),
}


def _symbols() -> list[str]:
    return sorted(p.stem for p in PRICES_DIR.glob("*.csv") if p.stem != "VNINDEX")


def _fetch_end_date() -> date:
    """Only fetch completed sessions: before 16:00 local, stop at yesterday."""
    now = datetime.now()
    end = now.date()
    if now.hour < 16:
        end = end - timedelta(days=1)
    return end


# ---------------------------------------------------------------- 1. prices
def refresh_prices() -> dict:
    import pandas as pd

    end = _fetch_end_date()
    updated = skipped = failed = 0
    try:
        with redirect_stdout(io.StringIO()):
            from vnstock import Vnstock
    except Exception as exc:
        return {"status": "error", "error": f"vnstock import failed: {exc}"}

    for sym in _symbols() + ["VNINDEX"]:
        path = PRICES_DIR / f"{sym}.csv"
        try:
            df = pd.read_csv(path)
            df["date"] = df["date"].astype(str).str.slice(0, 10)
            last = date.fromisoformat(str(df["date"].max()))
        except Exception as exc:
            failed += 1
            print(f"  [prices] {sym} read FAIL {repr(exc)[:80]}", flush=True)
            continue
        if last >= end:
            skipped += 1
            continue
        start = last + timedelta(days=1)
        got = False
        for attempt in range(4):
            try:
                time.sleep(3.6)  # stay under guest 20 req/min with margin
                with redirect_stdout(io.StringIO()):
                    new = Vnstock().stock(symbol=sym, source="VCI").quote.history(
                        start=str(start), end=str(end), interval="1D")
                if new is None or len(new) == 0:
                    skipped += 1
                    got = True
                    break
                new = new.rename(columns={"time": "date"})[["date", "open", "high", "low", "close", "volume"]].copy()
                if sym != "VNINDEX":
                    for c in ["open", "high", "low", "close"]:
                        new[c] = new[c] * 1000.0
                new["date"] = new["date"].astype(str).str.slice(0, 10)
                merged = pd.concat([df, new], ignore_index=True).drop_duplicates(subset=["date"], keep="last").sort_values("date")
                merged.to_csv(path, index=False)
                updated += 1
                got = True
                print(f"  [prices] {sym}: +{len(new)} -> {merged['date'].max()}", flush=True)
                break
            except KeyboardInterrupt:
                raise
            except BaseException as exc:   # vnstock rate limiter raises SystemExit
                if attempt == 3:
                    print(f"  [prices] {sym} FAIL {repr(exc)[:80]}", flush=True)
                else:
                    time.sleep(20)         # rate-limit backoff, then retry
        if not got:
            failed += 1
    return {"updated": updated, "skipped_fresh": skipped, "failed": failed, "end": str(end)}


# ------------------------------------------------------- 2. foreign harvest
class _VietstockSession:
    def __init__(self):
        self.cj = http.cookiejar.CookieJar()
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cj))
        self.op.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")]
        html = self.op.open(VIETSTOCK_REFERER, timeout=30).read().decode("utf-8", "replace")
        self.token = re.search(
            r"__CHART_AjaxAntiForgeryForm[^>]*>.*?name=__RequestVerificationToken[^>]*value=([^ >]+)",
            html, re.S).group(1)

    def post(self, ep: str, params: dict):
        d = {"__RequestVerificationToken": self.token}
        d.update(params)
        req = urllib.request.Request(ep, data=urllib.parse.urlencode(d).encode(),
            headers={"X-Requested-With": "XMLHttpRequest",
                     "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                     "Referer": VIETSTOCK_REFERER, "User-Agent": "Mozilla/5.0"})
        return json.loads(self.op.open(req, timeout=30).read().decode("utf-8-sig", "replace"))


def _ms2date(s):
    m = re.search(r"(-?\d+)", s or "")
    if not m:
        return None
    ms = int(m.group(1))
    if ms < 0:
        return None
    return (datetime(1970, 1, 1) + timedelta(milliseconds=ms)).date()


def harvest_foreign() -> dict:
    FOREIGN_DIR.mkdir(parents=True, exist_ok=True)
    symbols = _symbols()
    try:
        ses = _VietstockSession()
    except Exception as exc:
        return {"status": "error", "error": f"vietstock session failed: {repr(exc)[:120]}"}
    added_total = {}
    for kind, (ep, base) in VIETSTOCK_ENDPOINTS.items():
        out_path = FOREIGN_DIR / f"{kind}.jsonl"
        seen = set()
        if out_path.exists():
            with out_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        seen.add((r["date"], r["symbol"]))
                    except Exception:
                        pass
        added = 0
        for sym in symbols:
            for attempt in range(3):
                try:
                    j = ses.post(ep, {**base, "stockCode": sym})
                    series = j[1] if isinstance(j, list) and len(j) > 1 else []
                    with out_path.open("a", encoding="utf-8") as f:
                        for r in series:
                            d = _ms2date(r.get("TradingDate"))
                            if d is None or (str(d), sym) in seen:
                                continue
                            f.write(json.dumps({
                                "date": str(d), "symbol": sym,
                                "buy_vol": r.get("BuyVol"), "buy_val": r.get("BuyVal"),
                                "sell_vol": r.get("SellVol"), "sell_val": r.get("SellVal"),
                            }, ensure_ascii=False) + "\n")
                            seen.add((str(d), sym))
                            added += 1
                    break
                except Exception:
                    time.sleep(4)
                    try:
                        ses = _VietstockSession()
                    except Exception:
                        pass
            time.sleep(random.uniform(0.4, 0.8))
        added_total[kind] = added
        print(f"  [foreign] {kind}: +{added} new symbol-days", flush=True)
    return added_total


# ------------------------------------------------------------- 3. snapshot
def snapshot_room() -> dict:
    try:
        from ..data.foreign_flows import snapshot_today
        n = snapshot_today(_symbols())
        return {"rows": n}
    except Exception as exc:
        return {"status": "error", "error": repr(exc)[:120]}


# ------------------------------------------------------------ 4. mr cache
def rebuild_mr_cache() -> dict:
    try:
        from ..features.mr_scan import mr_scan
        payload = mr_scan(force=True)
        return {"data_date": payload.get("data_date"),
                "buys": len(payload.get("buys", [])),
                "watches": len(payload.get("watches", [])),
                "prob_buys": len(payload.get("prob_buys", []))}
    except Exception as exc:
        return {"status": "error", "error": repr(exc)[:120]}


def rebuild_momentum_cache() -> dict:
    """Refresh the CORE momentum-rotation scan cache (bull-catcher panel)."""
    try:
        from ..features.momentum_scan import momentum_scan
        payload = momentum_scan(force=True)
        for p in payload.get("sell_alerts", []):
            print(f"  *** MOMENTUM SELL: {p['symbol']} — {p['sell_reason']} "
                  f"(now {p.get('current_price')}, {p.get('unrealized_pct')}%) ***", flush=True)
        return {"active": payload.get("active"), "picks": len(payload.get("picks", [])),
                "regime": payload.get("market", {}).get("state"),
                "momentum_sell_alerts": len(payload.get("sell_alerts", []))}
    except Exception as exc:
        return {"status": "error", "error": repr(exc)[:120]}


def check_sell_alerts() -> dict:
    """Evaluate open MR positions and print/persist any SELL alerts."""
    try:
        from ..features.position_manager import PositionStore, check_positions
        positions = check_positions(PositionStore())
        sells = [p for p in positions if p.get("live_status") == "SELL"]
        for p in sells:
            print(f"  *** SELL ALERT: {p['symbol']} — {p['sell_reason']} "
                  f"(entry {p['entry_price']}, now {p.get('current_price')}, "
                  f"{p.get('unrealized_pct')}%, held {p.get('held_days')}d) ***", flush=True)
        return {"open": len(positions), "sell_alerts": len(sells),
                "symbols": [p["symbol"] for p in sells]}
    except Exception as exc:
        return {"status": "error", "error": repr(exc)[:120]}


def main() -> None:
    t0 = time.time()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"=== EOD update {stamp} ===", flush=True)
    if date.today().weekday() >= 5:
        print("weekend — prices/foreign unchanged, refreshing caches only", flush=True)
    summary = {}
    print("[1/4] incremental price refresh...", flush=True)
    summary["prices"] = refresh_prices()
    print("[2/4] foreign/prop flow harvest...", flush=True)
    summary["foreign"] = harvest_foreign()
    print("[3/4] room snapshot...", flush=True)
    summary["snapshot"] = snapshot_room()
    print("[4/6] rebuild MR (bat day) scan cache...", flush=True)
    summary["mr_scan"] = rebuild_mr_cache()
    print("[5/6] rebuild momentum (CORE) scan cache...", flush=True)
    summary["momentum"] = rebuild_momentum_cache()
    print("[6/6] check open positions for SELL alerts...", flush=True)
    summary["positions"] = check_sell_alerts()
    print(f"=== DONE in {(time.time()-t0)/60:.1f} min: {json.dumps(summary, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
