"""Daily EOD update job — designed to run unattended (Windows Task Scheduler, ~17:05).

    python -m stock_agent.pipeline.eod_update

Steps (each fail-soft so one bad provider never kills the run):
  1. Incremental price refresh: append missing recent sessions to every CSV in
     data/raw/prices_hist (VN100 + VNINDEX) via vnstock VCI. Only completed sessions
     are fetched (before 16:00 VN time the end date is yesterday).
  2. Foreign/prop flow harvest: shared dated v2 collector, raw hashes and availability.
  3. Undated price-board room snapshots are disabled pending source session proof.
  4. Rebuild the dashboard mean-reversion scan cache so the web UI opens fresh.

All output goes to stdout; redirect to data/pipeline/eod_update.log in the task.
"""
from __future__ import annotations

import io
import json
import os
import time
from contextlib import redirect_stdout
from datetime import date, datetime
from pathlib import Path
from ..data.exchange_calendar import completed_session_date
from ..data.eod import completed_bars

PRICES_DIR = Path("data/raw/prices_hist")


def _symbols() -> list[str]:
    return sorted(p.stem for p in PRICES_DIR.glob("*.csv") if p.stem != "VNINDEX")


def _fetch_end_date() -> date:
    """Use the same Vietnam-time completed-session boundary as every EOD scan."""
    return completed_session_date()


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
        # Re-fetch the tail even if its date already exists: it may be an old
        # intraday snapshot. Do not assume a date match proves a completed bar.
        start = min(last, end)
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
                new = completed_bars(new, end)
                if new.empty:
                    skipped += 1
                    got = True
                    break
                merged = pd.concat([df, new], ignore_index=True).drop_duplicates(subset=["date"], keep="last").sort_values("date")
                # Atomic write: a killed/slept process mid-write must not truncate the CSV
                # (a partial history would make the next run backfill from a wrong last date).
                tmp = path.with_suffix(".csv.tmp")
                merged.to_csv(tmp, index=False)
                os.replace(tmp, path)
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
def harvest_foreign() -> dict:
    """Shared v2 collector; never append ambiguous legacy JSONL."""
    from .foreign_refresh import collection_symbols
    from ..data.foreign_flows import collect_flows
    return collect_flows(collection_symbols())


def snapshot_room() -> dict:
    """Undated price-board room snapshots are disabled pending source session proof."""
    return {"status": "disabled", "reason": "price-board session provenance unavailable"}


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
        try:
            from ..features.swing_scan import swing_scan
            swing_scan(force=True)   # warm the RSI2 swing cache too
        except Exception:
            pass
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
    print("[1/7] incremental price refresh...", flush=True)
    summary["prices"] = refresh_prices()
    print("[2/7] foreign/prop flow harvest...", flush=True)
    summary["foreign"] = harvest_foreign()
    print("[3/7] room snapshot...", flush=True)
    summary["snapshot"] = snapshot_room()
    print("[4/7] rebuild MR (bat day) scan cache...", flush=True)
    summary["mr_scan"] = rebuild_mr_cache()
    print("[5/7] rebuild momentum (CORE) scan cache...", flush=True)
    summary["momentum"] = rebuild_momentum_cache()
    print("[6/7] check open positions for SELL alerts...", flush=True)
    summary["positions"] = check_sell_alerts()
    print("[7/7] append today's picks to the forward-test ledger...", flush=True)
    summary["forward_test"] = log_forward_test()
    print(f"=== DONE in {(time.time()-t0)/60:.1f} min: {json.dumps(summary, ensure_ascii=False)}", flush=True)


def log_forward_test() -> dict:
    """Persist today's MR + momentum picks to the scoreable forward-test ledger."""
    try:
        from ..features.mr_scan import mr_scan
        from ..features.momentum_scan import momentum_scan
        from .forward_test import log_recommendations
        # read the caches just rebuilt above (no force -> uses fresh cache)
        return log_recommendations(mr_scan(), momentum_scan())
    except Exception as exc:
        return {"status": "error", "error": repr(exc)[:120]}


if __name__ == "__main__":
    main()
