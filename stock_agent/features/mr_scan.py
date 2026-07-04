"""Mean-reversion (bat day) scan service for the dashboard.

Scans the VN100 price history (data/raw/prices_hist) with configs/rules_mr.json through
the production signal engine and returns JSON-able payloads:

  mr_scan_latest()   -> market regime banner + today's BUY_SETUP / WATCH lists
  mr_recent_signals()-> every BUY_SETUP fired in the last N days (what the funnel caught)

Results are cached per (latest data date, rules hash) because a full VN100 pass costs
~15-30s; the dashboard can poll freely.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import compute_rules_hash, load_json
from .indicators import ema, sma
from .signal_engine import prepare_signal_frame, score_precomputed_at
from . import win_probability as wp
from .position_manager import money_cfg, suggest_size

DEFAULT_MIN_WIN_PROB = 0.55

PRICES_DIR = Path("data/raw/prices_hist")
MR_RULES_PATH = Path("configs/rules_mr.json")
CACHE_PATH = Path("data/pipeline/mr_scan_cache.json")


def _load_rules() -> dict:
    return load_json(MR_RULES_PATH)


def _load_frames(prices_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for p in sorted(prices_dir.glob("*.csv")):
        if p.stem == "VNINDEX":
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if len(df) < 90:
            continue
        df["date"] = df["date"].astype(str).str.slice(0, 10)
        frames[p.stem] = df.sort_values("date").reset_index(drop=True)
    return frames


def _market_state(prices_dir: Path, frames: dict[str, pd.DataFrame]) -> dict:
    out = {"state": "UNKNOWN", "index_close": None, "index_ema50": None,
           "breadth_pct": None, "date": None}
    idx_path = prices_dir / "VNINDEX.csv"
    if idx_path.exists():
        idx = pd.read_csv(idx_path)
        idx["date"] = idx["date"].astype(str).str.slice(0, 10)
        idx = idx.sort_values("date").reset_index(drop=True)
        close = float(idx["close"].iloc[-1])
        e50 = float(ema(idx["close"], 50).iloc[-1])
        out.update(index_close=round(close, 2), index_ema50=round(e50, 2),
                   date=str(idx["date"].iloc[-1]))
        above = tot = 0
        for df in frames.values():
            m = sma(df["close"], 20)
            if np.isfinite(m.iloc[-1]):
                tot += 1
                if float(df["close"].iloc[-1]) > float(m.iloc[-1]):
                    above += 1
        breadth = above / tot * 100 if tot else 0.0
        out["breadth_pct"] = round(breadth, 1)
        if close > e50 and breadth >= 40:
            out["state"] = "RISK_ON"
        elif close < e50:
            out["state"] = "PANIC_BEAR"
        else:
            out["state"] = "GRIND"
    return out


def _signal_payload(symbol: str, sig, signal_date: str, win_prob: float | None = None,
                    cfg: dict | None = None) -> dict:
    gates = {e.name: bool(e.passed) for e in sig.evidence}
    confirm = "rev+climax" if (gates.get("Reversal bar") and gates.get("Volume climax")) else \
              ("VSA" if gates.get("VSA stopping volume") else "-")
    rsi = next((e.value for e in sig.evidence if e.name == "Oversold RSI"), None)
    rp = sig.risk_plan
    payload = {
        "symbol": symbol,
        "date": signal_date,
        "decision": sig.decision,
        "close": sig.latest_close,
        "rsi14": rsi,
        "confirm": confirm,
        "win_prob": round(win_prob, 4) if win_prob is not None else None,
        "entry_reference": rp.entry_reference if rp else None,
        "stop_loss": rp.stop_loss if rp else None,
        "take_profit": rp.take_profit_1 if rp else None,
        "reward_risk": rp.reward_risk if rp else None,
        "max_hold_days": rp.holding_period_days if rp else None,
        "missing": [e.name for e in sig.evidence if not e.passed],
    }
    # position sizing (#1): how many shares to buy at this entry/stop
    if cfg and rp and rp.entry_reference and rp.stop_loss:
        size = suggest_size(cfg["account_nav"], rp.entry_reference, rp.stop_loss, cfg)
        payload["size"] = size
    return payload


def _rules_selector(rules: dict):
    """VN30 large-caps are less oversold than the broad VN100; apply a looser MR gate
    (rsi<35, band x1.02 — sweep-validated OOS) to VN30 symbols, strict gate to the rest."""
    from ..config import load_universe
    try:
        vn30 = {s.upper() for s in load_universe()["symbols"]}
    except Exception:
        vn30 = set()
    over = rules.get("vn30_mean_reversion")

    def pick(symbol: str):
        if over and symbol.upper() in vn30:
            r = dict(rules)
            r["mean_reversion"] = {**rules.get("mean_reversion", {}), **over}
            return r, "vn30"
        return rules, "strict"
    return pick


def _compute(recent_days: int, min_win_prob: float) -> dict:
    rules = _load_rules()
    cfg = money_cfg(rules)
    rules_for = _rules_selector(rules)
    frames = _load_frames(PRICES_DIR)
    market = _market_state(PRICES_DIR, frames)

    # win-probability model + context (regime / breadth / index 20d return by date)
    model = wp.WinProbModel.load()
    regime, idx_ret20 = wp._context(PRICES_DIR)
    breadth = wp._breadth_map(frames)

    def win_prob_at(feats: pd.DataFrame, i: int):
        if model is None:
            return None, None
        sd = str(feats["date"].iloc[i])
        fr = wp.feature_row(feats, i, regime.get(sd, 1.0), breadth.get(sd, 0.5), idx_ret20.get(sd))
        if fr is None:
            return None, None
        return model.predict(fr), fr

    buys: list[dict] = []
    watches: list[dict] = []
    recent: list[dict] = []
    prob_buys: list[dict] = []
    cutoff = None
    all_dates = sorted({df["date"].iloc[-1] for df in frames.values()})
    latest_date = all_dates[-1] if all_dates else None
    if latest_date:
        cutoff = str(pd.Timestamp(latest_date) - pd.Timedelta(days=recent_days))[:10]

    for symbol, df in frames.items():
        try:
            srules, gate = rules_for(symbol)
            feats = prepare_signal_frame(df, srules)
            last = len(feats) - 1
            sig = score_precomputed_at(symbol, feats, last, srules)
            signal_date = str(feats["date"].iloc[-1])
            p_last, fr_last = win_prob_at(feats, last)
            if sig.decision == "BUY_SETUP":
                pay = _signal_payload(symbol, sig, signal_date, p_last, cfg); pay["gate"] = gate
                buys.append(pay)
            elif sig.decision == "WATCH":
                pay = _signal_payload(symbol, sig, signal_date, p_last, cfg); pay["gate"] = gate
                watches.append(pay)

            # probability track: loose dip candidate on the latest bar, ranked by P(win)
            if fr_last is not None and wp.is_candidate(fr_last) and p_last is not None \
                    and p_last >= min_win_prob and sig.decision != "BUY_SETUP":
                pay = _signal_payload(symbol, sig, signal_date, p_last, cfg)
                pay["decision"] = "PROB_BUY"; pay["gate"] = gate
                prob_buys.append(pay)

            # recent history: cheap prefilter, scorer decides
            if cutoff:
                rsi = feats["rsi14"]
                pre = (rsi < 38) & (feats["close"] <= feats["bb_lower"] * 1.03) & \
                      (feats["date"] >= cutoff) & (feats.index >= 90)
                for i in np.where(pre.fillna(False))[0]:
                    if int(i) == last:
                        continue  # already reported above
                    s = score_precomputed_at(symbol, feats, int(i), srules)
                    if s.decision == "BUY_SETUP":
                        # NO win_prob for historical signals: the deployed model was trained
                        # on data INCLUDING these dates, so its P(win) here would be in-sample
                        # (leaked). Only live/today signals get an honest ex-ante P(win).
                        pay = _signal_payload(symbol, s, str(feats["date"].iloc[int(i)]), None)
                        pay["gate"] = gate
                        recent.append(pay)
        except Exception:
            continue

    buys.sort(key=lambda x: (-(x.get("win_prob") or 0), -(x.get("reward_risk") or 0)))
    watches.sort(key=lambda x: (x.get("rsi14") or 100))
    prob_buys.sort(key=lambda x: -(x.get("win_prob") or 0))
    recent.sort(key=lambda x: x["date"], reverse=True)
    meta = model.meta if model else {}
    # open positions + live SELL alerts (#3)
    positions, sell_alerts = [], []
    try:
        from .position_manager import PositionStore, check_positions
        positions = check_positions(PositionStore())
        sell_alerts = [p for p in positions if p.get("live_status") == "SELL"]
    except Exception:
        pass
    return {
        "mode": "mean_reversion_hybrid",
        "rules_hash": compute_rules_hash(rules),
        "data_date": market.get("date"),
        "symbols_scanned": len(frames),
        "min_win_prob": min_win_prob,
        "money": {"account_nav": cfg["account_nav"], "risk_per_trade_pct": cfg["risk_per_trade_pct"],
                  "max_positions": cfg["max_positions"]},
        "model": {"available": model is not None, **({k: meta.get(k) for k in ("calib_auc", "base_win_rate", "trained_at", "n_candidates")} if model else {})},
        "market": market,
        "buys": buys,
        "watches": watches,
        "prob_buys": prob_buys,
        "recent_signals": recent,
        "positions": positions,
        "sell_alerts": sell_alerts,
    }


def mr_scan(recent_days: int = 120, force: bool = False,
           min_win_prob: float = DEFAULT_MIN_WIN_PROB) -> dict:
    """Cached MR scan; recomputes when data date, rules, or the threshold change."""
    rules_hash = compute_rules_hash(_load_rules())
    if not force and CACHE_PATH.exists():
        try:
            cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            latest_csv_date = None
            idx = PRICES_DIR / "VNINDEX.csv"
            if idx.exists():
                latest_csv_date = str(pd.read_csv(idx)["date"].astype(str).str.slice(0, 10).max())
            if (cached.get("rules_hash") == rules_hash
                    and cached.get("data_date") == latest_csv_date
                    and cached.get("min_win_prob") == min_win_prob):
                # positions change independently of the (cached) scan — refresh them live
                try:
                    from .position_manager import PositionStore, check_positions
                    cached["positions"] = check_positions(PositionStore())
                    cached["sell_alerts"] = [p for p in cached["positions"] if p.get("live_status") == "SELL"]
                except Exception:
                    pass
                return cached
        except Exception:
            pass
    payload = _compute(recent_days, min_win_prob)
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return payload
