from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..schemas import RiskPlan, RuleEvidence
from .feature_store import build_feature_snapshot
from .indicators import add_indicators

SIGNAL_FEATURE_COLUMNS = {
    "return_1d",
    "ema9",
    "ema21",
    "ema50",
    "rsi14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "atr",
    "tenkan",
    "kijun",
    "senkou_a",
    "senkou_b",
    "vwap20",
    "obv_slope_5",
    "adl_slope_5",
    "adx14",
    "plus_di14",
    "minus_di14",
    "volume_ratio_20",
    "high_20",
    "sqz_on",
    "bb_kc_ratio",
    "spread_ratio",
    "vsa_stopping_volume",
    "vsa_no_demand",
    "vsa_no_supply",
}


@dataclass
class SignalOutput:
    decision: str
    score: float
    confidence_delta: float
    latest_close: float
    latest_date: str
    risk_plan: RiskPlan | None
    evidence: list[RuleEvidence]
    warnings: list[str]
    features: dict


def _ev(symbol: str, name: str, passed: bool, points: float, detail: str, value=None) -> RuleEvidence:
    safe_name = name.lower().replace(" ", "_").replace("/", "_")
    return RuleEvidence(
        evidence_id=f"{symbol}:{safe_name}",
        name=name,
        passed=passed,
        points=points if passed else 0.0,
        detail=detail,
        value=value,
    )


def _threshold(thresholds: dict, key: str, default):
    return thresholds.get(key, default)


def has_signal_features(df: pd.DataFrame) -> bool:
    return SIGNAL_FEATURE_COLUMNS.issubset(set(df.columns))


def prepare_signal_frame(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    risk_rules = rules["risk"]
    if has_signal_features(df):
        return df
    return add_indicators(df, atr_period=int(risk_rules["atr_period"]))


def precompute_signal_frames(symbol_frames: dict[str, pd.DataFrame], rules: dict) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol, frame in symbol_frames.items():
        if frame is None or frame.empty:
            continue
        frames[symbol.upper()] = prepare_signal_frame(frame.sort_values("date").reset_index(drop=True).copy(), rules)
        
    try:
        from .feature_engineering_v2 import add_cross_sectional_features, add_regime_features
        cs_frames = add_cross_sectional_features(frames)
        for symbol, frame in cs_frames.items():
            frames[symbol] = add_regime_features(frame)
    except Exception:
        pass
        
    return frames


def score_precomputed_symbol(symbol: str, features_df: pd.DataFrame, rules: dict) -> SignalOutput:
    if not has_signal_features(features_df):
        raise ValueError("score_precomputed_symbol requires a frame prepared by prepare_signal_frame")
    return _score_symbol_from_features(symbol, features_df, rules)


def score_precomputed_at(symbol: str, features_df: pd.DataFrame, idx: int, rules: dict) -> SignalOutput:
    if not has_signal_features(features_df):
        raise ValueError("score_precomputed_at requires a frame prepared by prepare_signal_frame")
    return _score_symbol_from_features(symbol, features_df, rules, idx=idx, include_features=True)


def score_symbol(symbol: str, df: pd.DataFrame, rules: dict) -> SignalOutput:
    return _score_symbol_from_features(symbol, prepare_signal_frame(df, rules), rules)


def _score_symbol_from_features(
    symbol: str,
    features_df: pd.DataFrame,
    rules: dict,
    idx: int | None = None,
    include_features: bool = True,
) -> SignalOutput:
    if str(rules.get("strategy_mode", "momentum")).lower() == "mean_reversion":
        return _score_mean_reversion_from_features(
            symbol, features_df, rules, idx=idx, include_features=include_features
        )
    thresholds = rules["signal_thresholds"]
    risk_rules = rules["risk"]
    row_idx = len(features_df) - 1 if idx is None else idx
    latest = features_df.iloc[row_idx]
    prev = features_df.iloc[row_idx - 1]
    score = 0.0
    warnings: list[str] = []
    evidence: list[RuleEvidence] = []

    close = float(latest["close"])
    latest_date = str(latest["date"])
    atr_value = float(latest.get("atr") or 0)
    kijun = float(latest.get("kijun") or close)
    support_stop = kijun * (1 - risk_rules["stop_support_buffer_pct"] / 100.0)
    atr_stop = close - risk_rules["stop_atr_multiple"] * atr_value if atr_value > 0 else close * 0.97
    # Chọn khoảng dừng rộng hơn giữa Support Stop và ATR Stop, và đảm bảo cách Close tối thiểu 0.5%
    stop_loss = min(support_stop, atr_stop)
    stop_loss = min(stop_loss, close * 0.995)
    if stop_loss >= close:
        stop_loss = close * 0.97
    risk_per_share = close - stop_loss
    
    # Tính mục tiêu chốt lời linh hoạt dựa trên tỷ lệ R:R kỳ vọng (target_rr) thay vì bị trần cứng
    take_profit_1 = max(close + risk_per_share * (risk_rules["target_rr"] * 0.5), close * 1.005)
    take_profit_2 = max(close + risk_per_share * risk_rules["target_rr"], take_profit_1)
    stop_loss_pct = risk_per_share / close * 100
    reward_risk = (take_profit_1 - close) / max(risk_per_share, 1e-9)
    risk_plan = RiskPlan(
        entry_reference=round(close, 4),
        stop_loss=round(stop_loss, 4),
        take_profit_1=round(take_profit_1, 4),
        take_profit_2=round(take_profit_2, 4),
        stop_loss_pct=round(stop_loss_pct, 3),
        reward_risk=round(reward_risk, 3),
        holding_period_days=rules["time_horizon_days"],
    )

    # Đọc trọng số điểm của các luật giao dịch từ cấu hình động (hoặc lấy giá trị mặc định)
    pts = lambda k, d: float(rules.get("rule_points", {}).get(k, d))
    
    trend_pass = close > float(latest["ema21"]) and float(latest["ema9"]) >= float(latest["ema21"])
    evidence.append(_ev(symbol, "EMA trend", trend_pass, pts("ema_trend", 14), "close > EMA21 and EMA9 >= EMA21"))

    medium_trend = float(latest["ema21"]) >= float(latest["ema50"])
    evidence.append(_ev(symbol, "Medium trend", medium_trend, pts("medium_trend", 8), "EMA21 >= EMA50"))

    sma_trend = float(latest.get("sma5") or close) >= float(latest.get("sma20") or close)
    evidence.append(_ev(symbol, "SMA short trend", sma_trend, pts("sma_short_trend", 5), "SMA5 >= SMA20"))

    macd_pass = float(latest["macd_hist"]) > 0 and float(latest["macd"]) >= float(latest["macd_signal"])
    macd_cross = float(prev["macd_hist"]) <= 0 < float(latest["macd_hist"])
    evidence.append(
        _ev(
            symbol,
            "MACD momentum",
            macd_pass,
            pts("macd_momentum", 12) + (pts("macd_cross_extra", 6) if macd_cross else 0),
            "MACD histogram positive; recent cross earns extra points",
            round(float(latest["macd_hist"]), 4),
        )
    )

    rsi_value = float(latest["rsi14"])
    rsi_pass = thresholds["rsi_min"] <= rsi_value <= thresholds["rsi_max"] and rsi_value >= float(prev["rsi14"]) - 1
    evidence.append(_ev(symbol, "RSI T2 zone", rsi_pass, pts("rsi_t2_zone", 10), "RSI is in T+2 momentum zone", round(rsi_value, 2)))

    volume_ratio = float(latest["volume_ratio_20"]) if pd.notna(latest["volume_ratio_20"]) else 0.0
    volume_pass = volume_ratio >= thresholds["volume_surge_min"]
    evidence.append(
        _ev(symbol, "Volume confirmation", volume_pass, pts("volume_confirmation", 14), "volume / avg20 exceeds threshold", round(volume_ratio, 3))
    )

    cloud_top = max(float(latest.get("senkou_a") or close), float(latest.get("senkou_b") or close))
    ichimoku_pass = close > kijun and float(latest["tenkan"]) >= kijun
    evidence.append(_ev(symbol, "Ichimoku base", ichimoku_pass, pts("ichimoku_base", 12), "close > Kijun and Tenkan >= Kijun"))
    cloud_pass = close >= cloud_top
    evidence.append(_ev(symbol, "Cloud position", cloud_pass, pts("cloud_position", 8), "close is above or inside upper cloud boundary"))

    vwap_value = float(latest.get("vwap20") or close)
    vwap_pass = close >= vwap_value
    evidence.append(_ev(symbol, "VWAP position", vwap_pass, pts("vwap_position", 5), "close is above rolling VWAP20 proxy", round(vwap_value, 4)))

    adx_value = float(latest.get("adx14") or 0.0)
    plus_di = float(latest.get("plus_di14") or 0.0)
    minus_di = float(latest.get("minus_di14") or 0.0)
    adx_pass = adx_value >= float(_threshold(thresholds, "adx_min", 18)) and plus_di >= minus_di
    evidence.append(
        _ev(
            symbol,
            "ADX trend strength",
            adx_pass,
            pts("adx_trend_strength", 7),
            "ADX is strong enough and +DI >= -DI",
            {"adx": round(adx_value, 2), "plus_di": round(plus_di, 2), "minus_di": round(minus_di, 2)},
        )
    )

    obv_slope = float(latest.get("obv_slope_5") or 0.0)
    adl_slope = float(latest.get("adl_slope_5") or 0.0)
    accumulation_pass = obv_slope > 0 or adl_slope > 0
    evidence.append(
        _ev(
            symbol,
            "Accumulation flow",
            accumulation_pass,
            pts("accumulation_flow", 5),
            "OBV or ADL slope is positive over the recent window",
            {"obv_slope_5": round(obv_slope, 3), "adl_slope_5": round(adl_slope, 3)},
        )
    )

    high_20 = float(latest["high_20"]) if pd.notna(latest["high_20"]) else close
    breakout_pass = close >= high_20 * 0.985
    evidence.append(_ev(symbol, "Breakout proximity", breakout_pass, pts("breakout_proximity", 8), "close is near 20-session high", round(high_20, 4)))

    rr_pass = reward_risk + 1e-9 >= thresholds["min_rr"] and stop_loss_pct <= thresholds["max_stop_loss_pct"]
    evidence.append(
        _ev(
            symbol,
            "Risk reward",
            rr_pass,
            pts("risk_reward", 14),
            "reward/risk and stop distance pass",
            {"rr": round(reward_risk, 3), "stop_loss_pct": round(stop_loss_pct, 3)},
        )
    )

    gap_pct = float(latest["return_1d"]) * 100 if pd.notna(latest["return_1d"]) else 0.0
    if gap_pct > thresholds["max_one_day_gain_pct"]:
        warnings.append(f"one-day gain {gap_pct:.2f}% exceeds T+2 chase limit")
        evidence.append(_ev(symbol, "Gap risk", False, 0, "too extended for T+2 entry", round(gap_pct, 3)))
        score -= pts("gap_risk_penalty", 12)
    else:
        evidence.append(_ev(symbol, "Gap risk", True, pts("gap_risk", 6), "one-day move below chase limit", round(gap_pct, 3)))

    # Market Regime Filter
    regime_trend = float(latest.get("regime_trend", 1.0))
    regime_pass = regime_trend >= 0
    regime_detail = f"Market regime: {'Bull' if regime_trend == 1.0 else 'Sideways' if regime_trend == 0.0 else 'Bear'}"
    evidence.append(
        _ev(
            symbol,
            "Market regime",
            regime_pass,
            pts("market_regime", 10) if regime_trend == 1.0 else pts("market_regime_sideways", 5) if regime_trend == 0.0 else 0.0,
            regime_detail,
            regime_trend
        )
    )

    # Relative Strength (RS)
    rs_slope = float(latest.get("rs_slope_5", 0.01))
    rs_pass = rs_slope > 0
    evidence.append(
        _ev(
            symbol,
            "Relative strength",
            rs_pass,
            pts("relative_strength", 10),
            "Stock is outperforming the market index proxy",
            round(rs_slope, 4)
        )
    )

    # Volatility Squeeze
    sqz_on = int(latest.get("sqz_on", 0))
    sqz_pass = sqz_on == 1
    evidence.append(
        _ev(
            symbol,
            "Volatility squeeze",
            sqz_pass,
            pts("volatility_squeeze", 8),
            "Bollinger Bands are inside Keltner Channels",
            sqz_on
        )
    )

    # VSA Stopping Volume
    vsa_stopping = int(latest.get("vsa_stopping_volume", 0))
    vsa_stop_pass = vsa_stopping == 1
    evidence.append(
        _ev(
            symbol,
            "VSA stopping volume",
            vsa_stop_pass,
            pts("vsa_stopping_volume", 12),
            "High volume narrow spread down bar indicating institutional buying",
            vsa_stopping
        )
    )

    # Pullback Entry
    ema50_val = float(latest.get("ema50", close))
    kijun_val = float(latest.get("kijun", close))
    vsa_no_supply = int(latest.get("vsa_no_supply", 0))
    near_ema50 = (close >= ema50_val * 0.99) and (close <= ema50_val * 1.02)
    near_kijun = (close >= kijun_val * 0.99) and (close <= kijun_val * 1.02)
    pullback_pass = (near_ema50 or near_kijun) and (vsa_no_supply == 1)
    evidence.append(
        _ev(
            symbol,
            "Pullback entry",
            pullback_pass,
            pts("pullback_entry", 12),
            "Price is near support (EMA50 or Kijun) with low volume/no supply",
            pullback_pass
        )
    )

    # Market-regime filter ("chong du dinh"): block momentum longs when the market is
    # risk-off (VNINDEX below EMA50 and/or weak breadth). Uses the market_regime column
    # attached upstream; default OFF and fail-open so behaviour is unchanged unless a
    # config enables it and the column is present. See features/market_regime.py.
    market_filter = rules.get("market_filter", {})
    mf_enabled = bool(market_filter.get("enabled", False))
    market_regime = float(latest.get("market_regime", 1.0)) if "market_regime" in features_df.columns else 1.0
    market_blocked = mf_enabled and market_regime < 0
    if mf_enabled:
        evidence.append(
            _ev(symbol, "Market filter", not market_blocked, pts("market_filter", 8),
                "market risk-on (VNINDEX>EMA50 and breadth ok)", market_regime)
        )

    for item in evidence:
        score += item.points
    score = max(-100.0, min(100.0, score))

    if close < float(latest["ema21"]):
        warnings.append("close below EMA21; no buy setup")
    if stop_loss_pct > thresholds["max_stop_loss_pct"]:
        warnings.append("stop distance too wide for T+2")
    if not rr_pass:
        warnings.append("reward/risk below T+2 threshold")
    if regime_trend == -1:
        warnings.append("market in bear regime; no buy setup")
    if market_blocked:
        warnings.append("market risk-off (VNINDEX below EMA50 / weak breadth); no buy setup")

    # Hard-block conditions: only structural problems block BUY_SETUP.
    # R:R and stop-distance warnings are already penalised via rule_points (risk_reward=0pts)
    # so they should not double-penalise by blocking the decision outright.
    hard_blocked = (close < float(latest["ema21"])) or (regime_trend == -1) or market_blocked

    if score >= thresholds["buy_setup"] and not hard_blocked:
        decision = "BUY_SETUP"
    elif score >= thresholds["watch"]:
        decision = "WATCH"
    else:
        decision = "REJECT"

    confidence_delta = max(0.0, min(0.08, score / 1000.0))
    feature_snapshot = build_feature_snapshot(symbol, features_df, latest) if include_features else {}
    feature_snapshot["score"] = score
    return SignalOutput(
        decision=decision,
        score=round(score, 2),
        confidence_delta=confidence_delta,
        latest_close=round(close, 4),
        latest_date=latest_date,
        risk_plan=risk_plan,
        evidence=evidence,
        warnings=warnings,
        features=feature_snapshot,
    )


def _score_mean_reversion_from_features(
    symbol: str,
    features_df: pd.DataFrame,
    rules: dict,
    idx: int | None = None,
    include_features: bool = True,
) -> SignalOutput:
    """Bottom-fishing (bat day) scorer — a distinct strategy from the momentum scorecard.

    Validated on VN100 2020->now (scratch/edge_experiment.py): the strict form has
    positive net expectancy even through the 2022 crash (avg ~+3.5%/trade, PF>2), but
    it is deliberately RARE — it only fires on capitulation + reversal. Unlike the
    momentum path it does NOT hard-block in a bear regime, because catching washed-out
    dips below EMA50 is the whole point. Entry conditions mirror the winning experiment
    variant; exit is delegated to the backtest via a tight ATR stop and an EMA21 target.
    """
    mr = rules.get("mean_reversion", {})
    rsi_max = float(mr.get("rsi_max", 30.0))
    band_touch = float(mr.get("band_touch_pct", 1.01))          # close <= bb_lower * band_touch
    vol_climax_min = float(mr.get("vol_climax_min", 1.5))
    stop_atr = float(mr.get("stop_atr_multiple", 1.5))
    min_tp_pct = float(mr.get("min_take_profit_pct", 1.0))
    max_hold = int(mr.get("max_hold_days", rules.get("time_horizon_days", 8)))
    # Hybrid transplants from the momentum scorecard (VN100-validated, see
    # scratch/edge_experiment.py MR+hybrid): Kijun as reversion target, VSA stopping
    # volume as alternative confirmation, RR gate, cloud-overhead clearance.
    target_anchor = str(mr.get("target_anchor", "ema21")).lower()
    allow_vsa_confirm = bool(mr.get("allow_vsa_confirm", False))
    min_rr = float(mr.get("min_rr", 0.0))
    require_cloud_clear = bool(mr.get("require_cloud_clear", False))

    row_idx = len(features_df) - 1 if idx is None else idx
    latest = features_df.iloc[row_idx]
    prev = features_df.iloc[row_idx - 1]
    warnings: list[str] = []
    evidence: list[RuleEvidence] = []

    close = float(latest["close"])
    latest_date = str(latest["date"])
    atr_value = float(latest.get("atr") or 0.0)
    prev_close = float(prev["close"])
    rsi_value = float(latest["rsi14"])
    bb_lower = float(latest.get("bb_lower") or close)
    ema21 = float(latest.get("ema21") or close)
    kijun = float(latest.get("kijun") or 0.0) or ema21
    vol_ratio = float(latest["volume_ratio_20"]) if pd.notna(latest.get("volume_ratio_20")) else 0.0
    vsa_stopping = int(latest.get("vsa_stopping_volume") or 0)

    anchor = kijun if target_anchor == "kijun" else ema21

    oversold = rsi_value < rsi_max
    at_band = close <= bb_lower * band_touch
    reversal = close > prev_close
    climax = vol_ratio >= vol_climax_min
    confirmed = (reversal and climax) or (allow_vsa_confirm and vsa_stopping == 1)

    pts = lambda k, d: float(rules.get("rule_points", {}).get(k, d))
    evidence.append(_ev(symbol, "Oversold RSI", oversold, pts("mr_oversold", 30), f"RSI14 < {rsi_max}", round(rsi_value, 2)))
    evidence.append(_ev(symbol, "At lower band", at_band, pts("mr_at_band", 25), "close at/below Bollinger lower band", round(bb_lower, 4)))
    evidence.append(_ev(symbol, "Reversal bar", reversal, pts("mr_reversal", 25), "close > previous close (up-reversal)", round(prev_close, 4)))
    evidence.append(_ev(symbol, "Volume climax", climax, pts("mr_volume_climax", 20), f"volume/avg20 >= {vol_climax_min} (capitulation)", round(vol_ratio, 3)))
    if allow_vsa_confirm:
        evidence.append(_ev(symbol, "VSA stopping volume", vsa_stopping == 1, 0.0,
                            "absorption bar accepted as alternative confirmation", vsa_stopping))

    # Tight ATR stop (falling-knife protection); reversion target = Kijun/EMA21 anchor.
    atr_stop = close - stop_atr * atr_value if atr_value > 0 else close * 0.95
    stop_loss = min(atr_stop, close * 0.995)
    if stop_loss >= close:
        stop_loss = close * 0.95
    risk_per_share = close - stop_loss
    take_profit_1 = max(anchor, close * (1 + min_tp_pct / 100.0))
    take_profit_2 = take_profit_1
    stop_loss_pct = risk_per_share / close * 100
    reward_risk = (take_profit_1 - close) / max(risk_per_share, 1e-9)
    risk_plan = RiskPlan(
        entry_reference=round(close, 4),
        stop_loss=round(stop_loss, 4),
        take_profit_1=round(take_profit_1, 4),
        take_profit_2=round(take_profit_2, 4),
        stop_loss_pct=round(stop_loss_pct, 3),
        reward_risk=round(reward_risk, 3),
        holding_period_days=max_hold,
    )

    # RR gate (transplanted from momentum's heaviest criterion): bounce room to the
    # anchor must cover the stop distance. min_rr=0 disables.
    rr_ok = True
    if min_rr > 0 and risk_per_share > 0:
        rr_ok = (anchor - close) >= min_rr * risk_per_share
        evidence.append(_ev(symbol, "MR risk reward", rr_ok, 0.0,
                            f"(anchor-close) >= {min_rr}x stop distance",
                            round((anchor - close) / risk_per_share, 3) if risk_per_share else None))

    # Ichimoku cloud clearance: reject if the cloud bottom blocks the path to the anchor.
    cloud_ok = True
    if require_cloud_clear:
        senkou_a = float(latest.get("senkou_a") or 0.0)
        senkou_b = float(latest.get("senkou_b") or 0.0)
        cloud_bottom = min(senkou_a, senkou_b) if senkou_a and senkou_b else 0.0
        blocked = bool(cloud_bottom) and (close < cloud_bottom < anchor)
        cloud_ok = not blocked
        evidence.append(_ev(symbol, "Cloud clearance", cloud_ok, 0.0,
                            "path to reversion target not blocked by Ichimoku cloud",
                            round(cloud_bottom, 4)))

    core = oversold and at_band and confirmed and rr_ok and cloud_ok
    score = max(-100.0, min(100.0, sum(item.points for item in evidence)))

    if core:
        decision = "BUY_SETUP"
    elif oversold and at_band:
        decision = "WATCH"
        warnings.append("oversold at lower band but confirmation/RR/cloud gate not met yet")
    else:
        decision = "REJECT"

    confidence_delta = max(0.0, min(0.08, score / 1000.0))
    feature_snapshot = build_feature_snapshot(symbol, features_df, latest) if include_features else {}
    feature_snapshot["score"] = score
    return SignalOutput(
        decision=decision,
        score=round(score, 2),
        confidence_delta=confidence_delta,
        latest_close=round(close, 4),
        latest_date=latest_date,
        risk_plan=risk_plan,
        evidence=evidence,
        warnings=warnings,
        features=feature_snapshot,
    )
