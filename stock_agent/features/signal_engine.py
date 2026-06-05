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
    return frames


def score_precomputed_symbol(symbol: str, features_df: pd.DataFrame, rules: dict) -> SignalOutput:
    if not has_signal_features(features_df):
        raise ValueError("score_precomputed_symbol requires a frame prepared by prepare_signal_frame")
    return _score_symbol_from_features(symbol, features_df, rules)


def score_precomputed_at(symbol: str, features_df: pd.DataFrame, idx: int, rules: dict) -> SignalOutput:
    if not has_signal_features(features_df):
        raise ValueError("score_precomputed_at requires a frame prepared by prepare_signal_frame")
    return _score_symbol_from_features(symbol, features_df, rules, idx=idx, include_features=False)


def score_symbol(symbol: str, df: pd.DataFrame, rules: dict) -> SignalOutput:
    return _score_symbol_from_features(symbol, prepare_signal_frame(df, rules), rules)


def _score_symbol_from_features(
    symbol: str,
    features_df: pd.DataFrame,
    rules: dict,
    idx: int | None = None,
    include_features: bool = True,
) -> SignalOutput:
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

    for item in evidence:
        score += item.points
    score = max(-100.0, min(100.0, score))

    if close < float(latest["ema21"]):
        warnings.append("close below EMA21; no buy setup")
    if stop_loss_pct > thresholds["max_stop_loss_pct"]:
        warnings.append("stop distance too wide for T+2")
    if not rr_pass:
        warnings.append("reward/risk below T+2 threshold")

    if score >= thresholds["buy_setup"] and not warnings:
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
