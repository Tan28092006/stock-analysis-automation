from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=max(2, span // 2)).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def bollinger(close: pd.Series, period: int = 20, stddev: float = 2.0) -> pd.DataFrame:
    middle = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    upper = middle + stddev * std
    lower = middle - stddev * std
    width = (upper - lower) / middle.replace(0, np.nan)
    return pd.DataFrame({"bb_mid": middle, "bb_upper": upper, "bb_lower": lower, "bb_width": width})


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def rolling_vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    volume = df["volume"].replace(0, np.nan)
    weighted = (typical * volume).rolling(period, min_periods=period).sum()
    volume_sum = volume.rolling(period, min_periods=period).sum()
    return weighted / volume_sum.replace(0, np.nan)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume.fillna(0)).cumsum()


def accumulation_distribution(df: pd.DataFrame) -> pd.Series:
    high_low = (df["high"] - df["low"]).replace(0, np.nan)
    money_flow_multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / high_low
    money_flow_volume = money_flow_multiplier.fillna(0) * df["volume"].fillna(0)
    return money_flow_volume.cumsum()


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_wilder = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_wilder.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_wilder.replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx_value = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return pd.DataFrame({"adx14": adx_value.fillna(0), "plus_di14": plus_di.fillna(0), "minus_di14": minus_di.fillna(0)})


def ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou_reference = close.shift(26)
    return pd.DataFrame(
        {
            "tenkan": tenkan,
            "kijun": kijun,
            "senkou_a": span_a,
            "senkou_b": span_b,
            "chikou_reference": chikou_reference,
        }
    )


def keltner_channel(df: pd.DataFrame, period: int = 20, multiplier: float = 1.5) -> pd.DataFrame:
    close = df["close"]
    middle = ema(close, period)
    atr_val = atr(df, period)
    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val
    return pd.DataFrame({"kc_mid": middle, "kc_upper": upper, "kc_lower": lower})


def squeeze_detector(df: pd.DataFrame, bb_period: int = 20, kc_period: int = 20, multiplier: float = 1.5) -> pd.DataFrame:
    bb = bollinger(df["close"], bb_period, stddev=2.0)
    kc = keltner_channel(df, kc_period, multiplier)
    
    sqz_on = (bb["bb_upper"] < kc["kc_upper"]) & (bb["bb_lower"] > kc["kc_lower"])
    bb_width = bb["bb_upper"] - bb["bb_lower"]
    kc_width = kc["kc_upper"] - kc["kc_lower"]
    bb_kc_ratio = bb_width / kc_width.replace(0, np.nan)
    return pd.DataFrame({
        "kc_upper": kc["kc_upper"],
        "kc_lower": kc["kc_lower"],
        "sqz_on": sqz_on.astype(int),
        "bb_kc_ratio": bb_kc_ratio.fillna(1.0)
    })


def add_vsa_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    spread = out["high"] - out["low"]
    avg_spread = spread.rolling(20, min_periods=5).mean()
    spread_ratio = spread / avg_spread.replace(0, np.nan)
    
    close = out["close"]
    low = out["low"]
    high = out["high"]
    open_val = out["open"]
    volume = out["volume"]
    avg_vol = volume.rolling(20, min_periods=5).mean()
    volume_ratio = volume / avg_vol.replace(0, np.nan)
    
    # Stopping Volume: price down (close < open or return_1d < 0), high volume, narrow spread, close in upper half of bar
    spread_range = (high - low).replace(0, np.nan)
    close_pos_in_bar = (close - low) / spread_range
    vsa_stopping_volume = (
        (close < open_val) & 
        (volume_ratio >= 1.5) & 
        (spread_ratio <= 1.0) & 
        (close_pos_in_bar >= 0.4)
    ).astype(int)
    
    # No Demand: price up, low volume, narrow spread
    vsa_no_demand = (
        (close > open_val) & 
        (volume_ratio <= 0.8) & 
        (spread_ratio <= 0.8)
    ).astype(int)
    
    # No Supply (cạn cung): price down/flat, very low volume, narrow spread
    vsa_no_supply = (
        (close <= open_val) & 
        (volume_ratio <= 0.7) & 
        (spread_ratio <= 0.8)
    ).astype(int)
    
    out["spread_ratio"] = spread_ratio.fillna(1.0)
    out["vsa_stopping_volume"] = vsa_stopping_volume
    out["vsa_no_demand"] = vsa_no_demand
    out["vsa_no_supply"] = vsa_no_supply
    return out


def add_indicators(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    out["return_1d"] = close.pct_change()
    out["log_return_1d"] = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    out["return_2d"] = close.pct_change(2)
    out["return_3d"] = close.pct_change(3)
    out["sma5"] = sma(close, 5)
    out["sma20"] = sma(close, 20)
    out["sma50"] = sma(close, 50)
    out["ema9"] = ema(close, 9)
    out["ema21"] = ema(close, 21)
    out["ema50"] = ema(close, 50)
    out["rsi14"] = rsi(close, 14)
    out = pd.concat([out, macd(close), bollinger(close), ichimoku(out), adx(out, 14), squeeze_detector(out)], axis=1)
    out["atr"] = atr(out, atr_period)
    out["vwap20"] = rolling_vwap(out, 20)
    out["obv"] = obv(close, out["volume"])
    out["obv_slope_5"] = out["obv"].diff(5)
    out["adl"] = accumulation_distribution(out)
    out["adl_slope_5"] = out["adl"].diff(5)
    out["avg_volume_20"] = out["volume"].rolling(20).mean()
    out["volume_ratio_20"] = out["volume"] / out["avg_volume_20"].replace(0, np.nan)
    out["volume_z20"] = (out["volume"] - out["avg_volume_20"]) / out["volume"].rolling(20).std().replace(0, np.nan)
    out["value_traded"] = out["close"] * out["volume"]
    out["avg_value_20"] = out["value_traded"].rolling(20).mean()
    
    # Add VSA features
    out = add_vsa_features(out)
    
    out["spread_proxy_pct"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    out["rolling_mean_5"] = close.rolling(5).mean()
    out["rolling_mean_20"] = close.rolling(20).mean()
    out["rolling_std_5"] = close.pct_change().rolling(5).std()
    out["rolling_std_20"] = close.pct_change().rolling(20).std()
    out["high_20"] = out["high"].rolling(20).max()
    out["low_20"] = out["low"].rolling(20).min()
    for lag in (1, 2, 3):
        out[f"close_lag_{lag}"] = close.shift(lag)
        out[f"rsi14_lag_{lag}"] = out["rsi14"].shift(lag)
        out[f"macd_hist_lag_{lag}"] = out["macd_hist"].shift(lag)
        out[f"volume_ratio_20_lag_{lag}"] = out["volume_ratio_20"].shift(lag)
    return out
