"""R3: versioned feature registry and causal F0 / F2 / F3 builders.

Features at session ``t`` use bars <= t only and become available at the session
close (16:00 Asia/Ho_Chi_Minh). Insufficient warmup, missing and stale are states,
never zeros. F1 (market/breadth) and F4 (sector/fundamental) stay BLOCKED until
PIT sources exist; they are intentionally absent from the registry.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from .contracts import VN_TZ, MissingState, assert_no_target_leak

FEATURE_SET_VERSION = "f0f2f3-v1"
CLOSE_HOUR = 16


@dataclass(frozen=True)
class FeatureSpec:
    feature_id: str
    family: str
    hypothesis: str
    unit: str
    lookback: int          # sessions of history required
    missing_policy: str    # what a consumer must do with non-observed states
    strategies: tuple[str, ...] = ("mr", "momentum")


REGISTRY: dict[str, FeatureSpec] = {s.feature_id: s for s in [
    FeatureSpec("f0_ret_1", "F0", "short reversal", "ratio", 2, "state+nan"),
    FeatureSpec("f0_ret_5", "F0", "short reversal", "ratio", 6, "state+nan"),
    FeatureSpec("f0_mom_126_21", "F0", "12-1 style momentum, skip last 21", "ratio", 127, "state+nan",
                ("momentum",)),
    FeatureSpec("f0_vol_20", "F0", "realised volatility", "ratio", 21, "state+nan"),
    FeatureSpec("f0_high_prox_120", "F0", "52w-high proximity (120 sessions)", "ratio", 120, "state+nan"),
    FeatureSpec("f0_drawdown_60", "F0", "drawdown from 60-session high", "ratio", 60, "state+nan", ("mr",)),
    FeatureSpec("f0_volume_anomaly_20", "F0", "volume / 20-session mean", "ratio", 20, "state+nan"),
    FeatureSpec("f2_adv_value_20", "F2", "liquidity (close*volume, VND)", "VND", 20, "state+nan"),
    FeatureSpec("f3_foreign_net_adv_5", "F3", "lagged foreign net value / ADV", "ratio", 5, "state+nan"),
]}


def registry_hash() -> str:
    payload = json.dumps({k: asdict(v) for k, v in sorted(REGISTRY.items())}, sort_keys=True)
    return hashlib.sha256((FEATURE_SET_VERSION + payload).encode()).hexdigest()


def feature_columns(frame_columns: list[str]) -> list[str]:
    cols = [c for c in frame_columns if c in REGISTRY]
    assert_no_target_leak(cols, allowlist=set(REGISTRY))
    return cols


def _f0_symbol(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("session_date").reset_index(drop=True)
    c, v = g["close"], g["volume"]
    ret1 = c.pct_change()
    out = pd.DataFrame({"symbol": g.symbol, "session_date": g.session_date})
    out["f0_ret_1"] = ret1
    out["f0_ret_5"] = c.pct_change(5)
    out["f0_mom_126_21"] = c.shift(21) / c.shift(126) - 1
    out["f0_vol_20"] = ret1.rolling(20).std()
    out["f0_high_prox_120"] = c / c.rolling(120).max()
    out["f0_drawdown_60"] = c / c.rolling(60).max() - 1
    out["f0_volume_anomaly_20"] = v / v.rolling(20).mean().replace(0, np.nan)
    out["f2_adv_value_20"] = (c * v).rolling(20).mean()
    out["_n"] = np.arange(1, len(g) + 1)
    return out


def build_price_features(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per symbol x session. ``<feature>_state`` marks warmup / missing."""
    parts = [_f0_symbol(g) for _, g in bars.groupby("symbol", sort=False)]
    out = pd.concat(parts, ignore_index=True)
    for fid, spec in REGISTRY.items():
        if fid not in out:
            continue
        warm = out["_n"] < spec.lookback
        out.loc[warm, fid] = np.nan
        invalid = ~np.isfinite(out[fid].to_numpy(dtype=float))
        out[fid + "_state"] = np.where(warm, MissingState.INSUFFICIENT_HISTORY.value,
                                       np.where(invalid, MissingState.MISSING.value,
                                                MissingState.OBSERVED.value))
        out.loc[invalid & ~warm, fid] = np.nan
    out["decision_time"] = out["session_date"].dt.tz_localize(VN_TZ) + pd.Timedelta(hours=CLOSE_HOUR)
    return out.drop(columns="_n")


def build_foreign_feature(price_feats: pd.DataFrame, foreign: pd.DataFrame, sessions: list[pd.Timestamp],
                          *, lag: int = 1, window: int = 5, min_support: int = 4) -> pd.DataFrame:
    """Lagged net foreign value / ADV over the exchange-session skeleton.

    ``foreign`` must be pre-filtered to verified rows knowable at decision time.
    Missing session slots stay missing (never zero); a window with fewer than
    ``min_support`` present slots is MISSING, and coverage is recorded.
    """
    skeleton = pd.DatetimeIndex(sorted(pd.to_datetime(sessions)))
    net = (foreign.assign(net=foreign.buy_value - foreign.sell_value)
           .groupby(["symbol", "session_date"], as_index=False).net.sum())
    out = price_feats.copy()
    vals = np.full(len(out), np.nan)
    cover = np.zeros(len(out), dtype=int)
    for symbol, idx in out.groupby("symbol").groups.items():
        series = (net[net.symbol == symbol].set_index("session_date").net
                  .reindex(skeleton))  # missing slots preserved as NaN
        lagged = series.shift(lag)
        roll = lagged.rolling(window, min_periods=1)
        total, present = roll.sum(), lagged.notna().rolling(window, min_periods=1).sum()
        pos = skeleton.get_indexer(out.loc[idx, "session_date"])
        ok = pos >= 0
        s = np.where(ok, total.to_numpy()[pos.clip(0)], np.nan)
        p = np.where(ok, present.to_numpy()[pos.clip(0)], 0)
        vals[out.index.get_indexer(idx)] = s
        cover[out.index.get_indexer(idx)] = p
    adv = out["f2_adv_value_20"].replace(0, np.nan)
    feat = pd.Series(vals, index=out.index) / adv
    feat[cover < min_support] = np.nan
    out["f3_foreign_net_adv_5"] = feat
    out["f3_foreign_net_adv_5_coverage"] = cover / window
    out["f3_foreign_net_adv_5_state"] = np.where(feat.notna(), MissingState.OBSERVED.value,
                                                 MissingState.MISSING.value)
    return out


def feature_row_lineage(rows: pd.DataFrame) -> dict:
    cols = feature_columns(list(rows.columns))
    digest = hashlib.sha256(rows[["symbol", "session_date", *cols]].to_json(
        orient="split", date_format="iso").encode()).hexdigest()
    return {"feature_set_version": FEATURE_SET_VERSION, "registry_sha256": registry_hash(),
            "rows": len(rows), "feature_ids": cols, "rows_sha256": digest,
            "available_at_max": str(rows["decision_time"].max())}
