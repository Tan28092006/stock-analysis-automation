from __future__ import annotations

from datetime import date
import hashlib

import numpy as np
import pandas as pd


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def make_demo_ohlcv(symbol: str, end: date, rows: int = 180, variant: int = 0) -> pd.DataFrame:
    """Create deterministic demo data. This is never labeled as real market data."""
    seed = _stable_int(symbol.upper()) % (2**32)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp(end), periods=rows)

    base = 18000 + (_stable_int(symbol.upper()) % 90000)
    drift = 0.0008 + ((_stable_int(symbol.upper()[:2]) % 7) / 10000)
    noise = rng.normal(0, 0.018, size=rows)
    close = base * np.exp(np.cumsum(drift + noise))

    # Give a subset of symbols a recent T+2-style momentum setup.
    if _stable_int(symbol.upper()) % 5 in {0, 2}:
        close[-8:] *= np.linspace(0.985, 1.065, 8)

    open_ = close * (1 + rng.normal(0, 0.006, size=rows))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.002, 0.018, size=rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.002, 0.018, size=rows))
    volume = rng.integers(250_000, 3_500_000, size=rows).astype(float)
    if _stable_int(symbol.upper()) % 5 in {0, 2}:
        volume[-3:] *= np.array([1.15, 1.35, 1.7])

    if variant:
        check_rng = np.random.default_rng((seed + variant * 7919) % (2**32))
        close = close * (1 + check_rng.normal(0, 0.0015, size=rows))
        open_ = open_ * (1 + check_rng.normal(0, 0.0015, size=rows))
        high = np.maximum.reduce([high, open_, close])
        low = np.minimum.reduce([low, open_, close])

    return pd.DataFrame(
        {
            "date": dates.date,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
