from __future__ import annotations

import pandas as pd


def pass_basic_filter(symbol: str, df: pd.DataFrame, rules: dict) -> tuple[bool, list[str]]:
    filter_rules = rules["fundamental_filter"]
    warnings: list[str] = []
    if symbol.upper() in set(filter_rules.get("restricted_symbols", [])):
        return False, [f"{symbol} is in restricted symbol list"]

    avg_volume_20 = float(df["volume"].tail(20).mean()) if len(df) >= 20 else 0.0
    if avg_volume_20 < float(filter_rules["min_avg_volume_20"]):
        warnings.append(f"avg_volume_20 {avg_volume_20:.0f} below threshold")
    return not warnings, warnings

