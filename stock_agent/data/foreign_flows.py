"""Daily foreign-flow snapshot collector + loader.

Two sources, complementary:

1. ``snapshot_today()`` — forward accumulator. One vnstock ``price_board`` call for the
   whole universe captures today's foreign buy/sell value/volume and room. Run it as part
   of the EOD pipeline (17:00) and history builds itself from today onward, independent of
   any third-party scraping. Appends to ``data/raw/foreign/price_board_snapshots.jsonl``.

2. ``load_flows()`` — unified loader that merges the harvested Vietstock history
   (``ndtnn.jsonl`` / ``tudoanh.jsonl`` from scratch/harvest_foreign_flows.py, ~9 months
   back on the free tier) with the forward snapshots, normalised to one row per
   (date, symbol) with columns:
       f_buy_val, f_sell_val, f_net_val   (billions VND, matched)
       f_buy_vol, f_sell_vol, f_net_vol
       own_ratio, room_remain_pct         (when available)
       td_buy_val, td_sell_val, td_net_val (tu doanh, when available)

Validation gate: these features must pass the IC test (scratch/foreign_ic.py) before any
integration into the signal path — same discipline as price features.
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path

import pandas as pd

FOREIGN_DIR = Path("data/raw/foreign")
SNAPSHOT_PATH = FOREIGN_DIR / "price_board_snapshots.jsonl"
NDTNN_PATH = FOREIGN_DIR / "ndtnn.jsonl"
TUDOANH_PATH = FOREIGN_DIR / "tudoanh.jsonl"


def snapshot_today(symbols: list[str]) -> int:
    """Capture today's foreign flows for `symbols` via vnstock price_board.

    Returns number of rows written. Safe to call repeatedly (same-day rows are
    replaced on load by keeping the last occurrence).
    """
    from vnstock import Vnstock

    FOREIGN_DIR.mkdir(parents=True, exist_ok=True)
    with redirect_stdout(io.StringIO()):
        board = Vnstock().stock(symbol=symbols[0], source="VCI").trading.price_board(symbols)
    if board is None or board.empty:
        return 0
    listing = board["listing"] if "listing" in board.columns.get_level_values(0) else board
    match = board["match"] if "match" in board.columns.get_level_values(0) else board
    today = str(date.today())
    n = 0
    with SNAPSHOT_PATH.open("a", encoding="utf-8") as f:
        for i in range(len(board)):
            try:
                rec = {
                    "date": today,
                    "symbol": str(listing["symbol"].iloc[i]),
                    "f_buy_vol": float(match["foreign_buy_volume"].iloc[i] or 0),
                    "f_sell_vol": float(match["foreign_sell_volume"].iloc[i] or 0),
                    "f_buy_val": float(match["foreign_buy_value"].iloc[i] or 0) / 1e9,
                    "f_sell_val": float(match["foreign_sell_value"].iloc[i] or 0) / 1e9,
                    "room_current": float(match["current_room"].iloc[i] or 0),
                    "room_total": float(match["total_room"].iloc[i] or 0),
                }
                rec["f_net_val"] = rec["f_buy_val"] - rec["f_sell_val"]
                rec["f_net_vol"] = rec["f_buy_vol"] - rec["f_sell_vol"]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            except Exception:
                continue
    return n


def _load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows)


def load_flows() -> pd.DataFrame:
    """Merge harvested Vietstock history + forward snapshots into one panel."""
    frames = []

    nd = _load_jsonl(NDTNN_PATH)
    if not nd.empty:
        nd = nd.rename(columns={
            "StockCode": "symbol", "BuyVal": "f_buy_val", "SellVal": "f_sell_val",
            "BuyVol": "f_buy_vol", "SellVol": "f_sell_vol",
            "OwnedRatio": "own_ratio", "RemainRoom": "room_remain_pct",
        })
        nd["f_net_val"] = nd["f_buy_val"].astype(float) - nd["f_sell_val"].astype(float)
        nd["f_net_vol"] = nd["f_buy_vol"].astype(float) - nd["f_sell_vol"].astype(float)
        keep = ["date", "symbol", "f_buy_val", "f_sell_val", "f_net_val",
                "f_buy_vol", "f_sell_vol", "f_net_vol", "own_ratio", "room_remain_pct"]
        frames.append(nd[[c for c in keep if c in nd.columns]])

    snap = _load_jsonl(SNAPSHOT_PATH)
    if not snap.empty:
        frames.append(snap)

    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.drop_duplicates(subset=["date", "symbol"], keep="last")

    td = _load_jsonl(TUDOANH_PATH)
    if not td.empty:
        td = td.rename(columns={"StockCode": "symbol"})
        td["td_buy_val"] = td.get("GTBuy_Total", td.get("BuyVal", 0)).astype(float)
        td["td_sell_val"] = td.get("GTSell_Total", td.get("SellVal", 0)).astype(float)
        td["td_net_val"] = td["td_buy_val"] - td["td_sell_val"]
        panel = panel.merge(td[["date", "symbol", "td_buy_val", "td_sell_val", "td_net_val"]],
                            on=["date", "symbol"], how="left")

    panel["date"] = pd.to_datetime(panel["date"]).dt.date
    return panel.sort_values(["symbol", "date"]).reset_index(drop=True)
