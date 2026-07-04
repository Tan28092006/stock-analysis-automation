"""What did the 8 recent MR signals actually make?

Two exits compared, net of 0.6% round-trip costs, entry = next session's OPEN:
  A) HOLD-TO-TODAY: sell at the latest close (user's question)
  B) SYSTEM EXIT:   stop (low<=stop) / target (high>=target) / time-stop 8 sessions,
                    with the T+2 settlement lock (cannot sell before entry+2 sessions)
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

DATA = Path(r"D:\Chungkhoan\data\raw\prices_hist")
COST = 0.60  # % round trip

SIGNALS = [
    ("2026-03-03", "DIG", 13198, 15275),
    ("2026-03-10", "CTR", 74579, 90200),
    ("2026-03-10", "MSN", 64607, 75800),
    ("2026-03-10", "VTP", 62047, 76315),
    ("2026-05-04", "PC1", 18005, 24975),
    ("2026-06-09", "ANV", 19577, 22100),
    ("2026-06-10", "FRT", 110663, 127815),
    ("2026-06-15", "SCS", 49034, 51625),
]

MAX_HOLD = 8
T2_LOCK = 2


def main():
    rows = []
    for sig_date, sym, stop, target in SIGNALS:
        df = pd.read_csv(DATA / f"{sym}.csv")
        df["date"] = df["date"].astype(str).str.slice(0, 10)
        df = df.sort_values("date").reset_index(drop=True)
        i_sig = df.index[df["date"] == sig_date]
        if len(i_sig) == 0:
            print(f"{sym}: signal date not found"); continue
        i_entry = int(i_sig[0]) + 1
        if i_entry >= len(df):
            continue
        entry = float(df.at[i_entry, "open"])
        entry_date = df.at[i_entry, "date"]
        last_close = float(df["close"].iloc[-1])
        last_date = df["date"].iloc[-1]

        hold_net = (last_close - entry) / entry * 100 - COST

        # system exit
        exit_px, exit_reason, exit_date = None, "TIME", None
        i_end = min(i_entry + MAX_HOLD, len(df) - 1)
        for j in range(i_entry, i_end + 1):
            if j < i_entry + T2_LOCK:
                continue
            lo, hi = float(df.at[j, "low"]), float(df.at[j, "high"])
            if lo <= stop:
                exit_px, exit_reason, exit_date = stop, "STOP", df.at[j, "date"]; break
            if hi >= target:
                exit_px, exit_reason, exit_date = target, "TARGET", df.at[j, "date"]; break
        if exit_px is None:
            exit_px, exit_date = float(df.at[i_end, "close"]), df.at[i_end, "date"]
        sys_net = (exit_px - entry) / entry * 100 - COST

        rows.append((sig_date, sym, entry_date, entry, last_close, hold_net,
                     exit_reason, exit_date, sys_net))

    print(f"Gia moi nhat: {rows[0][4] and ''}{'':s}(EOD {pd.read_csv(DATA/'VNINDEX.csv')['date'].astype(str).str.slice(0,10).max()}) | phi khu hoi 0.6%\n")
    print(f"{'signal':10s} {'ma':5s} {'entry ngay':10s} {'entry':>9s} | {'gia nay':>9s} {'OM->NAY%':>9s} | {'he thong':>8s} {'ngay ra':>10s} {'HT net%':>8s}")
    print("-" * 100)
    ha, sa = [], []
    for r in rows:
        sig_date, sym, ed, entry, lc, hn, reason, xd, sn = r
        ha.append(hn); sa.append(sn)
        print(f"{sig_date:10s} {sym:5s} {ed:10s} {entry:>9,.0f} | {lc:>9,.0f} {hn:>+8.2f}% | {reason:>8s} {xd:>10s} {sn:>+7.2f}%")
    print("-" * 100)
    n = len(ha)
    print(f"{'TRUNG BINH':38s}{'':11s} {sum(ha)/n:>+8.2f}% |{'':21s}{sum(sa)/n:>+7.2f}%")
    print(f"Win rate: om-toi-nay {sum(1 for x in ha if x>0)}/{n}  |  he-thong {sum(1 for x in sa if x>0)}/{n}")
    eq_h = 1.0
    eq_s = 1.0
    for x in ha: eq_h *= (1 + x/100)
    for x in sa: eq_s *= (1 + x/100)
    print(f"Neu chia von deu 8 keo: om-toi-nay tong {(sum(ha)/n):+.2f}%/keo  |  he-thong {(sum(sa)/n):+.2f}%/keo")


if __name__ == "__main__":
    main()
