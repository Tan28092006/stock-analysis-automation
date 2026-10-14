"""R5 weekly momentum lane (protocol ml_research_v2_weekly). Research only, per-lot diagnostics.

Registered config: configs/research/ml_protocol_v2_weekly.json (committed before results).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from scripts.ml_research_r5 import feature_table, families, load
from stock_agent.research_ml import evidence as E
from stock_agent.research_ml import harness as H
from stock_agent.research_ml.labels import CostConfig, build_labels

warnings.filterwarnings("ignore", category=FutureWarning)
PROTOCOL = Path("configs/research/ml_protocol_v2_weekly.json")


def circular_block_ci(x: np.ndarray, block: int, draws: int, seed: int) -> dict:
    """Paired circular block bootstrap of the mean of a date-level excess series."""
    n = len(x)
    rng = np.random.default_rng(seed + block)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(draws, nb))
    idx = (starts[:, :, None] + np.arange(block)) % n
    means = x[idx.reshape(draws, -1)[:, :n]].mean(axis=1)
    return {"block": block, "mean": float(x.mean()), "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
            "p_one_sided_le0": float((means <= 0).mean())}


def topk_by_date(frame: pd.DataFrame, score_col: str, k: int) -> pd.DataFrame:
    return (frame.sort_values(["signal_date", score_col, "symbol"], ascending=[True, False, True])
            .groupby("signal_date").head(k))


def date_series(frame: pd.DataFrame, score_col: str, k: int, extra_cost: float = 0.0) -> pd.DataFrame:
    top = topk_by_date(frame, score_col, k).groupby("signal_date").net_return_pct.mean() - extra_cost
    uni = frame.groupby("signal_date").net_return_pct.mean()
    out = pd.DataFrame({"top": top, "universe": uni}).dropna()
    out["excess"] = out.top - out.universe
    return out


def rank_ic(frame: pd.DataFrame, score_col: str) -> pd.Series:
    return frame.groupby("signal_date").apply(
        lambda g: g[score_col].rank().corr(g.net_return_pct.rank()) if len(g) > 4 else np.nan)


def summarize(frame, score_col, proto, k, cost):
    s = date_series(frame, score_col, k)
    inf = proto["inference"]
    out = {"dates": int(len(s)), "top10_mean_net_pct": float(s.top.mean()), "universe_mean_net_pct": float(s.universe.mean()),
           "excess_mean_pct": float(s.excess.mean()), "rank_ic_mean": float(rank_ic(frame, score_col).mean()),
           "cost_x2_excess_mean_pct": float(date_series(frame, score_col, k, cost.round_trip_pct).excess.mean()),
           "bootstrap": [circular_block_ci(s.excess.to_numpy(), b, inf["draws"], inf["seed"]) for b in inf["block_lengths_dates"]]}
    return out


def build_dataset(proto, tl, bars, frames):
    win = proto["data"]["decision_window"]
    cal = frames["VNINDEX"]
    days = cal[(cal.date >= win["start"]) & (cal.date <= win["end"])].date.tolist()
    dec_days = set(pd.to_datetime(days[::5]))
    feats = feature_table(bars, tl, win["start"], win["end"])
    dec = feats[feats.session_date.isin(dec_days)]
    sig = dec[["symbol", "session_date"]].rename(columns={"session_date": "signal_date"})
    labels = build_labels(bars, sig, protocol="mom_5", cost=CostConfig(**proto["costs"]))
    df = labels.merge(dec, left_on=["symbol", "signal_date"], right_on=["symbol", "session_date"])
    return df, len(dec_days)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    proto = json.loads(PROTOCOL.read_text())
    seed, k = proto["model_m1"]["seed"], proto["model_m1"]["top_k"]
    cost = CostConfig(**proto["costs"])
    tl, manifest, frames, bars = load(a.manifest, proto)
    df, n_decision = build_dataset(proto, tl, bars, frames)
    mature = df[df.status == "matured"].sort_values(["signal_date", "symbol"]).reset_index(drop=True)
    fl = proto["floors"]
    fg = proto["folds"]
    folds = H.purged_grouped_folds(mature, 0, min_train_dates=fl["train_dates_min"], min_test_dates=fl["test_dates_min"],
                                   first_train_dates=fg["first_train_dates"], test_block_dates=fg["test_block_dates"])
    usable = [f for f in folds if f.conclusive]
    metrics = {"decision_dates": n_decision, "matured_dates": int(mature.signal_date.nunique()), "matured_rows": len(mature),
               "label_status": df.status.value_counts().to_dict(),
               "folds": [dict(index=f.index, train_dates=f.train_dates, test_dates=f.test_dates, purged_rows=f.purged_rows,
                              problems=f.problems) for f in folds], "trials": {}}
    test_rows = np.concatenate([f.test for f in usable]) if usable else np.array([], dtype=int)
    oos = mature.iloc[test_rows].copy()
    if len(oos):
        mm = oos.dropna(subset=["f0_mom_126_21"])
        metrics["m0_rule_only_oos"] = summarize(mm, "f0_mom_126_21", proto, k, cost)
        metrics["m0_rule_only_all_dates"] = summarize(mature.dropna(subset=["f0_mom_126_21"]), "f0_mom_126_21", proto, k, cost)
    tf = lambda: H.FoldTransform(tuple(proto["preprocessing"]["winsor"]), proto["preprocessing"]["scaler"])
    ridge = {"ridge": lambda: Ridge(alpha=proto["model_m1"]["ridge"]["alpha"], random_state=seed)}
    rng = np.random.default_rng(seed)
    for trial in proto["trials"] + ["control_label_shuffle_within_date"]:
        control = trial.startswith("control_")
        cols = [c for c in families(proto, "baseline_all_features" if control else trial) if mature[c].notna().any()]
        r = {"features": cols}
        try:
            y = mature.net_return_pct.copy()
            if control:  # shuffle target within each date; evaluation still uses true returns
                y = mature.groupby("signal_date").net_return_pct.transform(lambda s: rng.permutation(s.to_numpy()))
            oof, _ = H.forward_oof(mature[cols], y, folds, ridge, tf)
            if oof.empty:
                r["status"] = "no_conclusive_folds"
            else:
                sc = mature.loc[oof.row].assign(pred=oof.pred.to_numpy())
                r.update(summarize(sc, "pred", proto, k, cost))
        except Exception as exc:  # retain failures
            r["error"] = f"{type(exc).__name__}: {exc}"
        metrics["trials"][trial] = r
    base = metrics["trials"].get("baseline_all_features", {})
    for trial, r in metrics["trials"].items():
        if trial.startswith("drop_") and "excess_mean_pct" in r and "excess_mean_pct" in base:
            r["delta_excess_vs_baseline_pct"] = r["excess_mean_pct"] - base["excess_mean_pct"]

    run = E.init_bundle(a.output, proto["protocol"], "r5-weekly-momentum", a.run_id)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    floors = {"dates": fl["train_dates_min"] + fl["test_dates_min"]}
    E.write_json(run / "experiment.json", {"hypotheses": ["H-MO02(simplified, per-lot)"], "protocol": proto, "git_commit": commit,
                                           "sample_floors": floors,
                                           "activity_floor": {"status": "not_evaluated_in_R5", **fl["activity"]}})
    E.write_json(run / "source_manifest.json", {"manifest_sha256": E.sha256_file(a.manifest), "as_of": manifest["as_of"],
                                                "symbols": len(frames), "limitations": manifest["verification_scope"]})
    E.write_json(run / "environment.json", {"python": sys.version, "seed": seed,
                                            "pkgs": {m: __import__(m).__version__ for m in ("pandas", "numpy", "sklearn")}})
    E.write_json(run / "dataset_manifest.json", {k_: metrics[k_] for k_ in ("decision_dates", "matured_dates", "matured_rows", "label_status")})
    E.write_json(run / "splits.json", {"folds": metrics["folds"]})
    metrics["sample_support"] = {"dates": metrics["matured_dates"]}
    E.write_json(run / "metrics.json", metrics)
    for d, note in (("transforms", "fit inside forward_oof per fold; hashes not persisted in R5 diagnostics"),
                    ("predictions", "OOF summarized in metrics.json only"),
                    ("regression", "R7 gate/books/timeslices not run"), ("portfolio", "no NAV replay in R5")):
        E.write_json(run / d / "README.json", {"note": note})
    E.write_json(run / "decision.json", {"status": "BLOCKED", "live_eligible": False})  # provisional
    E.seal(run)
    verdict = E.validate_bundle(run)
    status = {"engineering_complete": "INCONCLUSIVE"}.get(verdict["status"], verdict["status"])  # R7 still required
    E.write_json(run / "decision.json", {"status": status, "validator": verdict, "live_eligible": False, "reviewer": None,
                                         "limitations": ["development data", "per-lot diagnostics, no NAV", "R7 regression absent"]})
    print(json.dumps({"run": str(run), "decision": status, "validator": verdict}, indent=2))


if __name__ == "__main__":
    main()
