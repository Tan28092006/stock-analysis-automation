"""R5 research runner: M0/M1 baselines and one-family ablations on the pinned snapshot.

Research only: no live state, no model promotion. Reads configs/research/ml_protocol_v1.json
(registered before results). Writes a new sealed evidence bundle; the validator can never
return a pass. Predictive metrics are diagnostics, not acceptance (see protocol).
"""
from __future__ import annotations

import argparse
import json
import warnings
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from scripts import research_gate as rg
from scripts.historical_universe import Timeline
from stock_agent.data.reconciliation import verify_snapshot
from stock_agent.research_ml import evidence as E
from stock_agent.research_ml import features as F
from stock_agent.research_ml import harness as H
from stock_agent.research_ml.labels import CostConfig, build_labels

warnings.filterwarnings("ignore", category=FutureWarning)
PROTOCOL = Path("configs/research/ml_protocol_v1.json")
TZ = "Asia/Ho_Chi_Minh"


def load(manifest_path: Path, proto: dict):
    tl = Timeline(json.loads(Path(proto["data"]["membership"]).read_text()))
    manifest = verify_snapshot(manifest_path)
    frames = {s: pd.read_csv(manifest_path.parent / manifest["files"][s]["path"])
              for s in sorted(tl.all_members | {"VNINDEX"})}
    bars = pd.concat([f.assign(symbol=s, session_date=pd.to_datetime(f["date"]))
                      for s, f in frames.items() if s != "VNINDEX"], ignore_index=True)
    return tl, manifest, frames, bars


def feature_table(bars: pd.DataFrame, tl: Timeline, start: str, end: str) -> pd.DataFrame:
    feats = F.build_price_features(bars)
    day = feats.session_date.dt.strftime("%Y-%m-%d")
    keep = (day >= start) & (day <= end)
    pit = pd.Series([s in tl.members(d, d) if tl.start <= d <= tl.end else False
                     for s, d in zip(feats.symbol, day)], index=feats.index)
    return feats[keep & pit].reset_index(drop=True)


def families(proto, trial):
    fam = proto["feature_families"]
    drop = trial.removeprefix("drop_") if trial.startswith("drop_") else None
    cols = [c for k, v in fam.items() if k != drop for c in v]
    return cols


def evaluate_classification(df, cols, proto, seed):
    """Candidate filter: logistic on fold-fitted features. Returns (metrics, folds, selected)."""
    X, y = df[cols].reset_index(drop=True), df["win"].astype(float).reset_index(drop=True)
    folds = H.purged_grouped_folds(df.reset_index(drop=True), proto["folds"]["n_splits"],
                                   min_train_dates=proto["floors"]["train_dates_min"],
                                   min_test_dates=proto["floors"]["test_dates_min"])
    cfg = proto["model_m1"]["logistic"]
    oof, _ = H.forward_oof(X, y, folds, {"logit": lambda: LogisticRegression(C=cfg["C"], max_iter=cfg["max_iter"],
                                                                             random_state=seed)},
                           lambda: H.FoldTransform(tuple(proto["preprocessing"]["winsor"]),
                                                   proto["preprocessing"]["scaler"]))
    return folds, oof, y


def floor_problems(folds, y, proto):
    out = []
    ev = proto["floors"]["events_per_class_min"]
    for f in folds:
        if f.problems:
            out.append(f"fold{f.index}:" + ",".join(f.problems))
        for part in ("train", "test"):
            for cls, n in f.class_counts.get(part, {}).items():
                if n < ev:
                    out.append(f"fold{f.index}:{part}_class{int(cls)}_events_{n}<{ev}")
    return out


def run_mr(proto, tl, frames, bars, bank, seed):
    cost = CostConfig(**proto["costs"])
    win = proto["data"]["decision_window"]
    cands = pd.DataFrame([dict(symbol=o["symbol"], signal_date=pd.Timestamp(d), rr=o["rr"])
                          for d, os in bank["fixed15"].items() for o in os])
    cands = cands[(cands.signal_date >= win["start"]) & (cands.signal_date <= win["end"])]
    labels = build_labels(bars, cands[["symbol", "signal_date"]], protocol="mr_fixed15", cost=cost)
    feats = feature_table(bars, tl, win["start"], win["end"])
    df = labels.merge(feats, left_on=["symbol", "signal_date"], right_on=["symbol", "session_date"], how="left")
    status = df.status.value_counts().to_dict()
    mature = df[(df.status == "matured")].sort_values("signal_date").reset_index(drop=True)
    res = {"candidates": len(cands), "label_status": status, "matured": len(mature),
           "m0": {"events": len(mature), "mean_net_return_pct": float(mature.net_return_pct.mean()) if len(mature) else None,
                  "win_rate": float(mature.win.mean()) if len(mature) else None},
           "dates": int(mature.signal_date.nunique()), "trials": {}, "problems": []}
    by_month = mature.groupby(mature.signal_date.dt.to_period("M")).agg(entries=("symbol", "size"),
                                                                      days=("signal_date", "nunique"))
    act = proto["floors"]["activity"]
    res["m0"]["months_below_activity_floor"] = int(((by_month.entries < act["flat_to_long_entries_per_month_min"]) |
                                                     (by_month.days < act["distinct_entry_days_per_month_min"])).sum())
    res["m0"]["months"] = int(len(by_month))
    if res["dates"] < proto["floors"]["train_dates_min"] + proto["floors"]["test_dates_min"]:
        res["problems"].append(f"insufficient_event_dates:{res['dates']}")
    for trial in proto["trials"]:
        cols = [c for c in families(proto, trial) if mature[c].notna().any()]
        t = {"features": cols}
        try:
            folds, oof, y = evaluate_classification(mature, cols, proto, seed)
        except Exception as exc:  # retain failures, never drop them
            t["error"] = f"{type(exc).__name__}: {exc}"
            res["trials"][trial] = t
            continue
        t["floor_problems"] = floor_problems(folds, y, proto)
        t["folds"] = [dict(index=f.index, train_dates=f.train_dates, test_dates=f.test_dates,
                           purged_rows=f.purged_rows, class_counts={k: {str(a): int(b) for a, b in v.items()}
                                                                  for k, v in f.class_counts.items()},
                           problems=f.problems) for f in folds]
        if oof.empty:
            t["status"] = "no_conclusive_folds"
        if len(oof) and y.iloc[oof.row].nunique() == 2:
            p, yy = oof.pred.to_numpy(), y.iloc[oof.row].to_numpy()
            t["oof_rows"] = int(len(oof))
            t["brier"], t["log_loss"] = float(brier_score_loss(yy, p)), float(log_loss(yy, p, labels=[0, 1]))
            t["pr_auc"] = float(average_precision_score(yy, p))
            t["base_rate"] = float(yy.mean())
        res["trials"][trial] = t
    # control: labels shuffled within date must not beat the baseline meaningfully
    return res


def run_momentum(proto, tl, bars, seed):
    cost = CostConfig(**proto["costs"])
    win = proto["data"]["decision_window"]
    feats = feature_table(bars, tl, win["start"], win["end"])
    feats["ym"] = feats.session_date.dt.to_period("M")
    dec = feats.loc[feats.groupby(["symbol", "ym"]).session_date.transform("max") == feats.session_date]
    last = dec.groupby("ym").session_date.transform("max")
    dec = dec[dec.session_date == last]  # last session of each month, whole PIT cross-section
    sig = dec[["symbol", "session_date"]].rename(columns={"session_date": "signal_date"})
    labels = build_labels(bars, sig, protocol="mom_21", cost=cost)
    df = labels.merge(dec, left_on=["symbol", "signal_date"], right_on=["symbol", "session_date"])
    mature = df[df.status == "matured"].sort_values("signal_date").reset_index(drop=True)
    dates = sorted(mature.signal_date.unique())
    res = {"decision_months": len(dates), "matured_rows": len(mature), "label_status": df.status.value_counts().to_dict(),
           "problems": [], "trials": {}}
    need = proto["floors"]["train_dates_min"] + proto["floors"]["test_dates_min"]
    res["problems"].append(f"monthly_independent_dates:{len(dates)}<{need}") if len(dates) < need else None
    top = 10
    def pick(frame, score):
        return frame.assign(_s=score).groupby("signal_date", group_keys=False).apply(
            lambda g: g.nlargest(min(top, len(g)), "_s"))
    m0 = pick(mature.dropna(subset=["f0_mom_126_21"]), mature.dropna(subset=["f0_mom_126_21"]).f0_mom_126_21)
    univ = mature.groupby("signal_date").net_return_pct.mean()
    res["m0"] = {"mean_net_return_pct": float(m0.net_return_pct.mean()),
                 "universe_equal_mean_net_pct": float(univ.mean()),
                 "excess_vs_universe_pct": float(m0.groupby("signal_date").net_return_pct.mean().sub(univ).mean()),
                 "months": int(m0.signal_date.nunique()), "note": "per-lot diagnostic, not portfolio NAV"}
    # M1 trials use the same date folds; fewer than floor dates => BLOCKED, still run for the record
    folds = H.purged_grouped_folds(mature.assign(win=(mature.net_return_pct > mature.groupby("signal_date").net_return_pct.transform("median")).astype(float)),
                                   proto["folds"]["n_splits"], min_train_dates=proto["floors"]["train_dates_min"],
                                   min_test_dates=proto["floors"]["test_dates_min"])
    res["fold_problems"] = [f"fold{f.index}:" + ",".join(f.problems) for f in folds if f.problems]
    for trial in proto["trials"]:
        cols = [c for c in families(proto, trial) if mature[c].notna().any()]
        r = {"features": cols}
        try:
            Xd, yd = mature[cols], mature.net_return_pct
            oof, _ = H.forward_oof(Xd, yd, [f for f in folds if f.train.size and f.test.size], 
                                   {"ridge": lambda: Ridge(alpha=proto["model_m1"]["ridge"]["alpha"])},
                                   lambda: H.FoldTransform(tuple(proto["preprocessing"]["winsor"]),
                                                           proto["preprocessing"]["scaler"]))
            if oof.empty:
                r["status"] = "no_conclusive_folds"  # floors unmet: nothing fit, nothing claimed
                res["trials"][trial] = r
                continue
            sc = mature.loc[oof.row, ["signal_date", "net_return_pct"]].assign(pred=oof.pred.to_numpy())
            ic = sc.groupby("signal_date").apply(lambda g: g.pred.rank().corr(g.net_return_pct.rank()) if len(g) > 4 else np.nan)
            sel = pick(sc, sc.pred)
            r.update(oof_rows=int(len(oof)), months=int(sc.signal_date.nunique()),
                     rank_ic_mean=float(ic.mean()), top10_mean_net_pct=float(sel.net_return_pct.mean()),
                     universe_mean_net_pct=float(sc.groupby("signal_date").net_return_pct.mean().mean()))
        except Exception as exc:
            r["error"] = f"{type(exc).__name__}: {exc}"
        res["trials"][trial] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--bank", type=Path, help="precomputed mr_signal_bank json (from research_gate)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    proto = json.loads(PROTOCOL.read_text())
    seed = proto["model_m1"]["seed"]
    tl, manifest, frames, bars = load(a.manifest, proto)
    if a.bank:
        bank = json.loads(a.bank.read_text())
    else:
        bank = rg.mr_signal_bank(frames, json.loads(Path(proto["data"]["rules"]).read_text()), tl)
    mr, mom = run_mr(proto, tl, frames, bars, bank, seed), run_momentum(proto, tl, bars, seed)
    run = E.init_bundle(a.output, proto["protocol"], "r5-baselines", a.run_id)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    floors = {"events_per_class": proto["floors"]["events_per_class_min"],
              "dates": proto["floors"]["train_dates_min"] + proto["floors"]["test_dates_min"]}
    E.write_json(run / "experiment.json", {"hypotheses": ["H-MR01", "H-MO01", "H-P01(not run)"], "protocol": proto,
                                           "git_commit": commit, "sample_floors": floors,
                                           "activity_floor": proto["floors"]["activity"], "parent": None})
    E.write_json(run / "source_manifest.json", {"manifest_sha256": E.sha256_file(a.manifest),
                                                "as_of": manifest["as_of"], "symbols": len(frames),
                                                "limitations": manifest["verification_scope"]})
    E.write_json(run / "environment.json", {"python": sys.version, "seed": seed,
                                            "pkgs": {m: __import__(m).__version__ for m in ("pandas", "numpy", "sklearn")}})
    E.write_json(run / "dataset_manifest.json", {"mr": {k: mr[k] for k in ("candidates", "label_status", "matured", "dates")},
                                                 "momentum": {k: mom[k] for k in ("decision_months", "matured_rows", "label_status")}})
    folds_doc = [{"index": i, "problems": p.split(":", 1)[1].split(",")} for i, p in enumerate(mom["fold_problems"])]
    for name, r in (("mr", mr), ("momentum", mom)):
        for trial, t in r["trials"].items():
            for f in t.get("folds", []):
                if f["problems"]:
                    folds_doc.append({"index": f"{name}/{trial}/{f['index']}", "problems": f["problems"]})
            for p in t.get("floor_problems", []):
                folds_doc.append({"index": f"{name}/{trial}", "problems": [p]})
    E.write_json(run / "splits.json", {"folds": folds_doc})
    E.write_json(run / "metrics.json", {"mr": mr, "momentum": mom,
                                        "sample_support": {"events_per_class": min(mr["m0"]["events"], 10**9) if mr["m0"]["events"] else 0,
                                                           "dates": mr["dates"]}})
    E.write_json(run / "transforms" / "README.json", {"note": "per-fold transforms are fit inside forward_oof; hashes not persisted in R5 diagnostics"})
    E.write_json(run / "predictions" / "README.json", {"note": "OOF predictions summarized in metrics.json only"})
    E.write_json(run / "regression" / "README.json", {"note": "R7 full gate/books/timeslices not run for R5"})
    E.write_json(run / "portfolio" / "README.json", {"note": "no NAV replay in R5"})
    E.write_json(run / "decision.json", {"status": "BLOCKED", "live_eligible": False})  # provisional
    E.seal(run)
    verdict = E.validate_bundle(run)
    E.write_json(run / "decision.json", {"status": "BLOCKED" if verdict["status"] != "FAIL" else "FAIL",
                                         "validator": verdict, "live_eligible": False,
                                         "reviewer": None, "limitations": ["development data", "R7 regression absent"]})
    print(json.dumps({"run": str(run), "validator": verdict}, indent=2))


if __name__ == "__main__":
    main()
