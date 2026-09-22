"""Reproducible, local-only integrity audit. Never trains or replaces an artifact.

Run: python scripts/audit_data_integrity.py --output docs/audits/integrity.json
The estimator spy records index overlap; it is not a performance benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_agent.config import load_rules
from stock_agent.data.sample_data import make_demo_ohlcv
from stock_agent.features.calibration import preprocess_features_robust
from stock_agent.features.ensemble_model import EnsembleConfig, EnsembleTrainer
from stock_agent.features.feature_engineering_v2 import add_regime_features
from stock_agent.features.indicators import add_indicators
from stock_agent.features.mr_exit import simulate_mr_exit
from stock_agent.features import win_probability as wp


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class FitSpy:
    """Track rows used at fit vs predict without training a real artifact."""
    def fit(self, x, y, **kwargs):
        self.fit_indices = set(x.index)
        return self

    def predict_proba(self, x):
        self.last_predict_indices = set(x.index)
        p = np.full(len(x), 0.7)
        return np.column_stack([1 - p, p])


def audit_causality():
    raw = make_demo_ohlcv("FPT", date(2026, 6, 1), rows=200)
    prepared = add_indicators(raw)
    prefix = add_indicators(raw.iloc[:110])
    cols = prefix.select_dtypes(include="number").columns
    stable = [c for c in cols if np.allclose(prefix[c], prepared[c].iloc[:110], equal_nan=True)]
    fr1 = wp.feature_row(prepared, 100, 1, .5, .01)
    fr2 = wp.feature_row(prefix, 100, 1, .5, .01)

    ret = np.concatenate([np.sin(np.arange(100)) * .002, np.sin(np.arange(100)) * .08])
    base = pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=200), "return_1d": ret})
    a = add_regime_features(base.iloc[:100].copy())
    b = add_regime_features(base)
    regime_changed = int((a.regime_vol_class != b.regime_vol_class.iloc[:100]).sum())

    data = pd.DataFrame({"feature_x": np.arange(200, dtype=float)})
    fitted, bounds = preprocess_features_robust(data, ["feature_x"])
    perturbed = data.copy()
    perturbed.loc[100:, "feature_x"] += 1e6
    fitted2, bounds2 = preprocess_features_robust(perturbed, ["feature_x"])
    first_bounds = preprocess_features_robust(data.iloc[:100], ["feature_x"])[1]

    trainer = EnsembleTrainer(EnsembleConfig(min_train_rows=20, n_splits=3))
    dataset = pd.DataFrame({"signal_date": pd.bdate_range("2025-01-01", periods=80),
                            "feature_x": np.arange(80), "net_t2_win": np.arange(80) % 2,
                            "net_t2_return_pct": np.where(np.arange(80) % 2, 1., -1.)})
    with patch.object(trainer, "_build_lgb", FitSpy), patch.object(trainer, "_build_xgb", FitSpy), \
         patch.object(trainer, "_build_ridge", FitSpy), patch.object(trainer, "_build_cat", FitSpy), \
         patch.object(trainer, "_ridge_proba", lambda model, x: model.predict_proba(x)[:, 1]):
        trainer.train(dataset, ["feature_x"])
    spy = trainer.lgb_model

    panel = pd.DataFrame({"signal_date": np.repeat(pd.bdate_range("2025-01-01", periods=40), 7)})
    panel["exit_date"] = panel.signal_date + pd.offsets.BDay(20)
    folds = []
    for tr, te in TimeSeriesSplit(n_splits=5).split(panel):
        boundary = panel.iloc[te].signal_date.min()
        folds.append({"same_signal_day_in_both": bool(panel.iloc[tr].signal_date.max() == boundary),
                      "train_labels_not_known_at_test_start": int((panel.iloc[tr].exit_date >= boundary).sum())})

    return {"base_indicators_prefix_columns_tested": len(cols), "base_indicators_prefix_columns_stable": len(stable),
            "mr_feature_vector_prefix_stable": all(np.isclose(fr1[c], fr2[c], equal_nan=True) for c in wp.FEATURES),
            "v2_past_regime_classes_changed_after_future_append": regime_changed,
            "ensemble_evaluated_rows": len(spy.last_predict_indices),
            "ensemble_evaluated_rows_seen_by_fit": len(spy.last_predict_indices & spy.fit_indices),
            "winsor_bounds_full": bounds, "winsor_bounds_future_perturbed": bounds2,
            "winsor_bounds_train_only": first_bounds,
            "panel_fold_label_overlap": folds}


def audit_legacy_ledger():
    path = ROOT / "data/pipeline/pending_predictions.jsonl"
    records = read_jsonl(path)
    demo = []; mismatch = []; close_match = []; missing = []
    for n, r in enumerate(records, 1):
        synthetic = float(make_demo_ohlcv(r["symbol"], date.fromisoformat(r["signal_date"])).close.iloc[-1])
        if np.isclose(synthetic, r["entry_reference"], rtol=0, atol=.00011):
            demo.append(n)
        p = ROOT / "data/raw/prices_hist" / f"{r['symbol']}.csv"
        if not p.exists():
            missing.append(n); continue
        df = pd.read_csv(p)
        hit = df[df.date.astype(str).str[:10] == r["signal_date"]]
        if hit.empty:
            missing.append(n); continue
        px = float(hit.iloc[-1].close)
        if abs(float(r["entry_reference"]) / px - 1) > .05:
            mismatch.append(n)
        else:
            close_match.append(n)
    return {"rows": len(records), "unique_symbol_dates": len({(r['symbol'], r['signal_date']) for r in records}),
            "exact_demo_close_rows": len(demo), "demo_line_numbers": demo,
            "market_close_discrepancy_over_5pct": len(mismatch), "close_compatible_rows": len(close_match),
            "missing_price_rows": len(missing), "sha256": digest(path),
            "fields": sorted(set().union(*(r.keys() for r in records)))}


def audit_prices():
    out = {}
    for folder in ("prices", "prices_hist"):
        latest = Counter(); invalid = []; jumps = []; weekends = 0; duplicates = 0; hashes = {}
        for p in sorted((ROOT / "data/raw" / folder).glob("*.csv")):
            df = pd.read_csv(p); hashes[p.name] = digest(p)
            dates = pd.to_datetime(df.date)
            latest[str(dates.max().date())] += 1
            duplicates += int(dates.duplicated().sum()); weekends += int((dates.dt.weekday >= 5).sum())
            price = df[["open", "high", "low", "close"]]
            bad = (price <= 0).any(axis=1) | ~np.isfinite(price).all(axis=1)
            bad |= df.high < df[["open", "close", "low"]].max(axis=1)
            bad |= df.low > df[["open", "close", "high"]].min(axis=1)
            bad |= df.volume.lt(0) | ~np.isfinite(df.volume)
            if bad.any(): invalid.append({"symbol": p.stem, "rows": int(bad.sum()), "examples": df.loc[bad].head(2).to_dict("records")})
            n = int(df.close.pct_change().abs().gt(.15).sum())
            if n: jumps.append({"symbol": p.stem, "jumps_over_15pct": n})
        out[folder] = {"files": len(hashes), "latest_dates": dict(latest), "invalid": invalid,
                       "large_jumps_requiring_corporate_action_review": jumps,
                       "weekend_rows": weekends, "duplicate_dates": duplicates,
                       "snapshot_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}
    return out


def audit_mr():
    path = ROOT / "data/models/win_prob_mr.pkl"
    with path.open("rb") as f:
        art = pickle.load(f)
    meta = {k: v for k, v in art.items() if k not in {"model", "iso"}}
    root = ROOT / "data/raw/prices_hist"
    candidates = []
    for p in sorted(root.glob("*.csv")):
        if p.stem == "VNINDEX": continue
        f = wp._prep(pd.read_csv(p))
        idxs = np.flatnonzero(((f.rsi14 < wp.CAND_RSI_MAX) & (f.close <= f.bb_lower * wp.CAND_BAND_MULT)).to_numpy())
        for i in idxs:
            if i < 90 or i >= len(f)-1: continue
            fr = wp.feature_row(f, int(i), 1., .5, 0.)
            if fr is None: continue
            entry = float(f.at[i+1, "open"])
            if not np.isfinite(entry) or entry <= 0: continue
            stop = entry - wp.STOP_ATR * fr["_atr"]
            target = max(fr["_kijun"], fr["_close"] * 1.01)
            j, px, reason, resolved = simulate_mr_exit(f, i+1, stop, target, wp.MAX_HOLD, wp.T2_LOCK)
            candidates.append({"date": str(f.at[i, "date"]), "exit_date": str(f.at[j, "date"]), "resolved": bool(resolved)})
    df = pd.DataFrame(candidates).sort_values("date").reset_index(drop=True)
    cut = int(len(df)*.85); boundary = df.iloc[cut].date
    train = df.iloc[:cut]
    ledger = read_jsonl(ROOT / "data/pipeline/forward_test.jsonl")
    return {"artifact_sha256": digest(path), "artifact_metadata": meta,
            "reconstruction_note": "Current local history with current code; not the unavailable original training snapshot.",
            "current_candidate_count": len(df), "partial_labels_accepted_by_current_trainer": int((~df.resolved).sum()),
            "calibration_start": boundary, "same_signal_date_in_train_and_cal": bool(train.date.max() == boundary),
            "train_labels_not_known_at_cal_start": int((train.exit_date >= boundary).sum()),
            "forward_rows": len(ledger), "forward_date_range": [min(r['signal_date'] for r in ledger), max(r['signal_date'] for r in ledger)],
            "forward_rows_missing_model_version": sum("model_version" not in r for r in ledger)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"audit_utc": datetime.now(timezone.utc).isoformat(),
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "contract": {"goal": "Audit existing artifacts before any retraining or promotion",
                           "success": "No data available after prediction time may influence features, split preprocessing or evaluation",
                           "unacceptable": "Treat synthetic, retrospectively fitted or unresolved records as live outcomes"}}
    for name, fn in [("causality", audit_causality), ("legacy_ledger", audit_legacy_ledger), ("prices", audit_prices), ("mr", audit_mr)]:
        print(f"Auditing {name}...", flush=True)
        report[name] = fn()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Evidence: {args.output}", flush=True)


if __name__ == "__main__":
    main()
