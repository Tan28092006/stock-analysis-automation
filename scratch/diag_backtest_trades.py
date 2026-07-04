import sys
sys.path.insert(0, r"D:\Chungkhoan")

import pandas as pd
from pathlib import Path
from stock_agent.constants import PRICE_CACHE_DIR
from stock_agent.data.validation import normalize_ohlcv
from stock_agent.config import load_rules
from stock_agent.features.backtest import BacktestConfig, run_portfolio_backtest

rules = load_rules()
# Let's force ML override enabled to trace it
rules["ml"]["enabled"] = True
rules["ml"]["override_enabled"] = True

symbols = ["ACB", "BID", "BSR", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG", "LPB", 
           "MBB", "MSN", "MWG", "PLX", "SAB", "SHB", "SSB", "SSI", "STB", "TCB", 
           "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VPL", "VRE"]

frames = {}
for sym in symbols:
    path = PRICE_CACHE_DIR / f"{sym}.csv"
    if path.exists():
        frames[sym] = normalize_ohlcv(pd.read_csv(path))

print("Loaded frames for", list(frames.keys()))

# Let's run a custom tracer by overriding score_precomputed_at or predict_model_signal
# to print candidates and their ML probabilities.
from stock_agent.features import ml_models

orig_predict = ml_models.predict_model_signal

def tracer_predict(symbol, signal, rules):
    res = orig_predict(symbol, signal, rules)
    if signal.decision == "BUY_SETUP":
        print(f"Candidate: {symbol} at {signal.latest_date} | Score: {signal.score} | Prob: {getattr(res, 'probability', None)} | Threshold: {getattr(res, 'threshold', None)} | Passed: {getattr(res, 'passed', None)}")
    return res

ml_models.predict_model_signal = tracer_predict

print("\n--- Running Portfolio Backtest ---")
result = run_portfolio_backtest(frames, rules)
print("\n--- Backtest Results ---")
print("Total Trades:", result.total_trades)
print("Final Equity:", result.final_equity)
