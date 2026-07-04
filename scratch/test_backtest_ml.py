import pandas as pd
from datetime import date
from stock_agent.config import load_rules
from stock_agent.data.validation import normalize_ohlcv
from stock_agent.features.backtest import BacktestConfig
from stock_agent.features.signal_engine import precompute_signal_frames, score_precomputed_at
from stock_agent.features.ml_models import predict_model_signal, _feature_row_from_signal
from stock_agent.features.ensemble_model import EnsembleTrainer
from stock_agent.constants import PRICE_CACHE_DIR

rules = load_rules()
symbol = "STB"
path = PRICE_CACHE_DIR / f"{symbol}.csv"
df = normalize_ohlcv(pd.read_csv(path))

frames = precompute_signal_frames({symbol: df}, rules)
frame = frames[symbol]

min_rows = int(rules["min_history_rows"])
print(f"Total rows in frame: {len(frame)}")

trainer = EnsembleTrainer.load()

found_any = False
for idx in range(min_rows - 1, len(frame) - 1):
    signal = score_precomputed_at(symbol, frame, idx, rules)
    if signal.decision == "BUY_SETUP":
        found_any = True
        row = _feature_row_from_signal(signal)
        prob, details = trainer.predict_single(row)
        print(f"Date: {frame.loc[idx, 'date']}")
        print(f"  Rule score: {signal.score}")
        print(f"  Ensemble Meta Prob: {prob}")
        print(f"  Base probs: {details['base_probabilities']}")
        print(f"  Agreement: {details['ensemble_agreement']}")

if not found_any:
    print("No BUY_SETUP signals found in rule base.")
