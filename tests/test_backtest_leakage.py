import unittest
from datetime import date
from unittest.mock import patch
import numpy as np
import pandas as pd

from stock_agent.config import load_rules
from stock_agent.data.sample_data import make_demo_ohlcv
from stock_agent.features.backtest import (
    BacktestConfig,
    run_backtest,
)
from stock_agent.features.signal_engine import (
    SignalOutput,
    prepare_signal_frame,
    score_precomputed_at,
)
from stock_agent.schemas import RiskPlan


class BacktestLeakageTests(unittest.TestCase):
    def setUp(self):
        self.rules = load_rules()
        # Ensure ML is disabled by default for these tests to avoid external model calls
        if "ml" in self.rules:
            self.rules["ml"]["enabled"] = False
        # Set min_history_rows to 5 to allow processing on small dataframes
        self.rules["min_history_rows"] = 5

    def test_indicator_precomputation_causality(self):
        """
        Test 1: Indicator Precomputation Causal Check
        Computes indicators on a complete DataFrame and on a sliced prefix of the same DataFrame.
        Asserts that computed indicator values on the prefix are identical to the corresponding
        portion of the complete DataFrame.
        """
        df = make_demo_ohlcv("HPG", date(2026, 5, 26), rows=100)
        
        # Compute indicators on full dataframe
        df_full = prepare_signal_frame(df, rules=self.rules)
        
        # Compute indicators on sliced dataframe (first 50 rows)
        slice_len = 50
        df_sliced = prepare_signal_frame(df.iloc[:slice_len].copy(), rules=self.rules)
        
        # Identify indicator columns (exclude base OHLCV + symbol + date columns)
        non_indicator_cols = {"symbol", "date", "open", "high", "low", "close", "volume"}
        indicator_cols = [c for c in df_sliced.columns if c not in non_indicator_cols]
        
        self.assertTrue(len(indicator_cols) > 0, "No indicators were computed!")
        
        for col in indicator_cols:
            series_full = df_full[col].iloc[:slice_len]
            series_sliced = df_sliced[col]
            
            # Use numpy testing to assert closeness, handling NaNs by filling them with a constant
            np.testing.assert_allclose(
                series_sliced.fillna(-9999.0).values,
                series_full.fillna(-9999.0).values,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Indicator '{col}' leaks future data (look-ahead bias detected)."
            )

    def test_signal_decision_causality(self):
        """
        Test 2: Signal Decision Causal Check
        Generates a signal at index i.
        Randomizes/corrupts all data at index i+1 onwards.
        Asserts that the signal decision, score, risk plan, and warnings at index i remain unchanged.
        """
        df = make_demo_ohlcv("HPG", date(2026, 5, 26), rows=100)
        df_prepared = prepare_signal_frame(df, rules=self.rules)
        
        # Choose a middle index to score
        idx = 50
        
        # Score on original precomputed features
        original_signal = score_precomputed_at("HPG", df_prepared, idx, self.rules)
        
        # Create a modified copy of the features, corrupting everything after idx
        df_modified = df_prepared.copy()
        for col in df_modified.columns:
            if col not in ("symbol", "date"):
                if pd.api.types.is_numeric_dtype(df_modified[col]):
                    df_modified.loc[idx + 1:, col] = 999999.9
                else:
                    df_modified.loc[idx + 1:, col] = None
        
        # Score again on modified features
        modified_signal = score_precomputed_at("HPG", df_modified, idx, self.rules)
        
        # Verify that the signal decision details are identical
        self.assertEqual(original_signal.decision, modified_signal.decision)
        self.assertEqual(original_signal.score, modified_signal.score)
        self.assertEqual(original_signal.latest_close, modified_signal.latest_close)
        self.assertEqual(original_signal.latest_date, modified_signal.latest_date)
        self.assertEqual(original_signal.warnings, modified_signal.warnings)
        
        if original_signal.risk_plan is not None:
            self.assertIsNotNone(modified_signal.risk_plan)
            self.assertEqual(original_signal.risk_plan.entry_reference, modified_signal.risk_plan.entry_reference)
            self.assertEqual(original_signal.risk_plan.stop_loss, modified_signal.risk_plan.stop_loss)
            self.assertEqual(original_signal.risk_plan.take_profit_1, modified_signal.risk_plan.take_profit_1)
        else:
            self.assertIsNone(modified_signal.risk_plan)

    @patch("stock_agent.features.backtest.score_precomputed_at")
    def test_trade_entry_execution_causality(self, mock_score):
        """
        Test 3: Trade Entry Execution Check
        Asserts that trade entry price at index i+1 (for a signal at i) is only determined by
        the open of i+1 and close of i (for price limits), and remains unchanged if future dates
        (i+2 onwards) are modified.
        """
        df = make_demo_ohlcv("HPG", date(2026, 5, 26), rows=20)
        
        # We want to force a BUY_SETUP at index 5
        signal_idx = 5
        entry_idx = signal_idx + 1
        
        # Mock signal response
        mock_risk_plan = RiskPlan(
            entry_reference=100.0,
            stop_loss=90.0,
            take_profit_1=110.0,
            take_profit_2=120.0,
            stop_loss_pct=10.0,
            reward_risk=1.0,
            holding_period_days=2
        )
        mock_signal = SignalOutput(
            decision="BUY_SETUP",
            score=80.0,
            confidence_delta=0.05,
            latest_close=100.0,
            latest_date=str(df.loc[signal_idx, "date"]),
            risk_plan=mock_risk_plan,
            evidence=[],
            warnings=[],
            features={}
        )
        
        def side_effect(symbol, data, i, rules):
            if i == signal_idx:
                return mock_signal
            return SignalOutput("REJECT", 0.0, 0.0, float(data.loc[i, "close"]), str(data.loc[i, "date"]), None, [], [], {})
            
        mock_score.side_effect = side_effect
        
        # Set specific prices at signal_idx and entry_idx
        df.loc[signal_idx, "close"] = 100.0
        df.loc[entry_idx, "open"] = 102.0  # open price for entry
        
        # Run original backtest
        result_original = run_backtest("HPG", df, self.rules, start=df["date"].iloc[0], end=df["date"].iloc[-1])
        self.assertEqual(len(result_original.trades), 1)
        original_trade = result_original.trades[0]
        
        # Modify future data (index entry_idx + 1 and onwards)
        df_modified = df.copy()
        df_modified.loc[entry_idx + 1:, ["open", "high", "low", "close", "volume"]] = 9999.0
        
        # Run backtest on modified future data
        result_modified = run_backtest("HPG", df_modified, self.rules, start=df["date"].iloc[0], end=df["date"].iloc[-1])
        self.assertEqual(len(result_modified.trades), 1)
        modified_trade = result_modified.trades[0]
        
        # Assert that signal date, entry date, and entry price are unchanged
        self.assertEqual(original_trade.signal_date, modified_trade.signal_date)
        self.assertEqual(original_trade.entry_date, modified_trade.entry_date)
        self.assertEqual(original_trade.entry_price, modified_trade.entry_price)

    @patch("stock_agent.features.backtest.score_precomputed_at")
    def test_trade_exit_execution_and_t2_settlement(self, mock_score):
        """
        Test 4: Trade Exit Execution Check & T+2 Settlement Rule
        - Asserts that trade exit price and date are only determined by price paths up to the exit date.
        - Verifies that the T+2 settlement rule (no exits on T+0 or T+1 relative to entry date)
          is strictly honored, even if stop loss or take profit conditions are met on those days.
        """
        df = make_demo_ohlcv("HPG", date(2026, 5, 26), rows=20)
        
        # Force a BUY_SETUP at index 5
        signal_idx = 5
        entry_idx = signal_idx + 1   # T+0
        t1_idx = entry_idx + 1       # T+1
        t2_idx = entry_idx + 2       # T+2
        
        # Stop loss is 95.0, Take profit is 105.0
        mock_risk_plan = RiskPlan(
            entry_reference=100.0,
            stop_loss=95.0,
            take_profit_1=105.0,
            take_profit_2=120.0,
            stop_loss_pct=5.0,
            reward_risk=1.0,
            holding_period_days=2
        )
        mock_signal = SignalOutput(
            decision="BUY_SETUP",
            score=80.0,
            confidence_delta=0.05,
            latest_close=100.0,
            latest_date=str(df.loc[signal_idx, "date"]),
            risk_plan=mock_risk_plan,
            evidence=[],
            warnings=[],
            features={}
        )
        
        def side_effect(symbol, data, i, rules):
            if i == signal_idx:
                return mock_signal
            return SignalOutput("REJECT", 0.0, 0.0, float(data.loc[i, "close"]), str(data.loc[i, "date"]), None, [], [], {})
            
        mock_score.side_effect = side_effect
        
        # Construct scenario: Low prices at T+0 and T+1 drop way below stop loss (e.g. 80.0),
        # but because of T+2 settlement, no exit should be triggered on T+0 or T+1.
        # On T+2, low is above stop loss, but close is 96.0. It should exit on T2 close.
        df.loc[signal_idx, "close"] = 100.0
        df.loc[entry_idx, ["open", "high", "low", "close"]] = [100.0, 101.0, 80.0, 100.0]
        df.loc[t1_idx, ["open", "high", "low", "close"]] = [100.0, 101.0, 80.0, 100.0]
        df.loc[t2_idx, ["open", "high", "low", "close"]] = [100.0, 102.0, 97.0, 96.0]
        
        result = run_backtest("HPG", df, self.rules, start=df["date"].iloc[0], end=df["date"].iloc[-1])
        self.assertEqual(len(result.trades), 1)
        trade = result.trades[0]
        
        # Verify exit occurs on T+2 (t2_idx) and not T+0 or T+1
        expected_exit_date = str(df.loc[t2_idx, "date"])
        self.assertEqual(trade.exit_date, expected_exit_date)
        self.assertEqual(trade.exit_reason, "T2_CLOSE")
        self.assertEqual(trade.exit_price, 96.0)
        
        # Construct Scenario 2: Stop loss hit on T+2
        df_sl = df.copy()
        df_sl.loc[t2_idx, "low"] = 94.0    # Drops below stop loss (95.0) on T+2
        
        result_sl = run_backtest("HPG", df_sl, self.rules, start=df["date"].iloc[0], end=df["date"].iloc[-1])
        self.assertEqual(len(result_sl.trades), 1)
        trade_sl = result_sl.trades[0]
        
        self.assertEqual(trade_sl.exit_date, expected_exit_date)
        self.assertEqual(trade_sl.exit_reason, "STOP_LOSS")
        self.assertEqual(trade_sl.exit_price, 95.0)
        
        # Assert that modifying data after the actual exit index does not affect the trade record
        df_sl_modified = df_sl.copy()
        df_sl_modified.loc[t2_idx + 1:, ["open", "high", "low", "close", "volume"]] = 9999.0
        
        result_sl_modified = run_backtest("HPG", df_sl_modified, self.rules, start=df["date"].iloc[0], end=df["date"].iloc[-1])
        trade_sl_modified = result_sl_modified.trades[0]
        
        self.assertEqual(trade_sl.exit_date, trade_sl_modified.exit_date)
        self.assertEqual(trade_sl.exit_reason, trade_sl_modified.exit_reason)
        self.assertEqual(trade_sl.exit_price, trade_sl_modified.exit_price)


if __name__ == "__main__":
    unittest.main()
