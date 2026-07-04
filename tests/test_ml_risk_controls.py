"""Unit tests for ML risk controls: robust preprocessing and dynamic thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from stock_agent.features.calibration import preprocess_features_robust
from stock_agent.features.ml_models import _predict_ensemble
from stock_agent.schemas import ScanCandidate, RuleEvidence

class TestMLRiskControls:

    def test_preprocess_features_robust_imputation(self):
        # Create a df with NaNs and infs
        df = pd.DataFrame({
            "feature_rsi14": [np.nan, 45.0, np.inf, np.nan, 55.0],
            "feature_volume_ratio_20": [1.5, np.nan, -np.inf, 1.2, np.nan],
            "feature_cross_rsi14_rank": [np.nan, np.nan, 0.4, np.nan, 0.6],
            "feature_return_1d": [0.02, np.nan, -0.01, np.nan, np.nan],
            "rule_ema_trend": [1, 0, 1, 0, 1]
        })
        
        feature_cols = [
            "feature_rsi14",
            "feature_volume_ratio_20",
            "feature_cross_rsi14_rank",
            "feature_return_1d",
            "rule_ema_trend"
        ]
        
        processed, bounds = preprocess_features_robust(df, feature_cols)
        
        # In rule_ema_trend, no NaN -> should be unchanged
        assert processed["rule_ema_trend"].tolist() == [1, 0, 1, 0, 1]
        
        # check imputation
        # RSI: neutral is 50. Row 0: NaN -> filled with 50. Row 2: inf -> ffilled to 45. Row 3: NaN -> ffilled to 45.
        assert processed["feature_rsi14"].iloc[0] == 50.0
        assert processed["feature_rsi14"].iloc[2] == 45.0
        assert processed["feature_rsi14"].iloc[3] == 45.0
        
        # Volume ratio: neutral is 1.0. Row 1: NaN -> ffilled to 1.5. Row 2: -inf -> ffilled to 1.5. Row 4: NaN -> ffilled to 1.2.
        assert processed["feature_volume_ratio_20"].iloc[1] == 1.5
        assert processed["feature_volume_ratio_20"].iloc[2] == 1.5
        assert processed["feature_volume_ratio_20"].iloc[4] == 1.2
        
        # Cross rsi rank: neutral is 0.5. Row 0, 1: NaN -> filled with 0.5. Row 3: NaN -> ffilled to 0.4.
        assert processed["feature_cross_rsi14_rank"].iloc[0] == 0.5
        assert processed["feature_cross_rsi14_rank"].iloc[1] == 0.5
        assert processed["feature_cross_rsi14_rank"].iloc[3] == 0.4
        
        # Return 1d: neutral is 0.0. Row 1: NaN -> ffilled to 0.02. Row 3: NaN -> ffilled to -0.01.
        assert processed["feature_return_1d"].iloc[1] == 0.02
        assert processed["feature_return_1d"].iloc[3] == -0.01

    def test_preprocess_features_robust_winsorization(self):
        # Create a df with an outlier
        df = pd.DataFrame({
            "feature_return_1d": [-0.5, 0.01, 0.02, 0.015, 0.6],  # Outliers: -0.5, 0.6
        })
        feature_cols = ["feature_return_1d"]
        
        # Fit bounds
        processed, bounds = preprocess_features_robust(df, feature_cols)
        
        assert "feature_return_1d" in bounds
        lower, upper = bounds["feature_return_1d"]
        # Outliers should be clipped
        assert processed["feature_return_1d"].iloc[0] > -0.5
        assert processed["feature_return_1d"].iloc[4] < 0.6
        assert processed["feature_return_1d"].iloc[1] == 0.01
        
        # Apply same bounds on a new df
        new_df = pd.DataFrame({
            "feature_return_1d": [-1.0, 0.01, 1.0]
        })
        processed_new, _ = preprocess_features_robust(new_df, feature_cols, winsorize_bounds=bounds)
        assert processed_new["feature_return_1d"].iloc[0] == lower
        assert processed_new["feature_return_1d"].iloc[2] == upper

    def test_dynamic_threshold_adjustments(self):
        # Mock signal with different regimes and volatilities
        class MockSignal:
            def __init__(self, features, evidence):
                self.features = features
                self.evidence = evidence
                self.decision = "BUY_SETUP"
                self.score = 60
                
        # Setup mock rules
        rules = {"ml": {"enabled": True, "model_family": "ensemble", "probability_threshold": 0.55}}
        ml_rules = rules["ml"]
        
        # We need a mock trainer registered or cached
        from unittest.mock import MagicMock
        import stock_agent.features.ml_models as ml_mod
        
        mock_trainer = MagicMock()
        mock_trainer.predict_single.return_value = (0.6, {
            "ensemble_agreement": 0.75,
            "shap_top5": {},
            "base_probabilities": {}
        })
        mock_trainer.trained_at = "2026-06-08T12:00:00"
        
        # Inject mock trainer
        ml_mod._cached_ensemble_trainer = mock_trainer
        
        # Test Case 1: Normal regime and typical volatility
        signal_normal = MockSignal(
            features={"atr_pct": 0.025, "regime_trend": 1.0, "regime_vol_class": 1.0},
            evidence=[RuleEvidence(evidence_id="rule:ema_trend", name="EMA Trend", passed=True, points=10.0, detail="Passed")]
        )
        res_normal = _predict_ensemble("FPT", signal_normal, rules, ml_rules)
        # Expected: base 0.55 + vol_adjust(0) + regime_adjust(0) = 0.55
        assert abs(res_normal.threshold - 0.55) < 0.001
        assert res_normal.passed is True  # prob 0.6 >= threshold 0.55
        
        # Test Case 2: Bear market and high volatility
        signal_bear_vol = MockSignal(
            features={"atr_pct": 0.045, "regime_trend": -1.0, "regime_vol_class": 2.0},
            evidence=[]
        )
        res_bear_vol = _predict_ensemble("FPT", signal_bear_vol, rules, ml_rules)
        # Expected: base 0.55
        # vol_adjust: (0.045 - 0.025) * 1.5 = 0.03 (clamped to 0.04) -> +0.03
        # regime_trend bear -> +0.08
        # regime_vol high -> +0.04
        # Total threshold = 0.55 + 0.03 + 0.08 + 0.04 = 0.70
        assert abs(res_bear_vol.threshold - 0.70) < 0.005
        assert res_bear_vol.passed is False  # prob 0.6 < threshold 0.70
        
        # Clear cache
        ml_mod.clear_ensemble_cache()
