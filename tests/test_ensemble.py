"""Tests for the stacking ensemble model and feature engineering v2."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Feature Engineering v2 Tests
# ---------------------------------------------------------------------------

class TestFeatureEngineeringV2:

    def _make_symbol_frames(self, n_symbols=5, n_days=60):
        """Create synthetic symbol DataFrames with indicator columns."""
        frames = {}
        base_date = date(2026, 1, 5)
        for i in range(n_symbols):
            symbol = f"SYM{i}"
            dates = [base_date + timedelta(days=d) for d in range(n_days)]
            close = 20.0 + i * 5 + np.cumsum(np.random.randn(n_days) * 0.3)
            df = pd.DataFrame({
                "date": dates,
                "open": close * 0.99,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": np.random.randint(100_000, 1_000_000, n_days),
                "rsi14": 30 + np.random.rand(n_days) * 40,
                "volume_ratio_20": 0.5 + np.random.rand(n_days) * 2,
                "return_1d": np.random.randn(n_days) * 0.02,
                "adx14": 10 + np.random.rand(n_days) * 30,
                "macd_hist": np.random.randn(n_days) * 0.5,
            })
            frames[symbol] = df
        return frames

    def test_cross_sectional_features_shape(self):
        from stock_agent.features.feature_engineering_v2 import add_cross_sectional_features

        frames = self._make_symbol_frames(n_symbols=5, n_days=30)
        result = add_cross_sectional_features(frames)

        assert len(result) == 5
        for symbol, df in result.items():
            assert "cross_rsi14_rank" in df.columns
            assert "cross_volume_ratio_20_rank" in df.columns
            assert "cross_return_1d_rank" in df.columns
            assert "vn30_avg_return_1d" in df.columns
            assert "vn30_advance_decline_ratio" in df.columns
            # Ranks should be between 0 and 1
            assert df["cross_rsi14_rank"].dropna().between(0, 1).all()

    def test_regime_features(self):
        from stock_agent.features.feature_engineering_v2 import add_regime_features

        df = self._make_symbol_frames(n_symbols=1, n_days=60)["SYM0"]
        result = add_regime_features(df)

        assert "regime_volatility_20" in result.columns
        assert "regime_vol_class" in result.columns
        assert "regime_trend" in result.columns
        assert "days_since_regime_change" in result.columns
        assert result["regime_vol_class"].isin([0, 1, 2]).all()
        assert result["regime_trend"].isin([-1, 0, 1]).all()

    def test_temporal_features(self):
        from stock_agent.features.feature_engineering_v2 import add_temporal_features

        df = self._make_symbol_frames(n_symbols=1, n_days=30)["SYM0"]
        result = add_temporal_features(df)

        assert "day_of_week_sin" in result.columns
        assert "day_of_week_cos" in result.columns
        assert "month_sin" in result.columns
        assert "days_to_expiry" in result.columns
        assert all(d >= 0 for d in result["days_to_expiry"])

    def test_target_engineering(self):
        from stock_agent.features.feature_engineering_v2 import engineer_targets

        df = pd.DataFrame({
            "net_return_pct": [-5.0, -1.0, 0.5, 3.0, 0.0],
            "atr_pct": [2.0, 1.5, 1.0, 2.0, 1.0],
            "vn30_avg_return_1d": [0.1, -0.2, 0.0, 0.5, -0.1],
        })
        result = engineer_targets(df, net_return_col="net_return_pct")

        assert result["label_binary"].tolist() == [0, 0, 1, 1, 0]
        assert result["label_rr_category"].tolist() == [0, 1, 2, 3, 2]  # 0.0 is small_win (>= 0)
        assert "label_risk_adjusted" in result.columns
        assert "label_excess_return" in result.columns

    def test_v2_feature_columns_list(self):
        from stock_agent.features.feature_engineering_v2 import get_v2_feature_columns

        cols = get_v2_feature_columns()
        assert len(cols) >= 15
        assert "cross_rsi14_rank" in cols
        assert "regime_trend" in cols
        assert "days_to_expiry" in cols


# ---------------------------------------------------------------------------
# Ensemble Model Tests
# ---------------------------------------------------------------------------

class TestEnsembleModel:

    def _make_dataset(self, n_rows=500):
        """Create a synthetic labeled dataset for ensemble training."""
        np.random.seed(42)
        dates = pd.date_range("2025-06-01", periods=n_rows, freq="B")
        n_features = 15

        # Create features with some signal
        features = np.random.randn(n_rows, n_features)
        # Inject real signal into first 3 features
        signal = features[:, 0] * 0.3 + features[:, 1] * 0.2 - features[:, 2] * 0.15
        noise = np.random.randn(n_rows) * 0.5
        label_prob = 1.0 / (1.0 + np.exp(-(signal + noise)))
        labels = (label_prob > 0.5).astype(int)
        net_returns = np.where(labels, np.random.uniform(0.1, 3.0, n_rows), np.random.uniform(-3.0, -0.1, n_rows))

        feature_cols = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(features, columns=feature_cols)
        df["signal_date"] = dates[:n_rows]
        df["symbol"] = np.random.choice(["FPT", "HPG", "VCB", "TCB", "MBB"], n_rows)
        df["net_t2_win"] = labels
        df["net_t2_return_pct"] = net_returns
        df["decision"] = np.where(labels, "BUY_SETUP", "REJECT")
        return df, feature_cols

    def test_ensemble_train(self):
        from stock_agent.features.ensemble_model import EnsembleTrainer, EnsembleConfig

        dataset, feature_cols = self._make_dataset(n_rows=500)
        config = EnsembleConfig(min_train_rows=100, n_splits=3)
        trainer = EnsembleTrainer(config=config)

        result = trainer.train(dataset, feature_cols=feature_cols)

        assert result["status"] == "trained"
        assert result["train_rows"] == 500
        assert trainer.lgb_model is not None
        assert trainer.xgb_model is not None
        assert trainer.ridge_pipeline is not None
        assert trainer.metrics.get("test_win_rate", 0) > 0

    def test_ensemble_predict(self):
        from stock_agent.features.ensemble_model import EnsembleTrainer, EnsembleConfig

        dataset, feature_cols = self._make_dataset(n_rows=500)
        config = EnsembleConfig(min_train_rows=100, n_splits=3)
        trainer = EnsembleTrainer(config=config)
        trainer.train(dataset, feature_cols=feature_cols)

        # Predict on last 10 rows
        test_data = dataset.tail(10)
        proba, shap_vals = trainer.predict(test_data, return_shap=True)

        assert len(proba) == 10
        assert all(0 <= p <= 1 for p in proba)
        assert shap_vals is not None
        assert len(shap_vals) == 10

    def test_ensemble_predict_single(self):
        from stock_agent.features.ensemble_model import EnsembleTrainer, EnsembleConfig

        dataset, feature_cols = self._make_dataset(n_rows=400)
        config = EnsembleConfig(min_train_rows=100, n_splits=3)
        trainer = EnsembleTrainer(config=config)
        trainer.train(dataset, feature_cols=feature_cols)

        row = {col: float(dataset.iloc[-1][col]) for col in feature_cols}
        proba, details = trainer.predict_single(row)

        assert 0 <= proba <= 1
        assert "ensemble_agreement" in details
        assert "base_probabilities" in details
        assert "shap_top5" in details
        assert 0 <= details["ensemble_agreement"] <= 1

    def test_ensemble_daily_update(self):
        from stock_agent.features.ensemble_model import EnsembleTrainer, EnsembleConfig

        dataset, feature_cols = self._make_dataset(n_rows=500)
        config = EnsembleConfig(min_train_rows=100, n_splits=3, incremental_trees=5)
        trainer = EnsembleTrainer(config=config)

        # Initial train on first 400 rows
        trainer.train(dataset.iloc[:400], feature_cols=feature_cols)
        initial_trees = trainer.lgb_model.n_estimators_

        # Incremental update with all 500 rows
        result = trainer.daily_update(dataset)

        assert result["status"] == "updated"
        assert result["mode"] == "incremental"

    def test_ensemble_insufficient_data(self):
        from stock_agent.features.ensemble_model import EnsembleTrainer, EnsembleConfig

        dataset, feature_cols = self._make_dataset(n_rows=50)
        config = EnsembleConfig(min_train_rows=100)
        trainer = EnsembleTrainer(config=config)

        result = trainer.train(dataset, feature_cols=feature_cols)
        assert result["status"] == "insufficient_data"

    def test_drift_detection(self):
        from stock_agent.features.ensemble_model import EnsembleTrainer, EnsembleConfig

        config = EnsembleConfig(drift_window=10, drift_warning_drop_pct=15, drift_disable_drop_pct=25)
        trainer = EnsembleTrainer(config=config)
        trainer.metrics = {"test_win_rate": 60.0}

        # Healthy: mostly correct predictions
        healthy_outcomes = [
            {"predicted_win": True, "actual_win": True} for _ in range(8)
        ] + [{"predicted_win": True, "actual_win": False} for _ in range(2)]
        status = trainer.detect_drift(healthy_outcomes)
        assert status.status == "healthy"

        # Degraded: mostly wrong
        bad_outcomes = [
            {"predicted_win": True, "actual_win": False} for _ in range(9)
        ] + [{"predicted_win": True, "actual_win": True}]
        status = trainer.detect_drift(bad_outcomes)
        assert status.status in ("warning", "degraded")

    def test_ensemble_save_load(self):
        from stock_agent.features.ensemble_model import EnsembleTrainer, EnsembleConfig
        import tempfile, shutil

        dataset, feature_cols = self._make_dataset(n_rows=300)
        config = EnsembleConfig(min_train_rows=100, n_splits=3)
        trainer = EnsembleTrainer(config=config)
        trainer.train(dataset, feature_cols=feature_cols)

        # Use workspace-local temp dir to avoid Windows permission issues
        test_dir = Path("data/test_tmp_ensemble")
        test_dir.mkdir(parents=True, exist_ok=True)
        try:
            save_path = test_dir / "test_ensemble.pkl"
            trainer.save(save_path)
            assert save_path.exists()

            loaded = EnsembleTrainer.load(save_path)
            assert loaded.feature_columns == trainer.feature_columns
            assert loaded.train_rows == trainer.train_rows

            # Verify predictions match
            test_row = {col: float(dataset.iloc[-1][col]) for col in feature_cols}
            p1, _ = trainer.predict_single(test_row)
            p2, _ = loaded.predict_single(test_row)
            assert abs(p1 - p2) < 0.001
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# ML Override Tests
# ---------------------------------------------------------------------------

class TestMLOverride:

    def test_override_config_in_rules(self):
        from stock_agent.config import load_rules
        rules = load_rules()
        ml = rules.get("ml", {})
        assert ml.get("model_family") == "ensemble"
        assert ml.get("override_enabled") is True
        assert "override_loss_threshold" in ml
        assert "ensemble" in ml

    def test_schema_has_ensemble_fields(self):
        from stock_agent.schemas import ModelSignal
        signal = ModelSignal(
            status="available",
            ensemble_agreement=0.67,
            shap_top_features=[{"feature_rsi14": 0.15}],
            model_vintage="2026-05-27T10:00:00",
            drift_status="healthy",
            base_probabilities={"lightgbm": 0.65, "xgboost": 0.60, "ridge": 0.55},
        )
        assert signal.ensemble_agreement == 0.67
        assert signal.drift_status == "healthy"
        assert len(signal.base_probabilities) == 3
