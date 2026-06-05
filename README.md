# 📈 VN30 T+2 Stock Agent & Algo-Trading Intelligence Platform

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML%20Stack-Scikit--Learn%20%7C%20XGBoost%20%7C%20LightGBM-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Hyperparameter Optimization](https://img.shields.io/badge/Optimization-Optuna%20%7C%20Bayesian-4B0082?style=for-the-badge&logo=opsgenie&logoColor=white)](https://optuna.org/)
[![Data APIs](https://img.shields.io/badge/Data-vnstock%20%7C%20yfinance-00C4CC?style=for-the-badge&logo=databricks&logoColor=white)](https://github.com/thinh-vu/vnstock)
[![AI Agent](https://img.shields.io/badge/AI--Agent-Evidence--First%20LLM-6366F1?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)

An advanced, end-to-end algorithmic swing trading platform focused on the **VN30 index** (Vietnam's top 30 blue-chip stocks). Designed specifically for a **T+2 holding horizon**, this platform integrates a data validation pipeline, multi-indicator scoring engine, machine learning classifiers, Bayesian optimizer, walk-forward simulator, and an evidence-based LLM summary report engine, all managed through a premium web dashboard and command-line interface.

> [!IMPORTANT]
> **Financial Disclaimer:** This platform is an educational MVP designed to produce auditable, evidence-backed trade candidates. It is **not** financial advice, nor does it place automatic live orders.

---

## 🏛️ System Architecture

The system operates on an **Evidence-First design pattern**, separating decision-making rules (deterministic and immutable) from machine learning (advisory and probabilistic) and natural language generation (strictly summaries of structured evidence).

```
               [ Data Sources: vnstock / yfinance / CSV / Demo ]
                                      │
                                      ▼
                        [ Data Validation & Cross-Check ]
                                      │
                                      ▼
                      [ Feature & Signal Scoring Engine ]
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
      [ ML Advisory Ensemble ]               [ Deterministic Rule Guards ]
      - Logistic / Random Forest             - Trend, Momentum, Support
      - XGBoost / LightGBM                   - Vol, Liquidity, Stop-Loss Limits
                   └──────────────────┬──────────────────┘
                                      ▼
                        [ Evidence Bundle & Consensus ]
                                      │
                   ┌──────────────────┼──────────────────┐
                   ▼                  ▼                  ▼
          [ HRP Allocation ]  [ Web Dashboard UI ]  [ LLM Post-Validator ]
```

---

## ✨ Key Capabilities

### 1. Robust Data & Audit Layer
* **Source Chain & Fallback:** Seamlessly resolves price/volume data across local CSV, `vnstock`, and `yfinance`.
* **Auditable Provenance:** Every OHLCV row goes through duplicate, validation, and freshness checks.
* **Cross-Source Tolerance Checks:** Ensures data consistency across multiple APIs before trading.

### 2. Multi-Indicator T+2 Signal Engine
* **Indicators:** Compiles technical features including Ichimoku Kinko Hyo, EMA crossovers, MACD momentum, RSI, Bollinger Bands, Average True Range (ATR), Volume Breakouts, OBV, and VWAP.
* **Point-Based Rules:** Generates a structured points table from technical indicators. Stocks with scores `≥ 55` trigger a `BUY_SETUP`, while scores `≥ 39` are placed on the `WATCH` list.

### 3. Machine Learning Advisory Ensemble
* **Algorithms:** Employs Logistic Regression, Random Forest, XGBoost, and LightGBM to estimate transition probabilities (T+2 positive returns).
* **Consensus Voting:** Uses out-of-sample probability thresholds to filter out low-conviction technical setups.

### 4. Bayesian Optimizer (Optuna)
* Fine-tunes rule scoring configurations, thresholds, and stop-loss boundaries via Bayesian hyperparameter search.
* Optimizes for Sharpe ratio, Win Rate, and Drawdown constraints over historical backtests.

### 5. Walk-Forward Backtest & Robustness Checks
* **Simulation Layer:** Emulates realistic Vietnamese market constraints: commissions (0.15%), sell taxes (0.1%), slippage (0.1%), strict T+2 settlement cycles, lot sizes (100 shares), and daily price bands (±7%).
* **Robustness Suite:** Integrates parameter sensitivity analysis, out-of-sample window verification, Monte Carlo path simulations, and adversarial stress testing.

### 6. Evidence-First LLM Summarizer
* Passes the structured JSON evidence directly to NVIDIA/Groq LLM models.
* **Post-Validation Guardrail:** Parses and validates the generated LLM text. If the model hallucinations or contradicts the deterministic rules (e.g., changes a "Reject" to "Buy" or cites fake evidence IDs), the report is immediately discarded.

---

## 📂 Project Directory Structure

```
chungkhoan/
├── configs/                     # System configs
│   ├── universe_vn30.json       # Stock list & tickers
│   └── rules_t2.json            # Technical & ML scoring config
├── data/                        # Project data (ignored from Git, except templates)
│   ├── raw/prices/              # Local raw OHLCV CSV cache
│   ├── interim/                 # Cleaned data
│   ├── features/                # ML feature snapshots
│   ├── processed/               # scan and portfolio cache files
│   ├── models/                  # Trained classifier binaries
│   └── reports/                 # PDF and text advisory files
├── logs/                        # Diagnostics and errors logs
├── stock_agent/                 # Main source package
│   ├── agents/                  # Scanning & reporting orchestrators
│   ├── ai/                      # LLM reporting & post-validators
│   ├── data/                    # Storage and validation repository
│   ├── features/                # Signal engine, Backtesting, Optuna & ML
│   ├── portfolio/               # Portfolio store & P&L tracker
│   ├── app.py                   # Custom Web API & HTTP server
│   ├── cli.py                   # Command-line interface
│   ├── config.py                # Config parser
│   ├── constants.py             # System paths
│   └── schemas.py               # Data types and models
├── tests/                       # Automated unit/smoke tests
├── web/                         # Dashboard Frontend
│   └── index.html               # Single-page glassmorphic UI
├── .env.example                 # API Keys template
├── requirements.txt             # Python packages
└── run_pipeline.bat             # Shortcut running script
```

---

## ⚡ Quick Start

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Copy `.env.example` to `.env` and configure your API keys:
```bash
cp .env.example .env
```
*Specify your API endpoint keys (e.g. `NVIDIA_API_KEY` or `GROQ_API_KEY`) for AI reporting.*

### 3. Command Line Interface (CLI)

The system exposes a powerful CLI (`stock_agent/cli.py`) for scanning, portfolio tracking, and ML backtesting:

#### 🔍 Run Scans
```bash
# Run a deterministic smoke-test scan using simulated demo data
python -m stock_agent.cli scan --demo

# Run a live scan fetching real-time data for VN30 from APIs
python -m stock_agent.cli scan
```

#### 💼 Manage Portfolio
```bash
# Add a stock position (FPT, bought at 105,000 VND, quantity 100)
python -m stock_agent.cli portfolio add --symbol FPT --buy-price 105000 --quantity 100

# Check real-time P&L status
python -m stock_agent.cli portfolio pnl
```

#### 🧠 Train ML Models
```bash
# Train the classifiers for FPT and HPG using Logistic Regression, Random Forest, and GBDTs
python -m stock_agent.cli train --symbols FPT,HPG --families logistic,random_forest,xgboost

# View current model training status and metrics
python -m stock_agent.cli model-status
```

#### 📊 Backtesting & Robustness Simulation
```bash
# Run a single stock T+2 backtest
python -m stock_agent.cli backtest --symbol FPT

# Backtest all stocks in the universe
python -m stock_agent.cli backtest --all

# Run Monte Carlo and parameter sensitivity robustness simulations
python -m stock_agent.cli robustness --symbols FPT,HPG --monte-carlo-runs 200
```

#### 🤖 Generate AI Advisory Reports
```bash
# Synthesize latest report with LLM and post-validate decisions
python -m stock_agent.cli ai report
```

### 4. Running the Web Dashboard

Start the integrated web server (using the Python Standard Library HTTP stack):
```bash
python -m stock_agent.app --host 127.0.0.1 --port 8000
```
Then navigate to: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

The dashboard provides a premium, responsive UI featuring:
* Real-time scan candidates filtered by Decision state (`BUY_SETUP`, `WATCH`, `REJECT`).
* Deep stock detail audits exposing individual indicator points and source confidence.
* Visual portfolio position tracking with real-time P&L valuation.
* Interactive controls to trigger scans, train models, backtest, and generate reports.

---

## 📈 Deep Research Framework

Our research workflow evaluates strategies using a comprehensive set of mathematical metrics:

| Metric | Target Value | Description |
|---|---|---|
| **Win Rate** | `> 55.0%` | Number of profitable trades / total trades. |
| **Sharpe Ratio** | `> 1.2` | Excess return per unit of volatility. |
| **Sortino Ratio** | `> 1.5` | Excess return adjusted only for downside volatility. |
| **Max Drawdown** | `< 15.0%` | Peak-to-trough decline of capital. |
| **Profit Factor** | `> 1.5` | Ratio of Gross Profits to Gross Losses. |

---

## 🤝 Contributing & Extension
Contributions are welcome! Please write unit tests for new signal calculations and verify code correctness:
```bash
python -m unittest discover -s tests
```
