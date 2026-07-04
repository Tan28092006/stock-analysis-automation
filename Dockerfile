# VN Swing-Trading System — dashboard + JSON API (stdlib http.server).
# Multi-arch, slim. libgomp1 is required at runtime by lightgbm/xgboost.
FROM python:3.11-slim

# libgomp1: OpenMP runtime for LightGBM/XGBoost. curl: health checks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + bundled data (prices_hist, configs, web). Auth cookies and models
# are excluded via .dockerignore; the P(win) model is retrained on first run if absent.
COPY . .

# Accept external connections; PaaS platforms inject PORT (Render/Railway/Fly).
ENV HOST=0.0.0.0 \
    PORT=8000 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

# Shell form so ${PORT} is expanded from the environment at runtime.
CMD python -m stock_agent.app --host "$HOST" --port "$PORT"
