from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PRICE_CACHE_DIR = RAW_DIR / "prices"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_DIR = DATA_DIR / "features"
MODEL_DIR = DATA_DIR / "models"
REPORT_DIR = DATA_DIR / "reports"
LOG_DIR = PROJECT_ROOT / "logs"
WEB_DIR = PROJECT_ROOT / "web"

LATEST_SCAN_PATH = PROCESSED_DIR / "latest_scan.json"
PORTFOLIO_PATH = PROCESSED_DIR / "portfolio.json"
TRAINING_EVENTS_PATH = FEATURE_DIR / "training_events.jsonl"

