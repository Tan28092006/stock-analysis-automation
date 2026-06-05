from __future__ import annotations

from datetime import date
from contextlib import redirect_stdout
import io
import time
from pathlib import Path
from typing import Protocol

import pandas as pd

from ..constants import PRICE_CACHE_DIR
from ..schemas import ProviderAudit
from .sample_data import make_demo_ohlcv
from .validation import ProviderFrame, normalize_ohlcv


def _save_cache(symbol: str, df: pd.DataFrame) -> None:
    """Save fetched OHLCV data to CSV cache for reuse."""
    try:
        PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PRICE_CACHE_DIR / f"{symbol.upper()}.csv", index=False)
    except Exception:
        pass


class HistoryProvider(Protocol):
    name: str
    source_type: str

    def history(self, symbol: str, start: date, end: date) -> ProviderFrame:
        ...


class LocalCsvProvider:
    name = "local_csv"
    source_type = "cache"

    def __init__(self, root: Path = PRICE_CACHE_DIR):
        self.root = root

    def history(self, symbol: str, start: date, end: date) -> ProviderFrame:
        path = self.root / f"{symbol.upper()}.csv"
        if not path.exists():
            return ProviderFrame(
                self.name,
                None,
                ProviderAudit(
                    self.name,
                    "missing",
                    error=f"{path} not found",
                    source_type=self.source_type,
                    rejection_reason="validation_failed",
                ),
            )
        try:
            df = normalize_ohlcv(pd.read_csv(path))
            df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
            return ProviderFrame(
                self.name,
                df,
                ProviderAudit(
                    self.name,
                    "ok",
                    rows=len(df),
                    latest_date=str(df["date"].iloc[-1]) if not df.empty else None,
                    latest_close=float(df["close"].iloc[-1]) if not df.empty else None,
                    source_type=self.source_type,
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive provider wrapper
            return ProviderFrame(
                self.name,
                None,
                ProviderAudit(
                    self.name,
                    "error",
                    error=str(exc),
                    source_type=self.source_type,
                    rejection_reason="validation_failed",
                ),
            )


class DemoProvider:
    def __init__(self, name: str, variant: int = 0):
        self.name = name
        self.variant = variant
        self.source_type = "demo_synthetic"

    def history(self, symbol: str, start: date, end: date) -> ProviderFrame:
        try:
            df = make_demo_ohlcv(symbol, end=end, rows=180, variant=self.variant)
            df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
            return ProviderFrame(
                self.name,
                df,
                ProviderAudit(
                    self.name,
                    "ok",
                    rows=len(df),
                    latest_date=str(df["date"].iloc[-1]) if not df.empty else None,
                    latest_close=float(df["close"].iloc[-1]) if not df.empty else None,
                    source_type=self.source_type,
                ),
            )
        except Exception as exc:  # pragma: no cover
            return ProviderFrame(
                self.name,
                None,
                ProviderAudit(
                    self.name,
                    "error",
                    error=str(exc),
                    source_type=self.source_type,
                    rejection_reason="validation_failed",
                ),
            )


class VnStockProvider:
    name = "vnstock"
    source_type = "market_api"

    def history(self, symbol: str, start: date, end: date) -> ProviderFrame:
        try:
            try:
                import vnstock
            except ImportError:
                return ProviderFrame(
                    self.name,
                    None,
                    ProviderAudit(
                        self.name,
                        "missing",
                        error="vnstock is not installed or has an unsupported API",
                        source_type=self.source_type,
                        rejection_reason="import_error",
                    ),
                )

            time.sleep(3.5)  # Throttle: vnstock Guest limit = 20 req/min
            try:
                from vnstock import Vnstock  # type: ignore

                with redirect_stdout(io.StringIO()):
                    stock = Vnstock().stock(symbol=symbol, source="VCI")
                df = stock.quote.history(start=str(start), end=str(end), interval="1D")
            except ImportError:
                try:
                    from vnstock import stock_historical_data  # type: ignore

                    df = stock_historical_data(symbol, str(start), str(end))
                except ImportError:
                    return ProviderFrame(
                        self.name,
                        None,
                        ProviderAudit(
                            self.name,
                            "missing",
                            error="vnstock is not installed or has an unsupported API",
                            source_type=self.source_type,
                            rejection_reason="import_error",
                        ),
                    )

            out = normalize_ohlcv(df)
            if not out.empty and float(out["close"].median()) < 1000:
                out[["open", "high", "low", "close"]] = out[["open", "high", "low", "close"]] * 1000
            _save_cache(symbol, out)
            return ProviderFrame(
                self.name,
                out,
                ProviderAudit(
                    self.name,
                    "ok",
                    rows=len(out),
                    latest_date=str(out["date"].iloc[-1]) if not out.empty else None,
                    latest_close=float(out["close"].iloc[-1]) if not out.empty else None,
                    source_type=self.source_type,
                ),
            )
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # pragma: no cover - catch SystemExit from vnstock rate limiter
            return ProviderFrame(
                self.name,
                None,
                ProviderAudit(
                    self.name,
                    "error",
                    error=str(exc),
                    source_type=self.source_type,
                    rejection_reason="network_error",
                ),
            )


class YahooProvider:
    name = "yfinance"
    source_type = "market_api_backup"

    def history(self, symbol: str, start: date, end: date) -> ProviderFrame:
        try:
            try:
                import yfinance as yf  # type: ignore
            except ImportError:
                return ProviderFrame(
                    self.name,
                    None,
                    ProviderAudit(
                        self.name,
                        "missing",
                        error="yfinance is not installed",
                        source_type=self.source_type,
                        rejection_reason="import_error",
                    ),
                )

            from datetime import timedelta
            query_end = end + timedelta(days=1)
            ticker = yf.Ticker(f"{symbol}.VN")
            raw = ticker.history(start=str(start), end=str(query_end), auto_adjust=True)
            if raw.empty:
                raw = yf.Ticker(f"{symbol}.HM").history(start=str(start), end=str(query_end), auto_adjust=True)
            raw = raw.reset_index()
            raw = raw.rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            out = normalize_ohlcv(raw)
            _save_cache(symbol, out)
            return ProviderFrame(
                self.name,
                out,
                ProviderAudit(
                    self.name,
                    "ok",
                    rows=len(out),
                    latest_date=str(out["date"].iloc[-1]) if not out.empty else None,
                    latest_close=float(out["close"].iloc[-1]) if not out.empty else None,
                    source_type=self.source_type,
                ),
            )
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # pragma: no cover - network/provider dependent
            return ProviderFrame(
                self.name,
                None,
                ProviderAudit(
                    self.name,
                    "error",
                    error=str(exc),
                    source_type=self.source_type,
                    rejection_reason="network_error",
                ),
            )


def build_providers(demo: bool) -> list[HistoryProvider]:
    if demo:
        return [DemoProvider("demo_primary", 0), DemoProvider("demo_cross_check", 1)]
    return [LocalCsvProvider(), YahooProvider(), VnStockProvider()]
