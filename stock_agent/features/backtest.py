from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..constants import PRICE_CACHE_DIR
from ..data.exchange_calendar import add_trading_days
from ..data.validation import normalize_ohlcv
from .signal_engine import precompute_signal_frames, prepare_signal_frame, score_precomputed_at


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 1_000_000_000.0
    commission_pct: float = 0.15
    sell_tax_pct: float = 0.10
    slippage_pct: float = 0.10
    price_limit_pct: float = 6.9
    holding_days: int = 2
    lot_size: int = 100
    risk_per_trade_pct: float = 1.0
    max_symbol_weight: float = 0.25
    max_gross_exposure: float = 0.50

    @classmethod
    def from_rules(cls, rules: dict[str, Any]) -> "BacktestConfig":
        backtest = rules.get("backtest", {})
        portfolio = rules.get("portfolio", {})
        risk = rules.get("risk", {})
        return cls(
            initial_capital=float(backtest.get("initial_capital", cls.initial_capital)),
            commission_pct=float(backtest.get("commission_pct", cls.commission_pct)),
            sell_tax_pct=float(backtest.get("sell_tax_pct", cls.sell_tax_pct)),
            slippage_pct=float(backtest.get("slippage_pct", cls.slippage_pct)),
            price_limit_pct=float(backtest.get("price_limit_pct", cls.price_limit_pct)),
            holding_days=int(rules.get("time_horizon_days", cls.holding_days)),
            lot_size=int(backtest.get("lot_size", cls.lot_size)),
            risk_per_trade_pct=float(portfolio.get("risk_per_trade_pct", cls.risk_per_trade_pct)),
            max_symbol_weight=float(portfolio.get("max_symbol_weight", risk.get("max_position_weight", cls.max_symbol_weight))),
            max_gross_exposure=float(portfolio.get("max_gross_exposure", cls.max_gross_exposure)),
        )


@dataclass
class TradeRecord:
    symbol: str
    signal_date: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit_1: float
    exit_reason: str
    score: float
    gross_return_pct: float
    net_return_pct: float
    round_trip_cost_pct: float


@dataclass
class BacktestResult:
    symbol: str
    start: str
    end: str
    total_trades: int
    win_rate: float
    avg_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float = 0.0
    expectancy_pct: float = 0.0
    profit_factor: float | None = None
    precision: float = 0.0
    recall: float = 0.0
    trades: list[TradeRecord] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioTradeRecord:
    symbol: str
    signal_date: str
    entry_date: str
    exit_date: str
    quantity: int
    entry_price: float
    exit_price: float
    entry_value: float
    exit_value: float
    stop_loss: float
    take_profit_1: float
    exit_reason: str
    score: float
    gross_return_pct: float
    net_return_pct: float
    net_pnl: float


@dataclass
class PortfolioEquityPoint:
    date: str
    cash: float
    exposure: float
    equity: float
    drawdown_pct: float


@dataclass
class PortfolioBacktestResult:
    start: str
    end: str
    initial_capital: float
    final_equity: float
    total_return_pct: float
    total_trades: int
    win_rate: float
    avg_return_pct: float
    expectancy_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float | None
    precision: float
    recall: float
    trades: list[PortfolioTradeRecord] = field(default_factory=list)
    equity_curve: list[PortfolioEquityPoint] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def round_trip_cost_pct(config: BacktestConfig) -> float:
    return config.commission_pct * 2 + config.sell_tax_pct + config.slippage_pct * 2


def get_hose_tick_size(price: float) -> float:
    if price < 1000.0:
        return 0.1
    elif price < 10000.0:
        return 10.0
    elif price < 50000.0:
        return 50.0
    else:
        return 100.0


def apply_slippage(price: float, is_buy: bool, ticks: int = 1) -> float:
    current_price = price
    for _ in range(ticks):
        tick_size = get_hose_tick_size(current_price)
        if is_buy:
            current_price += tick_size
        else:
            current_price -= tick_size
    return current_price


def enforce_price_limit(price: float, reference_price: float, limit_pct: float) -> float:
    if reference_price <= 0:
        return float(price)
    limit = limit_pct / 100.0
    lower = reference_price * (1 - limit)
    upper = reference_price * (1 + limit)
    return float(min(max(price, lower), upper))


def _max_drawdown(returns_pct: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for ret in returns_pct:
        equity *= 1 + ret / 100.0
        peak = max(peak, equity)
        drawdown = (equity - peak) / peak * 100.0
        max_dd = min(max_dd, drawdown)
    return abs(max_dd)


def _profit_factor(returns_pct: list[float]) -> float | None:
    gains = sum(ret for ret in returns_pct if ret > 0)
    losses = abs(sum(ret for ret in returns_pct if ret < 0))
    if losses == 0:
        return None if gains > 0 else 0.0
    return gains / losses


def _expectancy(returns_pct: list[float]) -> float:
    if not returns_pct:
        return 0.0
    wins = [ret for ret in returns_pct if ret > 0]
    losses = [ret for ret in returns_pct if ret <= 0]
    win_rate = len(wins) / len(returns_pct)
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = abs(float(np.mean(losses))) if losses else 0.0
    return win_rate * avg_win - (1 - win_rate) * avg_loss


def _sharpe_ratio(returns_pct: list[float]) -> float:
    if len(returns_pct) < 2:
        return 0.0
    returns = np.array(returns_pct, dtype=float) / 100.0
    std = float(returns.std(ddof=1))
    if std == 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(252 / 2))


def _sortino_ratio(returns_pct: list[float]) -> float:
    if len(returns_pct) < 2:
        return 0.0
    returns = np.array(returns_pct, dtype=float) / 100.0
    downside = returns[returns < 0]
    if len(downside) == 0:
        return 0.0
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else abs(float(downside[0]))
    if downside_std == 0:
        return 0.0
    return float(returns.mean() / downside_std * np.sqrt(252 / 2))


def _max_drawdown_equity(points: list[PortfolioEquityPoint]) -> float:
    if not points:
        return 0.0
    return round(max(abs(point.drawdown_pct) for point in points), 4)


def trade_metric_summary(returns_pct: list[float], selected_count: int | None = None, opportunity_count: int | None = None) -> dict[str, Any]:
    total = len(returns_pct)
    wins = [ret for ret in returns_pct if ret > 0]
    profit_factor = _profit_factor(returns_pct)
    selected = selected_count if selected_count is not None else total
    opportunities = opportunity_count if opportunity_count is not None else selected
    return {
        "total_trades": total,
        "win_rate": round(len(wins) / total, 3) if total else 0.0,
        "avg_return_pct": round(float(np.mean(returns_pct)), 4) if returns_pct else 0.0,
        "expectancy_pct": round(_expectancy(returns_pct), 4),
        "max_drawdown_pct": round(_max_drawdown(returns_pct), 4),
        "sharpe_ratio": round(_sharpe_ratio(returns_pct), 4),
        "sortino_ratio": round(_sortino_ratio(returns_pct), 4),
        "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
        "precision": round(len(wins) / selected * 100, 3) if selected else 0.0,
        "recall": round(len(wins) / opportunities * 100, 3) if opportunities else 0.0,
    }


def run_backtest(
    symbol: str,
    df: pd.DataFrame,
    rules: dict[str, Any],
    config: BacktestConfig | None = None,
    start: date | None = None,
    end: date | None = None,
) -> BacktestResult:
    cfg = config or BacktestConfig()
    data = df.sort_values("date").reset_index(drop=True).copy()
    if not data.empty:
        data = prepare_signal_frame(data, rules)
    if data.empty:
        return BacktestResult(
            symbol=symbol.upper(),
            start="",
            end="",
            total_trades=0,
            win_rate=0.0,
            avg_return_pct=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            expectancy_pct=0.0,
            profit_factor=0.0,
            precision=0.0,
            recall=0.0,
        )

    start = start or data["date"].iloc[0]
    end = end or data["date"].iloc[-1]
    min_rows = int(rules["min_history_rows"])
    cost_pct = round_trip_cost_pct(cfg)
    trades: list[TradeRecord] = []

    i = max(min_rows - 1, 1)
    while i < len(data) - 1:
        signal_date = data.loc[i, "date"]
        if signal_date < start or signal_date > end:
            i += 1
            continue

        signal = score_precomputed_at(symbol, data, i, rules)
        if signal.decision != "BUY_SETUP" or signal.risk_plan is None:
            i += 1
            continue

        # ML override
        ml_rules = rules.get("ml", {})
        if ml_rules.get("enabled", False) and ml_rules.get("override_enabled", False):
            from .ml_models import predict_model_signal
            model_signal = predict_model_signal(symbol, signal, rules)
            if model_signal.status == "available" and model_signal.probability is not None:
                override_loss_threshold = float(ml_rules.get("override_loss_threshold", 0.60))
                if model_signal.probability < (1.0 - override_loss_threshold):
                    signal.decision = "WATCH"

        if signal.decision != "BUY_SETUP" or signal.risk_plan is None:
            i += 1
            continue

        entry_idx = i + 1
        entry_row = data.loc[entry_idx]
        entry_date = entry_row["date"]
        if entry_date > end:
            break

        entry_reference = float(data.loc[i, "close"])
        entry_price = enforce_price_limit(float(entry_row["open"]), entry_reference, cfg.price_limit_pct)
        target_exit_date = add_trading_days(entry_date, cfg.holding_days)
        exit_idx = entry_idx
        while exit_idx + 1 < len(data) and data.loc[exit_idx, "date"] < target_exit_date:
            exit_idx += 1

        stop_loss = float(signal.risk_plan.stop_loss)
        take_profit_1 = float(signal.risk_plan.take_profit_1)
        exit_price = float(data.loc[exit_idx, "close"])
        exit_reason = "T2_CLOSE"
        actual_exit_idx = exit_idx
        for j in range(entry_idx, exit_idx + 1):
            if j < entry_idx + 2:
                # T+2 settlement rule: cannot exit on T+0 or T+1
                continue
            day = data.loc[j]
            if float(day["low"]) <= stop_loss:
                exit_price = stop_loss
                exit_reason = "STOP_LOSS"
                actual_exit_idx = j
                break
            if float(day["high"]) >= take_profit_1:
                exit_price = take_profit_1
                exit_reason = "TAKE_PROFIT_1"
                actual_exit_idx = j
                break

        exit_reference_idx = max(actual_exit_idx - 1, 0)
        exit_reference = float(data.loc[exit_reference_idx, "close"])
        exit_price = enforce_price_limit(exit_price, exit_reference, cfg.price_limit_pct)
        
        # Apply tick-size-aware slippage to prices for return calculations:
        entry_price_slipped = apply_slippage(entry_price, is_buy=True, ticks=1)
        entry_price_slipped = enforce_price_limit(entry_price_slipped, entry_reference, cfg.price_limit_pct)
        
        exit_price_slipped = apply_slippage(exit_price, is_buy=False, ticks=1)
        exit_price_slipped = enforce_price_limit(exit_price_slipped, exit_reference, cfg.price_limit_pct)
        
        gross_return_pct = (exit_price_slipped - entry_price_slipped) / entry_price_slipped * 100.0 if entry_price_slipped else 0.0
        net_return_pct = gross_return_pct - (cfg.commission_pct * 2 + cfg.sell_tax_pct)
        trades.append(
            TradeRecord(
                symbol=symbol.upper(),
                signal_date=str(signal_date),
                entry_date=str(entry_date),
                exit_date=str(data.loc[actual_exit_idx, "date"]),
                entry_price=round(entry_price, 4),
                exit_price=round(exit_price, 4),
                stop_loss=round(stop_loss, 4),
                take_profit_1=round(take_profit_1, 4),
                exit_reason=exit_reason,
                score=signal.score,
                gross_return_pct=round(gross_return_pct, 4),
                net_return_pct=round(net_return_pct, 4),
                round_trip_cost_pct=round(cost_pct, 4),
            )
        )
        i = max(actual_exit_idx + 1, i + 1)

    returns = [trade.net_return_pct for trade in trades]
    metrics = trade_metric_summary(returns)
    summary = {
        "cost_model": {
            "commission_pct_each_side": cfg.commission_pct,
            "sell_tax_pct": cfg.sell_tax_pct,
            "slippage_pct_each_side": cfg.slippage_pct,
            "round_trip_cost_pct": round(cost_pct, 4),
        },
        "holding_days": cfg.holding_days,
        "price_limit_pct": cfg.price_limit_pct,
        "lot_size": cfg.lot_size,
        "risk_per_trade_pct": cfg.risk_per_trade_pct,
        "unallocated_cash_allowed": True,
    }
    return BacktestResult(
        symbol=symbol.upper(),
        start=str(start),
        end=str(end),
        total_trades=metrics["total_trades"],
        win_rate=metrics["win_rate"],
        avg_return_pct=metrics["avg_return_pct"],
        max_drawdown_pct=metrics["max_drawdown_pct"],
        sharpe_ratio=metrics["sharpe_ratio"],
        sortino_ratio=metrics["sortino_ratio"],
        expectancy_pct=metrics["expectancy_pct"],
        profit_factor=metrics["profit_factor"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        trades=trades,
        summary=summary,
    )


def run_portfolio_backtest(
    symbol_frames: dict[str, pd.DataFrame],
    rules: dict[str, Any],
    config: BacktestConfig | None = None,
    start: date | None = None,
    end: date | None = None,
) -> PortfolioBacktestResult:
    cfg = config or BacktestConfig.from_rules(rules)
    frames = precompute_signal_frames(symbol_frames, rules)
    if not frames:
        return _empty_portfolio_result(cfg, start, end, "no price frames")

    all_dates = sorted({day for frame in frames.values() for day in frame["date"]})
    if start:
        all_dates = [day for day in all_dates if day >= start]
    if end:
        all_dates = [day for day in all_dates if day <= end]
    if not all_dates:
        return _empty_portfolio_result(cfg, start, end, "no dates in range")

    index_by_symbol = {
        symbol: {row["date"]: idx for idx, row in frame.iterrows()}
        for symbol, frame in frames.items()
    }
    cash = float(cfg.initial_capital)
    open_positions: dict[str, dict[str, Any]] = {}
    trades: list[PortfolioTradeRecord] = []
    equity_curve: list[PortfolioEquityPoint] = []
    min_rows = int(rules["min_history_rows"])

    for current_date in all_dates:
        cash += _close_due_positions(current_date, frames, index_by_symbol, open_positions, trades, cfg)
        equity_before_entries, exposure_before_entries = _mark_to_market(current_date, frames, index_by_symbol, open_positions, cash)
        cash = _open_new_positions(
            current_date,
            frames,
            index_by_symbol,
            open_positions,
            rules,
            cfg,
            cash,
            equity_before_entries,
            exposure_before_entries,
            min_rows,
        )
        equity, exposure = _mark_to_market(current_date, frames, index_by_symbol, open_positions, cash)
        peak = max([point.equity for point in equity_curve], default=cfg.initial_capital)
        peak = max(peak, equity)
        drawdown = (equity - peak) / peak * 100.0 if peak else 0.0
        equity_curve.append(
            PortfolioEquityPoint(
                date=str(current_date),
                cash=round(cash, 2),
                exposure=round(exposure, 2),
                equity=round(equity, 2),
                drawdown_pct=round(drawdown, 4),
            )
        )

    if open_positions:
        last_date = all_dates[-1]
        cash += _force_close_positions(last_date, frames, index_by_symbol, open_positions, trades, cfg)
        equity_curve[-1].cash = round(cash, 2)
        equity_curve[-1].equity = round(cash, 2)
        equity_curve[-1].exposure = 0.0

    final_equity = equity_curve[-1].equity if equity_curve else cfg.initial_capital
    returns = [trade.net_return_pct for trade in trades]
    metrics = trade_metric_summary(returns)
    return PortfolioBacktestResult(
        start=str(all_dates[0]),
        end=str(all_dates[-1]),
        initial_capital=round(cfg.initial_capital, 2),
        final_equity=round(final_equity, 2),
        total_return_pct=round((final_equity - cfg.initial_capital) / cfg.initial_capital * 100.0, 4) if cfg.initial_capital else 0.0,
        total_trades=metrics["total_trades"],
        win_rate=metrics["win_rate"],
        avg_return_pct=metrics["avg_return_pct"],
        expectancy_pct=metrics["expectancy_pct"],
        max_drawdown_pct=_max_drawdown_equity(equity_curve),
        sharpe_ratio=metrics["sharpe_ratio"],
        sortino_ratio=metrics["sortino_ratio"],
        profit_factor=metrics["profit_factor"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        trades=trades,
        equity_curve=equity_curve,
        summary={
            "cost_model": {
                "commission_pct_each_side": cfg.commission_pct,
                "sell_tax_pct": cfg.sell_tax_pct,
                "slippage_pct_each_side": cfg.slippage_pct,
            },
            "portfolio_constraints": {
                "lot_size": cfg.lot_size,
                "risk_per_trade_pct": cfg.risk_per_trade_pct,
                "max_symbol_weight": cfg.max_symbol_weight,
                "max_gross_exposure": cfg.max_gross_exposure,
            },
        },
    )


def run_portfolio_backtest_from_csvs(
    symbols: Iterable[str],
    rules: dict[str, Any],
    config: BacktestConfig | None = None,
    start: date | None = None,
    end: date | None = None,
    price_dir=PRICE_CACHE_DIR,
) -> PortfolioBacktestResult:
    import sys
    import hashlib
    from datetime import timezone, datetime
    from ..data.repository import write_json
    from ..schemas import to_plain_dict
    from ..pipeline.performance_tracker import create_and_log_manifest
    from ..constants import REPORT_DIR

    requested_symbols = [symbol.upper() for symbol in symbols]
    frames: dict[str, pd.DataFrame] = {}
    excluded_symbols: list[str] = []
    
    for symbol in requested_symbols:
        path = price_dir / f"{symbol}.csv"
        if not path.exists():
            excluded_symbols.append(symbol)
            continue
        try:
            frames[symbol] = normalize_ohlcv(pd.read_csv(path))
        except Exception:
            excluded_symbols.append(symbol)
            
    result = run_portfolio_backtest(frames, rules, config=config, start=start, end=end)
    
    # Check if we are running under pytest or unittest
    if "pytest" not in sys.modules and "unittest" not in sys.modules:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORT_DIR / f"backtest_{timestamp}.json"
        
        # Save the backtest report
        write_json(report_path, to_plain_dict(result))
        
        # Compute SHA-256 data hashes
        data_hashes: dict[str, str] = {}
        for symbol in frames:
            csv_path = price_dir / f"{symbol}.csv"
            if csv_path.exists():
                hasher = hashlib.sha256()
                with csv_path.open("rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        hasher.update(chunk)
                data_hashes[symbol] = hasher.hexdigest()
                
        # Combined hash
        combined_hasher = hashlib.sha256()
        for symbol in sorted(data_hashes):
            combined_hasher.update(data_hashes[symbol].encode("utf-8"))
        combined_data_hash = combined_hasher.hexdigest()
        
        # Log the manifest
        command = " ".join(sys.argv)
        create_and_log_manifest(
            command=command,
            rules=rules,
            data_start=result.start,
            data_end=result.end,
            symbols=list(frames.keys()),
            report_links={"backtest_report": str(report_path)},
            symbols_excluded=excluded_symbols,
            data_hash=combined_data_hash,
            data_hashes=data_hashes,
        )
        
    return result


def _empty_portfolio_result(cfg: BacktestConfig, start: date | None, end: date | None, reason: str) -> PortfolioBacktestResult:
    return PortfolioBacktestResult(
        start=str(start or ""),
        end=str(end or ""),
        initial_capital=cfg.initial_capital,
        final_equity=cfg.initial_capital,
        total_return_pct=0.0,
        total_trades=0,
        win_rate=0.0,
        avg_return_pct=0.0,
        expectancy_pct=0.0,
        max_drawdown_pct=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        profit_factor=0.0,
        precision=0.0,
        recall=0.0,
        summary={"warning": reason},
    )


def _close_due_positions(
    current_date: date,
    frames: dict[str, pd.DataFrame],
    index_by_symbol: dict[str, dict[date, int]],
    open_positions: dict[str, dict[str, Any]],
    trades: list[PortfolioTradeRecord],
    cfg: BacktestConfig,
) -> float:
    cash_delta = 0.0
    for symbol in list(open_positions):
        if current_date not in index_by_symbol[symbol]:
            continue
        position = open_positions[symbol]
        idx = index_by_symbol[symbol][current_date]
        day = frames[symbol].loc[idx]
        exit_price = float(day["close"])
        exit_reason = "T2_CLOSE"
        if current_date < position["planned_exit_date"]:
            continue
        if float(day["low"]) <= position["stop_loss"]:
            exit_price = float(position["stop_loss"])
            exit_reason = "STOP_LOSS"
        elif float(day["high"]) >= position["take_profit_1"]:
            exit_price = float(position["take_profit_1"])
            exit_reason = "TAKE_PROFIT_1"
        cash_delta += _record_exit(symbol, current_date, exit_price, exit_reason, position, trades, cfg)
        del open_positions[symbol]
    return cash_delta


def _force_close_positions(
    current_date: date,
    frames: dict[str, pd.DataFrame],
    index_by_symbol: dict[str, dict[date, int]],
    open_positions: dict[str, dict[str, Any]],
    trades: list[PortfolioTradeRecord],
    cfg: BacktestConfig,
) -> float:
    cash_delta = 0.0
    for symbol in list(open_positions):
        if current_date not in index_by_symbol[symbol]:
            continue
        idx = index_by_symbol[symbol][current_date]
        exit_price = float(frames[symbol].loc[idx, "close"])
        cash_delta += _record_exit(symbol, current_date, exit_price, "FORCED_CLOSE", open_positions[symbol], trades, cfg)
        del open_positions[symbol]
    return cash_delta


def _record_exit(
    symbol: str,
    current_date: date,
    exit_price: float,
    exit_reason: str,
    position: dict[str, Any],
    trades: list[PortfolioTradeRecord],
    cfg: BacktestConfig,
) -> float:
    effective_exit = apply_slippage(exit_price, is_buy=False, ticks=1)
    gross_return_pct = (effective_exit - position["entry_price"]) / position["entry_price"] * 100.0
    sell_cost = (cfg.commission_pct + cfg.sell_tax_pct) / 100.0 * effective_exit * position["quantity"]
    exit_value = effective_exit * position["quantity"]
    net_pnl = exit_value - sell_cost - position["entry_total_cost"]
    net_return_pct = net_pnl / position["entry_value"] * 100.0 if position["entry_value"] else 0.0
    trades.append(
        PortfolioTradeRecord(
            symbol=symbol,
            signal_date=str(position["signal_date"]),
            entry_date=str(position["entry_date"]),
            exit_date=str(current_date),
            quantity=int(position["quantity"]),
            entry_price=round(position["entry_price"], 4),
            exit_price=round(effective_exit, 4),
            entry_value=round(position["entry_value"], 2),
            exit_value=round(exit_value, 2),
            stop_loss=round(position["stop_loss"], 4),
            take_profit_1=round(position["take_profit_1"], 4),
            exit_reason=exit_reason,
            score=round(position["score"], 2),
            gross_return_pct=round(gross_return_pct, 4),
            net_return_pct=round(net_return_pct, 4),
            net_pnl=round(net_pnl, 2),
        )
    )
    return exit_value - sell_cost


def _open_new_positions(
    current_date: date,
    frames: dict[str, pd.DataFrame],
    index_by_symbol: dict[str, dict[date, int]],
    open_positions: dict[str, dict[str, Any]],
    rules: dict[str, Any],
    cfg: BacktestConfig,
    cash: float,
    equity: float,
    exposure: float,
    min_rows: int,
) -> float:
    candidates: list[tuple[float, str, Any, pd.Series, int]] = []
    for symbol, frame in frames.items():
        if symbol in open_positions or current_date not in index_by_symbol[symbol]:
            continue
        entry_idx = index_by_symbol[symbol][current_date]
        signal_idx = entry_idx - 1
        if signal_idx < min_rows - 1:
            continue
        signal = score_precomputed_at(symbol, frame, signal_idx, rules)
        if signal.decision != "BUY_SETUP" or signal.risk_plan is None:
            continue

        # ML override
        ml_rules = rules.get("ml", {})
        if ml_rules.get("enabled", False) and ml_rules.get("override_enabled", False):
            from .ml_models import predict_model_signal
            model_signal = predict_model_signal(symbol, signal, rules)
            if model_signal.status == "available" and model_signal.probability is not None:
                override_loss_threshold = float(ml_rules.get("override_loss_threshold", 0.60))
                if model_signal.probability < (1.0 - override_loss_threshold):
                    signal.decision = "WATCH"

        if signal.decision != "BUY_SETUP" or signal.risk_plan is None:
            continue
        candidates.append((signal.score, symbol, signal, frame.loc[entry_idx], signal_idx))

    for _, symbol, signal, entry_row, signal_idx in sorted(candidates, reverse=True):
        if cash <= 0 or exposure >= equity * cfg.max_gross_exposure:
            break
        entry_reference = float(frames[symbol].loc[signal_idx, "close"])
        raw_entry = apply_slippage(float(entry_row["open"]), is_buy=True, ticks=1)
        entry_price = enforce_price_limit(raw_entry, entry_reference, cfg.price_limit_pct)
        stop_loss = min(float(signal.risk_plan.stop_loss), entry_price * 0.995)
        risk_per_share = max(entry_price - stop_loss, entry_price * 0.005)
        risk_budget = equity * cfg.risk_per_trade_pct / 100.0
        max_value = equity * cfg.max_symbol_weight
        exposure_room = max(0.0, equity * cfg.max_gross_exposure - exposure)
        cash_room = max(0.0, cash / (1 + (cfg.commission_pct + cfg.slippage_pct) / 100.0))
        raw_quantity = min(risk_budget / risk_per_share, max_value / entry_price, exposure_room / entry_price, cash_room / entry_price)
        quantity = int(raw_quantity // max(cfg.lot_size, 1) * max(cfg.lot_size, 1))
        if quantity <= 0:
            continue
        entry_value = entry_price * quantity
        buy_cost = cfg.commission_pct / 100.0 * entry_value
        entry_total_cost = entry_value + buy_cost
        if entry_total_cost > cash:
            continue
        cash -= entry_total_cost
        exposure += entry_value
        open_positions[symbol] = {
            "signal_date": frames[symbol].loc[signal_idx, "date"],
            "entry_date": entry_row["date"],
            "planned_exit_date": add_trading_days(entry_row["date"], cfg.holding_days),
            "quantity": quantity,
            "entry_price": entry_price,
            "entry_value": entry_value,
            "entry_total_cost": entry_total_cost,
            "stop_loss": stop_loss,
            "take_profit_1": float(signal.risk_plan.take_profit_1),
            "score": signal.score,
        }
    return cash


def _mark_to_market(
    current_date: date,
    frames: dict[str, pd.DataFrame],
    index_by_symbol: dict[str, dict[date, int]],
    open_positions: dict[str, dict[str, Any]],
    cash: float,
) -> tuple[float, float]:
    exposure = 0.0
    for symbol, position in open_positions.items():
        idx = index_by_symbol[symbol].get(current_date)
        if idx is None:
            exposure += float(position["entry_value"])
            continue
        exposure += float(frames[symbol].loc[idx, "close"]) * float(position["quantity"])
    return cash + exposure, exposure
