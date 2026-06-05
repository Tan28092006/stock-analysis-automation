from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from .agents.orchestrator import run_scan
from .config import load_rules, load_universe
from .constants import PRICE_CACHE_DIR, REPORT_DIR
from .data.repository import write_json
from .data.validation import normalize_ohlcv
from .features.backtest import BacktestConfig, run_backtest, run_portfolio_backtest_from_csvs
from .features.calibration import build_labeled_t2_dataset, run_calibration
from .features.ml_models import model_status, train_model_suite
from .features.optimizer import OptimizerConfig, refresh_price_data, run_optimization
from .features.robustness import run_robustness
from .portfolio.pnl import (
    PortfolioStore,
    calculate_pnl,
    latest_prices_from_scan,
    log_portfolio_features,
)
from .ai.reporting import synthesize_latest_report
from .pipeline.daily_runner import DailyRunner
from .schemas import to_plain_dict


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def scan_cmd(args: argparse.Namespace) -> None:
    symbols = [item.strip().upper() for item in args.symbols.split(",")] if args.symbols else None
    result = run_scan(demo=args.demo, symbols=symbols, persist=True)
    print(json.dumps(to_plain_dict(result), ensure_ascii=False, indent=2))


def portfolio_cmd(args: argparse.Namespace) -> None:
    store = PortfolioStore()
    if args.action == "add":
        position = store.add_position(args.symbol, args.buy_price, args.quantity, args.buy_date)
        print(json.dumps(position, ensure_ascii=False, indent=2))
    elif args.action == "list":
        print(json.dumps({"positions": store.list_positions()}, ensure_ascii=False, indent=2))
    elif args.action == "pnl":
        payload = calculate_pnl(store.list_positions(), latest_prices_from_scan())
        log_portfolio_features(payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.action == "clear":
        store.clear()
        print(json.dumps({"status": "cleared"}, ensure_ascii=False, indent=2))


def _ai_progress(symbol: str, idx: int, total: int, phase: str) -> None:
    if phase == "REJECT_BATCH":
        print(f"  ⏳ Analyzing REJECT batch: {symbol}")
    elif phase == "OVERVIEW":
        print(f"  ⏳ Generating market overview...")
    else:
        print(f"  ⏳ Analyzing {symbol} ({idx}/{total}) [{phase}]")


def ai_cmd(args: argparse.Namespace) -> None:
    if args.action == "report":
        symbol = getattr(args, "symbol", None)
        force = getattr(args, "force", False)
        print(f"🤖 Generating AI report (model: Ollama local)...")
        payload = synthesize_latest_report(
            symbol=symbol,
            force=force,
            on_progress=_ai_progress if not symbol else None,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def backtest_cmd(args: argparse.Namespace) -> None:
    import pandas as pd

    rules = load_rules()
    if args.all:
        symbols = _symbols_from_args(args)
        result = run_portfolio_backtest_from_csvs(
            symbols=symbols,
            rules=rules,
            config=BacktestConfig.from_rules(rules),
            start=_parse_date(args.start),
            end=_parse_date(args.end),
        )
        payload = to_plain_dict(result)
        if args.output:
            write_json(Path(args.output), payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if not args.symbol:
        raise SystemExit("backtest requires --symbol or --all")
    symbol = args.symbol.upper()
    path = PRICE_CACHE_DIR / f"{symbol}.csv"
    if not path.exists():
        raise SystemExit(f"No local CSV found for {symbol}: {path}")
    df = normalize_ohlcv(pd.read_csv(path))
    result = run_backtest(
        symbol=symbol,
        df=df,
        rules=rules,
        config=BacktestConfig.from_rules(rules),
        start=_parse_date(args.start),
        end=_parse_date(args.end),
    )
    payload = to_plain_dict(result)
    if args.output:
        write_json(Path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def refresh_data_cmd(args: argparse.Namespace) -> None:
    rules = load_rules()
    payload = refresh_price_data(
        symbols=_symbols_from_args(args),
        years=args.years,
        rules=rules,
    )
    if args.output:
        write_json(Path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def calibrate_cmd(args: argparse.Namespace) -> None:
    rules = load_rules()
    symbols = _symbols_from_args(args)
    dataset = build_labeled_t2_dataset(
        symbols=symbols,
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        rules=rules,
        cost_config=BacktestConfig(),
    )
    report = run_calibration(dataset, objective="win_rate", min_trades=args.min_trades)
    payload = to_plain_dict(report)
    if args.output:
        output = Path(args.output)
        if output.exists() and output.is_dir():
            output = output / f"calibration_report_{report.created_at.replace(':', '').replace('+', 'Z')}.json"
        write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _symbols_from_args(args: argparse.Namespace) -> list[str]:
    if getattr(args, "symbols", None):
        return [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    return [item.upper() for item in load_universe()["symbols"]]


def train_cmd(args: argparse.Namespace) -> None:
    rules = load_rules()
    families = [item.strip() for item in args.families.split(",") if item.strip()] if args.families else None
    payload = train_model_suite(
        symbols=_symbols_from_args(args),
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        rules=rules,
        families=families,
    )
    if args.output:
        write_json(Path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def robustness_cmd(args: argparse.Namespace) -> None:
    rules = load_rules()
    payload = run_robustness(
        symbols=_symbols_from_args(args),
        rules=rules,
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        seed=args.seed,
        monte_carlo_runs=args.monte_carlo_runs,
    )
    if args.output:
        write_json(Path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def optimize_cmd(args: argparse.Namespace) -> None:
    rules = load_rules()
    cfg = OptimizerConfig(
        years=args.years,
        n_trials=args.n_trials,
        target_return_pct=args.target_return,
        target_win_rate=args.target_win_rate,
        min_final_trades=args.min_trades,
        max_drawdown_pct=args.max_drawdown,
        seed=args.seed,
        refresh_before=args.refresh,
    )
    payload = run_optimization(
        symbols=_symbols_from_args(args),
        rules=rules,
        config=cfg,
        apply_if_valid=args.apply_if_valid,
    )
    if args.output:
        write_json(Path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def daily_run_cmd(args: argparse.Namespace) -> None:
    force = getattr(args, "force", False)
    demo = getattr(args, "demo", False)

    def on_progress(stage: str, msg: str) -> None:
        print(f"  ⏳ [{stage}] {msg}")

    print("🤖 Starting daily ML pipeline...")
    runner = DailyRunner(demo=demo, force=force, on_progress=on_progress)
    result = runner.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    status = result.get("status", "unknown")
    duration = result.get("duration_seconds", 0)
    print(f"\n{'✅' if status == 'completed' else '⚠️'} Pipeline {status} in {duration:.1f}s")


def model_status_cmd(args: argparse.Namespace) -> None:
    print(json.dumps(model_status(), ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stock-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    scan = sub.add_parser("scan")
    scan.add_argument("--demo", action="store_true", help="Use deterministic synthetic demo providers.")
    scan.add_argument("--symbols", help="Comma-separated symbol override.")
    scan.set_defaults(func=scan_cmd)

    portfolio = sub.add_parser("portfolio")
    psub = portfolio.add_subparsers(dest="action", required=True)
    add = psub.add_parser("add")
    add.add_argument("--symbol", required=True)
    add.add_argument("--buy-price", type=float, required=True)
    add.add_argument("--quantity", type=float, required=True)
    add.add_argument("--buy-date")
    add.set_defaults(func=portfolio_cmd)
    psub.add_parser("list").set_defaults(func=portfolio_cmd)
    psub.add_parser("pnl").set_defaults(func=portfolio_cmd)
    psub.add_parser("clear").set_defaults(func=portfolio_cmd)

    ai = sub.add_parser("ai")
    aisub = ai.add_subparsers(dest="action", required=True)
    ai_report = aisub.add_parser("report")
    ai_report.add_argument("--symbol", help="Analyze a single stock only.")
    ai_report.add_argument("--force", action="store_true", help="Bypass cache and regenerate.")
    ai_report.set_defaults(func=ai_cmd)

    backtest = sub.add_parser("backtest")
    backtest.add_argument("--symbol")
    backtest.add_argument("--all", action="store_true", help="Run portfolio-level backtest across configured universe or --symbols.")
    backtest.add_argument("--symbols", help="Comma-separated symbol override for --all.")
    backtest.add_argument("--start")
    backtest.add_argument("--end")
    backtest.add_argument("--output")
    backtest.set_defaults(func=backtest_cmd)

    refresh = sub.add_parser("refresh-data")
    refresh.add_argument("--years", type=int, default=3)
    refresh.add_argument("--symbols", help="Comma-separated symbol override. Defaults to configured universe.")
    refresh.add_argument("--output")
    refresh.set_defaults(func=refresh_data_cmd)

    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--start")
    calibrate.add_argument("--end")
    calibrate.add_argument("--symbols", help="Comma-separated symbol override. Defaults to configured universe.")
    calibrate.add_argument("--min-trades", type=int, default=30)
    calibrate.add_argument("--output", help=f"Optional report path or directory. Suggested directory: {REPORT_DIR}")
    calibrate.set_defaults(func=calibrate_cmd)

    train = sub.add_parser("train")
    train.add_argument("--start")
    train.add_argument("--end")
    train.add_argument("--symbols", help="Comma-separated symbol override. Defaults to configured universe.")
    train.add_argument("--families", help="Comma-separated model families. Defaults to config ml.families.")
    train.add_argument("--output", help=f"Optional registry output path. Registry is always saved under data/models.")
    train.set_defaults(func=train_cmd)

    robustness = sub.add_parser("robustness")
    robustness.add_argument("--start")
    robustness.add_argument("--end")
    robustness.add_argument("--symbols", help="Comma-separated symbol override. Defaults to configured universe.")
    robustness.add_argument("--seed", type=int, default=42)
    robustness.add_argument("--monte-carlo-runs", type=int, default=200)
    robustness.add_argument("--output")
    robustness.set_defaults(func=robustness_cmd)

    optimize = sub.add_parser("optimize")
    optimize.add_argument("--years", type=int, default=3)
    optimize.add_argument("--n-trials", type=int, default=200)
    optimize.add_argument("--symbols", help="Comma-separated symbol override. Defaults to configured universe.")
    optimize.add_argument("--target-return", type=float, default=10.0)
    optimize.add_argument("--target-win-rate", type=float, default=60.0)
    optimize.add_argument("--min-trades", type=int, default=30)
    optimize.add_argument("--max-drawdown", type=float, default=20.0)
    optimize.add_argument("--seed", type=int, default=42)
    optimize.add_argument("--refresh", action="store_true", help="Fetch 3-year OHLCV before optimizing.")
    optimize.add_argument("--apply-if-valid", action="store_true", help="Apply selected rules only if final-test guards pass.")
    optimize.add_argument("--output")
    optimize.set_defaults(func=optimize_cmd)

    status = sub.add_parser("model-status")
    status.set_defaults(func=model_status_cmd)

    daily = sub.add_parser("daily-run", help="Run the automated daily ML pipeline.")
    daily.add_argument("--demo", action="store_true", help="Use demo data providers.")
    daily.add_argument("--force", action="store_true", help="Run even on non-trading days.")
    daily.set_defaults(func=daily_run_cmd)

    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
