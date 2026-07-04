from __future__ import annotations

import argparse
import base64
import hmac
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .agents.orchestrator import run_scan
from .config import load_rules, load_universe
from .constants import LATEST_SCAN_PATH, WEB_DIR
from .data.repository import read_json
from .features.backtest import BacktestConfig, run_portfolio_backtest_from_csvs
from .features.ml_models import model_status, train_model_suite
from .features.optimizer import OptimizerConfig, run_optimization
from .features.robustness import run_robustness
from .portfolio.pnl import PortfolioStore, calculate_pnl, latest_prices_from_scan, log_portfolio_features
from .schemas import to_plain_dict
from .ai.reporting import synthesize_latest_report
from .ai.chat import handle_chat


import threading

# Background data-update state (fetch fresh EOD prices -> rebuild scans). Separate from
# "re-scan" which only recomputes on the data already on disk.
_DATA_UPDATE = {"running": False, "message": "idle", "last_result": None}


def _do_data_update() -> None:
    from .pipeline.eod_update import refresh_prices, rebuild_mr_cache, rebuild_momentum_cache, check_sell_alerts
    try:
        _DATA_UPDATE["running"] = True
        _DATA_UPDATE["message"] = "Đang tải giá EOD mới nhất (vnstock, có thể vài phút)..."
        pr = refresh_prices()
        _DATA_UPDATE["message"] = f"Đã tải {pr.get('updated', 0)} mã tới {pr.get('end')}; rebuild scan..."
        rebuild_mr_cache()
        rebuild_momentum_cache()
        check_sell_alerts()
        _DATA_UPDATE["last_result"] = pr
        _DATA_UPDATE["message"] = (f"Xong · cập nhật {pr.get('updated', 0)} mã, "
                                   f"{pr.get('skipped_fresh', 0)} đã mới, {pr.get('failed', 0)} lỗi · tới {pr.get('end')}")
    except BaseException as exc:  # vnstock rate limiter can raise SystemExit
        _DATA_UPDATE["message"] = f"Lỗi cập nhật: {repr(exc)[:140]}"
    finally:
        _DATA_UPDATE["running"] = False


# --- HTTP Basic auth -------------------------------------------------------------------
# Gate every request behind a password when APP_PASSWORD is set (required for any public
# deploy). When it's empty the gate is OFF (local dev convenience). /api/health is always
# exempt so platform health checks (Render/Railway/Fly) can reach it without credentials.
_AUTH_USER = os.environ.get("APP_USERNAME", "admin")
_AUTH_PASS = os.environ.get("APP_PASSWORD", "")


def _auth_ok(handler: BaseHTTPRequestHandler) -> bool:
    if not _AUTH_PASS:
        return True
    if urlparse(handler.path).path == "/api/health":
        return True
    header = handler.headers.get("Authorization", "")
    if header.startswith("Basic "):
        try:
            user, _, pwd = base64.b64decode(header[6:]).decode("utf-8").partition(":")
            ok_user = hmac.compare_digest(user.encode(), _AUTH_USER.encode())
            ok_pass = hmac.compare_digest(pwd.encode(), _AUTH_PASS.encode())
            if ok_user and ok_pass:
                return True
        except Exception:
            pass
    handler.send_response(HTTPStatus.UNAUTHORIZED)
    handler.send_header("WWW-Authenticate", 'Basic realm="VN Swing", charset="UTF-8"')
    handler.send_header("Content-Length", "0")
    handler.end_headers()
    return False


def _json_response(handler: BaseHTTPRequestHandler, payload, status=HTTPStatus.OK) -> None:
    from .data.repository import clean_nan_inf
    raw = json.dumps(clean_nan_inf(payload), ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def _read_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def _parse_date_param(value: str | None):
    if not value:
        return None
    from datetime import date

    return date.fromisoformat(value)


def _symbols_from_params(params: dict, body: dict | None = None) -> list[str]:
    body = body or {}
    raw = body.get("symbols") or params.get("symbols", [None])[0]
    if isinstance(raw, list):
        return [str(item).strip().upper() for item in raw if str(item).strip()]
    if raw:
        return [item.strip().upper() for item in str(raw).split(",") if item.strip()]
    return [item.upper() for item in load_universe()["symbols"]]


class StockAgentHandler(BaseHTTPRequestHandler):
    server_version = "VN30T2MVP/0.1"

    def do_GET(self) -> None:  # noqa: N802
        if not _auth_ok(self):
            return
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            _json_response(self, {"status": "ok"})
            return
        if parsed.path == "/api/scan/latest":
            _json_response(self, read_json(LATEST_SCAN_PATH, default={"candidates": [], "warnings": []}))
            return
        if parsed.path == "/api/portfolio":
            store = PortfolioStore()
            pnl = calculate_pnl(store.list_positions(), latest_prices_from_scan())
            _json_response(self, {"positions": store.list_positions(), "pnl": pnl})
            return
        if parsed.path == "/api/report/latest":
            try:
                report = synthesize_latest_report()
                _json_response(self, report)
            except RuntimeError as exc:
                _json_response(self, {"error": str(exc)}, status=HTTPStatus.SERVICE_UNAVAILABLE)
            return
        if parsed.path == "/api/model/status":
            _json_response(self, model_status())
            return
        if parsed.path == "/api/data/update":
            _json_response(self, _DATA_UPDATE)
            return
        if parsed.path == "/api/mr/scan":
            params = parse_qs(parsed.query)
            from .features.mr_scan import mr_scan
            try:
                payload = mr_scan(
                    recent_days=int(params.get("recent_days", ["120"])[0]),
                    force=params.get("force", ["false"])[0] == "true",
                    min_win_prob=float(params.get("min_win_prob", ["0.55"])[0]),
                )
                _json_response(self, payload)
            except Exception as exc:
                _json_response(self, {"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if parsed.path == "/api/momentum/scan":
            params = parse_qs(parsed.query)
            from .features.momentum_scan import momentum_scan
            try:
                _json_response(self, momentum_scan(
                    top_n=int(params.get("top_n", ["10"])[0]),
                    force=params.get("force", ["false"])[0] == "true"))
            except Exception as exc:
                _json_response(self, {"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if parsed.path == "/api/backtest/portfolio":
            params = parse_qs(parsed.query)
            rules = load_rules()
            result = run_portfolio_backtest_from_csvs(
                symbols=_symbols_from_params(params),
                rules=rules,
                config=BacktestConfig.from_rules(rules),
                start=_parse_date_param(params.get("start", [None])[0]),
                end=_parse_date_param(params.get("end", [None])[0]),
            )
            _json_response(self, to_plain_dict(result))
            return
        if parsed.path == "/api/robustness":
            params = parse_qs(parsed.query)
            rules = load_rules()
            payload = run_robustness(
                symbols=_symbols_from_params(params),
                rules=rules,
                start=_parse_date_param(params.get("start", [None])[0]),
                end=_parse_date_param(params.get("end", [None])[0]),
            )
            _json_response(self, payload)
            return
        if parsed.path.startswith("/api/report/symbol/"):
            symbol = parsed.path.split("/")[-1].upper()
            try:
                report = synthesize_latest_report(symbol=symbol)
                _json_response(self, report)
            except RuntimeError as exc:
                _json_response(self, {"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
            return
        if parsed.path == "/" or parsed.path == "/index.html":
            self._serve_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        if not _auth_ok(self):
            return
        parsed = urlparse(self.path)
        if parsed.path == "/api/scan/trigger":
            params = parse_qs(parsed.query)
            body = _read_body(self)
            demo = bool(body.get("demo", params.get("demo", ["false"])[0] == "true"))
            symbols = body.get("symbols")
            result = run_scan(demo=demo, symbols=symbols, persist=True)
            _json_response(self, to_plain_dict(result))
            return
        if parsed.path == "/api/portfolio/positions":
            body = _read_body(self)
            store = PortfolioStore()
            position = store.add_position(
                symbol=body["symbol"],
                buy_price=float(body["buy_price"]),
                quantity=float(body["quantity"]),
                buy_date=body.get("buy_date"),
            )
            pnl = calculate_pnl(store.list_positions(), latest_prices_from_scan())
            log_portfolio_features(pnl)
            _json_response(self, {"position": position, "pnl": pnl}, status=HTTPStatus.CREATED)
            return
        if parsed.path == "/api/report/generate":
            body = _read_body(self)
            symbol = body.get("symbol")
            force = bool(body.get("force", False))
            try:
                report = synthesize_latest_report(symbol=symbol, force=force)
                _json_response(self, report)
            except RuntimeError as exc:
                _json_response(self, {"error": str(exc)}, status=HTTPStatus.SERVICE_UNAVAILABLE)
            return
        if parsed.path == "/api/model/train":
            body = _read_body(self)
            rules = load_rules()
            families = body.get("families")
            if isinstance(families, str):
                families = [item.strip() for item in families.split(",") if item.strip()]
            payload = train_model_suite(
                symbols=_symbols_from_params({}, body),
                start=_parse_date_param(body.get("start")),
                end=_parse_date_param(body.get("end")),
                rules=rules,
                families=families,
            )
            _json_response(self, payload, status=HTTPStatus.CREATED)
            return
        if parsed.path == "/api/optimize":
            body = _read_body(self)
            rules = load_rules()
            cfg = OptimizerConfig(
                years=int(body.get("years", 3)),
                n_trials=int(body.get("n_trials", 200)),
                target_return_pct=float(body.get("target_return", 10.0)),
                target_win_rate=float(body.get("target_win_rate", 60.0)),
                min_final_trades=int(body.get("min_trades", 30)),
                max_drawdown_pct=float(body.get("max_drawdown", 20.0)),
                seed=int(body.get("seed", 42)),
                refresh_before=bool(body.get("refresh", False)),
            )
            payload = run_optimization(
                symbols=_symbols_from_params({}, body),
                rules=rules,
                config=cfg,
                apply_if_valid=bool(body.get("apply", False)),
            )
            _json_response(self, payload, status=HTTPStatus.CREATED)
            return
        if parsed.path == "/api/mr/positions":
            body = _read_body(self)
            from .features.position_manager import PositionStore
            try:
                pos = PositionStore().add(
                    symbol=body["symbol"], entry_date=body.get("entry_date") or str(__import__("datetime").date.today()),
                    entry_price=float(body["entry_price"]), stop=float(body["stop"]),
                    target=float(body["target"]), max_hold_days=int(body.get("max_hold_days", 8)),
                    qty=int(body["qty"]),
                )
                _json_response(self, {"position": pos}, status=HTTPStatus.CREATED)
            except Exception as exc:
                _json_response(self, {"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/data/update":
            if not _DATA_UPDATE["running"]:
                threading.Thread(target=_do_data_update, daemon=True).start()
            _json_response(self, _DATA_UPDATE, status=HTTPStatus.ACCEPTED)
            return
        if parsed.path == "/api/momentum/positions":
            body = _read_body(self)
            from .features.position_manager import PositionStore
            try:
                pos = PositionStore().add(
                    symbol=body["symbol"], entry_date=body.get("entry_date") or str(__import__("datetime").date.today()),
                    entry_price=float(body["entry_price"]), stop=0.0, target=0.0,
                    max_hold_days=0, qty=int(body["qty"]), kind="momentum",
                )
                _json_response(self, {"position": pos}, status=HTTPStatus.CREATED)
            except Exception as exc:
                _json_response(self, {"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/chat":
            body = _read_body(self)
            message = body.get("message", "").strip()
            symbol = body.get("symbol")
            if not message:
                _json_response(self, {"error": "Message is required"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                response = handle_chat(message, symbol=symbol)
                _json_response(self, {"response": response})
            except Exception as exc:
                _json_response(self, {"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_DELETE(self) -> None:  # noqa: N802
        if not _auth_ok(self):
            return
        parsed = urlparse(self.path)
        if parsed.path == "/api/portfolio/positions":
            PortfolioStore().clear()
            _json_response(self, {"status": "cleared"})
            return
        if parsed.path == "/api/mr/positions":
            params = parse_qs(parsed.query)
            pid = params.get("id", [None])[0]
            from .features.position_manager import PositionStore as MRStore
            ok = MRStore().close(pid, reason=params.get("reason", ["manual"])[0]) if pid else False
            _json_response(self, {"closed": ok, "id": pid})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _serve_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        raw = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main() -> None:
    # Env-aware defaults so containers / PaaS (Render, Railway, Fly) work without flags:
    # HOST=0.0.0.0 to accept external connections, PORT injected by the platform.
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), StockAgentHandler)
    if _AUTH_PASS:
        print(f"[auth] Basic auth ON (user='{_AUTH_USER}')")
    elif args.host not in ("127.0.0.1", "localhost"):
        print("[auth] WARNING: listening on a public interface with NO password. "
              "Set APP_PASSWORD to protect this deployment.")
    else:
        print("[auth] disabled (local only). Set APP_PASSWORD to require a login.")
    print(f"Serving VN30 T+2 MVP at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
