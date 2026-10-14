"""Fail-closed dashboard input check, not a certificate of profitable trading."""
from pathlib import Path

BLOCKED_NOTE = ("DỮ LIỆU KHÔNG HỢP LỆ — đã chặn khuyến nghị mua/bán. "
                "Cần đủ rổ VN30, giá EOD mới nhất và lịch sử hợp lệ. "
                "Không sử dụng tín hiệu/cache cũ để vào lệnh; sổ vị thế không bị thay đổi.")


def readiness(prices_dir: Path, *, now=None) -> dict:
    # Lazy import: paper_runner uses the pure scan functions, not dashboard wrappers.
    from ..config import load_universe
    from ..pipeline.paper_runner import assess_market_data
    try:
        return assess_market_data(prices_dir, list(load_universe()['symbols']), now=now)
    except Exception:
        return {'data_ready': False, 'session': None,
                'issues': {'configuration': ['Cannot validate the configured universe/data']}}


def blocked(status: dict, mode: str) -> dict:
    return dict(mode=mode, status='blocked_data', live_approved=False, readiness=status,
                active=False, data_date=None, symbols_scanned=0, scanned_symbols=[],
                market={'state': 'UNKNOWN', 'date': None}, model={'available': False},
                picks=[], buys=[], watches=[], prob_buys=[], recent_signals=[],
                positions=[], sell_alerts=[], position_checks_available=False,
                exposure_pct=0, market_vol_pct=None, note=BLOCKED_NOTE)
