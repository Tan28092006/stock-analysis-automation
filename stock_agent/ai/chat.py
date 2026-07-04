from __future__ import annotations

import os
from typing import Any
import re
from pathlib import Path
import pandas as pd

from ..constants import LATEST_SCAN_PATH, PRICE_CACHE_DIR
from ..data.repository import read_json
from ..config import load_universe
from .ollama_client import OllamaChatClient
from .env import load_dotenv


def get_chat_client() -> OllamaChatClient | Any:
    """Unified LLM client factory.
    
    Prioritizes Nvidia NIM API client if API key is configured,
    otherwise falls back to local Ollama instance.
    """
    load_dotenv()
    
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if api_key:
        try:
            from .nvidia_client import NvidiaChatClient
            return NvidiaChatClient()
        except Exception as e:
            print(f"  ⚠ Failed to initialize Nvidia client: {e}")
            
    return OllamaChatClient()


def extract_symbol_from_message(message: str) -> str | None:
    """Extract and validate Vietnamese stock ticker symbol from message."""
    # Find all 3-letter combinations
    words = re.findall(r"\b[a-zA-Z]{3}\b", message)
    if not words:
        return None
        
    # Check against configured universe
    try:
        universe = load_universe()
        universe_symbols = {s.upper() for s in universe.get("symbols", [])}
    except Exception:
        universe_symbols = set()
        
    for w in words:
        w_upper = w.upper()
        if w_upper in universe_symbols:
            return w_upper
        # Fallback: check if local price cache file exists
        csv_path = PRICE_CACHE_DIR / f"{w_upper}.csv"
        if csv_path.exists():
            return w_upper
            
    return None


def get_symbol_context(symbol: str) -> dict:
    """Load latest scan candidate info and recent 10-day price history."""
    symbol = symbol.upper()
    scan = read_json(LATEST_SCAN_PATH, default={})
    candidates = scan.get("candidates", [])
    
    candidate = None
    for c in candidates:
        if c.get("symbol") == symbol:
            candidate = c
            break
            
    prices = []
    csv_path = PRICE_CACHE_DIR / f"{symbol}.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            # Fetch last 10 rows
            last_rows = df.tail(10)
            for _, row in last_rows.iterrows():
                prices.append({
                    "date": str(row.get("date")),
                    "open": float(row.get("open", 0)),
                    "high": float(row.get("high", 0)),
                    "low": float(row.get("low", 0)),
                    "close": float(row.get("close", 0)),
                    "volume": int(row.get("volume", 0))
                })
        except Exception as e:
            print(f"  ⚠ Error reading CSV for {symbol}: {e}")
            
    return {
        "candidate": candidate,
        "recent_prices": prices
    }


def handle_chat(message: str, symbol: str | None = None) -> str:
    """Compile stock technical indicators, rule checks and recent prices,
    then invoke LLM to generate swing trading entry point analysis.
    """
    if not symbol:
        symbol = extract_symbol_from_message(message)
        
    client = get_chat_client()
    context_text = ""
    symbol_found = False
    
    if symbol:
        symbol = symbol.upper()
        data = get_symbol_context(symbol)
        candidate = data.get("candidate")
        recent_prices = data.get("recent_prices", [])
        
        if candidate or recent_prices:
            symbol_found = True
            context_text += f"THÔNG TIN BỐI CẢNH CỦA MÃ CHỨNG KHOÁN: {symbol}\n"
            if candidate:
                context_text += f"- Quyết định chiến lược kỹ thuật hiện tại: {candidate.get('decision')} (Score: {candidate.get('score')})\n"
                context_text += f"- Giá đóng cửa gần nhất: {candidate.get('latest_close')} (Ngày: {candidate.get('latest_date')})\n"
                context_text += f"- Tỷ lệ phân bổ vốn đề xuất (allocation weight): {candidate.get('allocation_weight', 0.0) * 100:.2f}%\n"
                
                # Risk plan
                rp = candidate.get("risk_plan")
                if rp:
                    context_text += "- Kế hoạch quản trị rủi ro:\n"
                    context_text += f"  * Điểm mua tham chiếu (Entry Reference): {rp.get('entry_reference')}\n"
                    context_text += f"  * Điểm cắt lỗ (Stop Loss - SL): {rp.get('stop_loss')} ({rp.get('stop_loss_pct')}%)\n"
                    context_text += f"  * Điểm chốt lời 1 (Take Profit 1 - TP1): {rp.get('take_profit_1')}\n"
                    context_text += f"  * Điểm chốt lời 2 (Take Profit 2 - TP2): {rp.get('take_profit_2')}\n"
                    context_text += f"  * Tỷ lệ Reward/Risk (R:R): {rp.get('reward_risk')}\n"
                    context_text += f"  * Thời gian nắm giữ dự kiến: {rp.get('holding_period_days')} phiên (T+2)\n"
                else:
                    context_text += "- Không có kế hoạch quản trị rủi ro vì mã bị REJECT.\n"
                    
                # ML signals
                ml = candidate.get("model_signal")
                if ml and ml.get("status") == "available":
                    context_text += f"- Tín hiệu Machine Learning (Vai trò tham khảo): Xác suất tăng giá là {ml.get('probability', 0.0) * 100:.1f}% (Model: {ml.get('model_family')})\n"
                
                # Backtest & Robustness
                bt = candidate.get("backtest_summary") or {}
                if bt.get("win_rate") is not None:
                    context_text += "- Kết quả backtest lịch sử của mã này:\n"
                    context_text += f"  * Tỷ lệ thắng (Win Rate): {bt.get('win_rate'):.1f}%\n"
                    context_text += f"  * Số lượng lệnh giao dịch: {bt.get('total_trades')}\n"
                    context_text += f"  * Kỳ vọng (Expectancy): {bt.get('expectancy_pct', 0.0):.3f}%\n"
                
                rob = candidate.get("robustness_summary") or {}
                if rob.get("monte_carlo") is not None:
                    context_text += "- Kết quả kiểm nghiệm tính bền vững (Robustness):\n"
                    mc = rob.get("monte_carlo", {})
                    context_text += f"  * Lợi nhuận kỳ vọng Monte Carlo p50: {mc.get('total_return_pct_p50', 0.0):.2f}%\n"
                    context_text += f"  * Drawdown cực đại Monte Carlo p95: {mc.get('max_drawdown_pct_p95', 0.0):.2f}%\n"
                    
                # Technical checks evidence
                evidence = candidate.get("evidence", [])
                if evidence:
                    context_text += "- Chi tiết kết quả kiểm tra các luật giao dịch kỹ thuật:\n"
                    for ev in evidence:
                        status = "ĐẠT (PASS)" if ev.get("passed") else "KHÔNG ĐẠT (FAIL)"
                        val_str = f" (Giá trị: {ev.get('value')})" if ev.get("value") is not None else ""
                        context_text += f"  * Luật '{ev.get('name')}': {status} - {ev.get('detail')}{val_str} [Đóng góp: {ev.get('points')} điểm]\n"
                        
                # Warnings
                warnings = candidate.get("warnings", [])
                if warnings:
                    context_text += "- Cảnh báo hệ thống cho mã này:\n"
                    for w in warnings:
                        context_text += f"  * CẢNH BÁO: {w}\n"
            
            # Recent prices table
            if recent_prices:
                context_text += "\n- Diễn biến giá 10 phiên gần đây nhất:\n"
                context_text += "  Ngày | Mở cửa (Open) | Cao nhất (High) | Thấp nhất (Low) | Đóng cửa (Close) | Khối lượng (Volume)\n"
                for p in recent_prices:
                    context_text += f"  {p['date']} | {p['open']:,.0f} | {p['high']:,.0f} | {p['low']:,.0f} | {p['close']:,.0f} | {p['volume']:,}\n"

    # General overview context if symbol is not found/specified
    if not symbol_found:
        scan = read_json(LATEST_SCAN_PATH, default={})
        candidates = scan.get("candidates", [])
        buy_count = sum(1 for c in candidates if c.get("decision") == "BUY_SETUP")
        watch_count = sum(1 for c in candidates if c.get("decision") == "WATCH")
        reject_count = sum(1 for c in candidates if c.get("decision") == "REJECT")
        
        context_text += "THÔNG TIN TỔNG QUAN HỆ THỐNG SCAN VN30 GẦN NHẤT:\n"
        context_text += f"- ID lượt scan: {scan.get('scan_id', 'N/A')}\n"
        context_text += f"- Ngày thực hiện: {scan.get('created_at', 'N/A')}\n"
        context_text += f"- Universe: {scan.get('universe_name', 'N/A')}\n"
        context_text += f"- Tổng số mã quét: {scan.get('symbols_scanned', 0)}\n"
        context_text += f"- Số mã ĐỦ ĐIỀU KIỆN MUA (BUY_SETUP): {buy_count} mã\n"
        context_text += f"- Số mã CẦN THEO DÕI (WATCH): {watch_count} mã\n"
        context_text += f"- Số mã BỊ LOẠI (REJECT): {reject_count} mã\n"
        
        buy_symbols = [c.get("symbol") for c in candidates if c.get("decision") == "BUY_SETUP"]
        if buy_symbols:
            context_text += f"- Các mã BUY_SETUP: {', '.join(buy_symbols)}\n"
        watch_symbols = [c.get("symbol") for c in candidates if c.get("decision") == "WATCH"]
        if watch_symbols:
            context_text += f"- Các mã WATCH: {', '.join(watch_symbols)}\n"

    # Construct prompts
    system_prompt = (
        "Bạn là trợ lý AI chuyên nghiệp phân tích kỹ thuật chứng khoán Việt Nam (VN30), tối ưu cho chiến lược Swing trading T+2.\n"
        "Nhiệm vụ của bạn là trả lời câu hỏi của người dùng một cách chính xác, dựa HOÀN TOÀN và CHỈ DỰA trên thông tin bối cảnh (context) kỹ thuật được cung cấp.\n\n"
        "QUY TẮC BẮT BUỘC:\n"
        "1. KHÔNG được bịa đặt thông tin, số liệu giá, hoặc kết quả chỉ báo kỹ thuật nằm ngoài bối cảnh được cung cấp.\n"
        "2. TUYỆT ĐỐI không đưa các tin tức, tin đồn, nhận định vĩ mô, hoặc phân tích cơ bản (FA) ngoại trừ dữ liệu OHLCV và các chỉ báo kỹ thuật có sẵn trong bối cảnh.\n"
        "3. Nếu người dùng hỏi về một mã chứng khoán có trong bối cảnh, hãy trình bày một phân tích có cấu trúc rõ ràng:\n"
        "   - Nhận định chung (về xu hướng EMA, xu hướng trung hạn, ngắn hạn).\n"
        "   - Đánh giá các luật kỹ thuật (RSI, MACD, Volume confirmation, Ichimoku, ADX, VWAP).\n"
        "   - Điểm vào lệnh & Quản trị rủi ro (Entry, Stop Loss, Take Profit, tỷ lệ R:R, holding period). Giải thích rõ tại sao đạt điểm BUY_SETUP hoặc WATCH/REJECT.\n"
        "   - Tín hiệu bổ sung (xác suất Machine Learning, kết quả Backtest/Robustness lịch sử).\n"
        "4. Nếu người dùng hỏi về mã không có trong dữ liệu và không thể tìm thấy thông tin giá, hãy trả lời lịch sự rằng hệ thống hiện tại chưa có dữ liệu lịch sử hoặc kết quả quét cho mã này.\n"
        "5. Phản hồi bằng tiếng Việt tự nhiên, chuyên nghiệp và có định dạng Markdown rõ ràng, dễ nhìn."
    )
    
    user_payload = f"CÂU HỎI NGƯỜI DÙNG: {message}\n\nTHÔNG TIN BỐI CẢNH DỰ ÁN:\n{context_text}"
    
    # Check client availability
    if hasattr(client, "is_available") and not client.is_available():
        api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
        if api_key:
            try:
                from .nvidia_client import NvidiaChatClient
                client = NvidiaChatClient()
            except Exception:
                raise RuntimeError(
                    "Ollama is offline and Nvidia NIM client could not be loaded. "
                    "Please start Ollama (ollama serve) or check your API key."
                )
        else:
            raise RuntimeError(
                "Ollama is offline and no NVIDIA_API_KEY is configured. "
                "Please start Ollama with: ollama serve"
            )
            
    return client.chat_text(system_prompt, user_payload, temperature=0.3)
