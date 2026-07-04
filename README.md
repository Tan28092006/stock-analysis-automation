# 📈 VN Swing-Trading System — Dual-Engine, Regime-Orchestrated

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![ML](https://img.shields.io/badge/ML-LightGBM%20%7C%20Isotonic%20Calibration-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Data](https://img.shields.io/badge/Data-vnstock%20%7C%20yfinance-00C4CC?style=for-the-badge)](https://github.com/thinh-vu/vnstock)

Hệ thống hỗ trợ **lướt sóng T+ cho thị trường Việt Nam (VN100)**, xây theo tư duy **định lượng cá nhân**: đọc chế độ thị trường (regime), rồi tự khuyến nghị đúng "động cơ" cho từng chế độ. Tín hiệu chạy **cuối phiên (EOD)**, không real-time, và **mọi thành phần đều được backtest chặt** (kiểm out-of-sample, chống leak, chống overfit).

> [!IMPORTANT]
> Công cụ nghiên cứu cá nhân, **không phải lời khuyên đầu tư**, không tự đặt lệnh.

> **Triết lý:** không một tín hiệu đơn nào thắng mọi chế độ. Bull → đi theo đà (momentum); crash → bắt đáy (mean-reversion); chợ hẹp → cầm tiền. Hệ đọc regime và **chỉ đúng công cụ**, thay vì gom mọi đề xuất lại (dễ đánh momentum ngay đợt sập).

---

## 1. Hai động cơ + một điều phối

### 🚀 CORE — Momentum (ăn bull)
Quant-grade momentum, ưu tiên khi RISK_ON:
- **12-1 momentum** (return t-252 → t-21, bỏ tháng gần nhất vì đảo chiều ngắn hạn).
- **Trọng số nghịch-vol** (risk parity) — mã vol thấp tỷ trọng cao hơn.
- **Vol-targeting** — exposure = target_vol / vol thị trường → tự giảm khi vol cao.
- **Buffering** — giữ tới khi rớt khỏi top-2N (giảm turnover/phí).
- Backtest VN30 (fill mở-cửa, phí 0.4%): trên rổ **30 mã VN30 hiện tại áp ngược** cho FULL +157–190%, Sharpe ~0.9. **⚠️ Đây là số survivorship-inflated** — kiểm bằng rổ *point-in-time* (top-30 theo giá trị giao dịch, membership xoay theo thời gian) thì thực tế chỉ **~+33%, Sharpe ~0.3, DD ~−56%**, xấp xỉ index. Coi các số cố định-rổ là **cận trên lạc quan**, không phải kỳ vọng thực.

### 🎯 SATELLITE — Bắt đáy / Mean-Reversion (crash alpha)
"Thợ săn kèo béo" — chỉ bắn khi có capitulation + đảo chiều, thoát nhanh:
- **Cổng cứng**: RSI14<30 (VN30: <35) + chạm dải BB dưới + (nến đảo chiều & climax volume | VSA stopping-volume) + RR≥0.5 (room về Kijun) + mây Ichimoku không chặn.
- **Rủi ro**: stop 3×ATR (bảo hiểm dao rơi), target Kijun, giữ ≤15 phiên, khóa T+2.
- **Lớp ML — P(win)**: meta-labeling (LightGBM + isotonic calibration) gán xác suất thắng *ex-ante* (không leak); calibrate thật (P≥55% → thắng ~56%).
- Backtest: 2022 crash **hòa vốn** khi VNINDEX −34%; YTD-2026 **+6.7%** khi index +4.1%.

### 🧭 Regime Orchestrator
Đọc chế độ từ **VNINDEX vs EMA50 + breadth** (% cổ phiếu trên SMA20):

| Chế độ | Điều kiện | Khuyến nghị |
|---|---|---|
| 🟢 RISK_ON | index > EMA50 & breadth ≥ 40% | ưu tiên Momentum |
| 🔴 PANIC | index < EMA50 | ưu tiên Bắt đáy (momentum tạm dừng) |
| 🟡 GRIND | index > EMA50 nhưng breadth yếu | thận trọng, ưu tiên tiền mặt |

Dashboard hiện banner khuyến nghị + **làm mờ engine không phù hợp** — bạn phán quyết cuối.

---

## 2. Kết quả backtest (trung thực, đã kiểm chứng out-of-sample)

Đọc regime đúng mọi năm, hai engine bù nhau:

| Năm | Regime hệ phát hiện | VNINDEX | 🚀 Momentum | 🎯 Bắt đáy |
|---|---|---|---|---|
| **2022** crash | PANIC 66% | −34% | −40% (bẫy) | ~0% (cứu vốn) |
| **2024** hồi phục | RISK_ON 63% | +12% | +19% ✅ | +0.1% |
| **2025** bull mạnh | RISK_ON 59% | +40% | +24% | −0.6% |
| **2026** crash+choppy | PANIC 31% | +4% | −11% | +6.7% ✅ |

> ⚠️ **Cột Momentum tính trên rổ VN30 hiện tại áp ngược → survivorship-inflated.** Trên rổ point-in-time (membership xoay) các con số này yếu hơn đáng kể và momentum chỉ xấp xỉ index. Đọc cùng §6.

**Sự thật đã chấp nhận:** không timing nào thắng buy-and-hold trong siêu bull. Giá trị của hệ là **tham gia bull + cứu vốn/thêm alpha trong crash + kiểm soát drawdown**. Nhiều "cải tiến" hào nhoáng đã bị backtest chặt **loại bỏ** vì overfit: kill-switch EMA200, filter bear, factor high52w, "follow khối ngoại", **circuit-breaker %-giảm-nhanh** (whipsaw), và **excess-momentum gate** (chỉ thắng trên đúng rổ survivor, biến mất khi test point-in-time/VN100/random-subset).

---

## 3. Dashboard

Mở `http://127.0.0.1:8000` (chạy từ `stock_agent/app.py`):
- 🧭 Banner điều phối regime (tự khuyến nghị engine + làm mờ engine không hợp).
- 🚀 Panel Momentum: top mã + trọng số nghịch-vol + exposure + nút "Ghi" vị thế.
- 🎯 Panel Bắt đáy: kèo BUY_SETUP / prob-buy (P≥55%) + KL gợi ý + P(win) + kèo 120 ngày qua.
- 💰 Theo dõi vị thế cả 2 engine + **cảnh báo BÁN** tự động (stop/target/rớt-nhóm).
- 🔄 Nút **"Cập nhật dữ liệu"** (tải giá EOD mới) tách khỏi Re-scan.

API chính: `GET /api/mr/scan`, `GET /api/momentum/scan`, `POST /api/data/update`, `POST/DELETE /api/mr|momentum/positions`.

---

## 4. Kiến trúc

```
stock_agent/
  features/
    momentum_scan.py     # CORE: quant momentum (12-1, inv-vol, vol-target, buffer)
    mr_scan.py           # SATELLITE: bottom-fishing scan + P(win) + sizing + positions
    signal_engine.py     # rule scorers (momentum scorecard + mean_reversion mode)
    win_probability.py   # meta-labeling P(win): train + calibrate + predict
    market_regime.py     # regime classifier (VNINDEX EMA50 + breadth)
    position_manager.py  # sizing + vòng đời vị thế + cảnh báo BÁN
    indicators.py        # RSI, Bollinger, Ichimoku, ADX, VSA, ATR...
    backtest.py          # backtest engine (T+2, tick slippage, price-limit)
  agents/orchestrator.py # run_scan (parallel fetch, regime filter)
  pipeline/eod_update.py # job EOD 17:05: giá + khối ngoại + scan + alert bán
  data/providers.py      # vnstock VCI + yfinance fallback + local CSV
  data/foreign_flows.py  # khối ngoại (accumulator, chờ đủ dữ liệu)
  app.py                 # HTTP server + JSON API + serve dashboard
  cli.py                 # CLI cũ (scan/backtest/train/portfolio — vẫn dùng được)
web/index.html           # dashboard 1 file
configs/rules_mr.json    # config mean-reversion (baseline)
scratch/                 # toàn bộ script backtest/nghiên cứu (bằng chứng)
docs/DEPLOY.md           # hướng dẫn deploy web
```

---

## 5. Chạy nhanh

```bash
pip install -r requirements.txt

# 1) Lấy/cập nhật dữ liệu giá VN100 (vnstock, có resume). Cũng là job EOD.
python -m stock_agent.pipeline.eod_update

# 2) (tùy chọn) train model P(win)
python -c "from stock_agent.features.win_probability import train_and_save; print(train_and_save())"

# 3) Bật dashboard
python -m stock_agent.app --host 127.0.0.1 --port 8000   # -> http://127.0.0.1:8000
```

**Tự động 17:05 (T2-T6):** đăng ký `run_eod_update.bat` vào Windows Task Scheduler, hoặc cron trên Linux — xem [docs/DEPLOY.md](docs/DEPLOY.md).

---

## 6. Lưu ý trung thực

- **EOD, không real-time** — giá đóng cửa; cảnh báo stop theo phiên, không theo tick.
- **Survivorship (nặng, cả VN30)** — mọi rổ (VN30 lẫn VN100) đều là danh sách *hiện tại* áp ngược → số backtest bị thổi phồng. Đã đo trực tiếp: momentum VN30 cố-định-rổ +190%/Sharpe 0.88 nhưng rổ **point-in-time** (membership xoay theo giá trị giao dịch) chỉ **+33%/Sharpe 0.31/DD −56%**. **Đừng tin "VN30 chuẩn hơn"** — nó cũng inflated; coi số backtest là cận trên lạc quan, kỳ vọng thực xấp xỉ index.
- **Edge nhỏ & theo regime** — P(win) tối đa ~57% (không phải phép màu); thắng nhờ *R:R bất đối xứng × tilt xác suất nhỏ*.
- **Không phòng được flash-crash** — sập nhanh vài phiên: vol-target (60 ngày) + cảnh báo top-2N (12-1 bỏ 21 phiên gần nhất) đều quá chậm; đây là giới hạn cấu trúc của EOD, không phải bug. Sập từ-từ thì bắt đáy mới ăn.
- **Khối ngoại** — đã dựng hạ tầng thu thập; gác lại làm tín hiệu (dữ liệu nói "follow khối ngoại" là sai; manh mối contrarian chưa đủ mẫu).
- **Tests:** `python -m pytest tests/ -q` (91 pass).
