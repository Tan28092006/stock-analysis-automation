# 📈 VN Swing-Trading System — Dual-Engine, Regime-Orchestrated

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![ML](https://img.shields.io/badge/ML-LightGBM%20%7C%20Isotonic%20Calibration-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Data](https://img.shields.io/badge/Data-vnstock%20%7C%20yfinance-00C4CC?style=for-the-badge)](https://github.com/thinh-vu/vnstock)

Hệ thống nghiên cứu **lướt sóng T+ cho thị trường Việt Nam**, gồm các nhánh rules, momentum và mean-reversion. Universe giao dịch hiện tại theo `configs/universe_vn30.json`; universe thu thập dữ liệu có thể rộng hơn. Tín hiệu chạy **cuối phiên (EOD)**, không real-time.

> **Mốc bằng chứng 30/09/2026:** xem [290 replay PIT và giới hạn thống kê](docs/audits/2026-09-30-strategy-reassessment.md), [chẩn đoán tần suất lệnh và nghiên cứu momentum từ sách](docs/audits/2026-09-30-momentum-books.md), [vận hành khối ngoại tự động](docs/FOREIGN-FLOWS.md). Kết quả cũ bên dưới không thay thế các báo cáo này. Chưa có chiến lược được chứng nhận lợi thế tiền thật; ML đang tắt.

> **Kiểm toán 22/09/2026: CHƯA ĐẠT gate dữ liệu/leakage.** Đã tìm thấy đánh giá ensemble in-sample, preprocessing nhìn tương lai, nhãn MR chưa chín và thiếu provenance ledger. Các số backtest/calibration dưới đây là kết quả nghiên cứu cũ, không phải hiệu quả đã chứng nhận ngoài mẫu. Xem [báo cáo kiến trúc, bằng chứng và điều kiện trước retrain](docs/audits/2026-09-22-architecture-and-leakage.md). Sửa universe/EOD/ledger không đồng nghĩa model đã sạch; chưa retrain hoặc thay artifact trong đợt kiểm toán này.

> **Cập nhật sau sửa:** code đã có split theo ngày/purge nhãn, holdout độc lập, preprocessing train-only, calibration MR không refit và gate model theo thời điểm. Xem [chi tiết sửa và test](docs/audits/2026-09-22-training-remediation.md). Artifact legacy bị chặn khỏi ML override/probability buy; chưa được thay bằng model mới. Còn 76 dòng giá sai, thiếu provenance/PIT universe; không chạy retrain/promotion production trước khi xác minh.

> [!IMPORTANT]
> Công cụ nghiên cứu cá nhân, **không phải lời khuyên đầu tư**, không tự đặt lệnh.

> **Giả thuyết nghiên cứu:** hiệu quả có thể khác theo regime. Banner thị trường không chứng minh đã chọn đúng chiến lược; momentum hiện tại không có hard gate RISK_ON. Bắt đáy cũng không bảo đảm kiếm tiền trong panic.

---

## 1. Hai động cơ + một điều phối

### 🚀 CORE — Momentum (ăn bull)
Momentum rule-only; chưa chứng minh lợi thế thống kê vượt VNINDEX:
- **12-1 momentum** (return t-252 → t-21, bỏ tháng gần nhất vì đảo chiều ngắn hạn).
- **Trọng số nghịch-vol** — mã vol thấp tỷ trọng cao hơn; không phải covariance-based risk parity.
- **Vol-targeting** — exposure = target_vol / vol thị trường → tự giảm khi vol cao.
- **Buffering** — giữ tới khi rớt khỏi top-2N (giảm turnover/phí).
- Replay chuẩn tái cân bằng theo tháng; picks trên dashboard là danh sách ứng viên, không phải 10 lệnh mới mỗi ngày. Cảnh báo thoát trên dashboard kiểm tra hằng ngày, chưa đồng nhất với replay tháng. Nhánh tuần/tín hiệu sách mới chỉ có trong research.
- Backtest VN30 (fill mở-cửa, phí 0.4%): trên rổ **30 mã VN30 hiện tại áp ngược** cho FULL +157–190%, Sharpe ~0.9. **⚠️ Đây là số survivorship-inflated** — kiểm bằng rổ *point-in-time* (top-30 theo giá trị giao dịch, membership xoay theo thời gian) thì thực tế chỉ **~+33%, Sharpe ~0.3, DD ~−56%**, xấp xỉ index. Coi các số cố định-rổ là **cận trên lạc quan**, không phải kỳ vọng thực.

### 🎯 SATELLITE — Bắt đáy / Mean-Reversion (crash alpha)
"Thợ săn kèo béo" — chỉ bắn khi có capitulation + đảo chiều, thoát nhanh:
- **Cổng cứng**: RSI14<30 (VN30: <35) + chạm dải BB dưới + (nến đảo chiều & climax volume | VSA stopping-volume) + RR≥0.5 (room về Kijun) + mây Ichimoku không chặn.
- **Rủi ro**: stop 3×ATR, target Kijun, time exit 15 phiên; không bảo đảm thoát khi sàn/không thanh khoản. Replay chuẩn dùng T+3 bảo thủ; UI và replay vẫn có khác biệt execution cần xử lý trước triển khai.
- **Lớp ML — P(win)**: `ml.enabled=false`, `override_enabled=false`. Artifact cũ không được xem là xác suất thắng ex-ante đã chứng nhận.
- Replay PIT 30/09: MR bracket cuối 2022 −1,738%, YTD-2026 +5,011%; đọc đúng cửa sổ, phí và dữ liệu trong báo cáo, không thay bằng số nghiên cứu legacy.

### 🧭 Regime Orchestrator
Đọc chế độ từ **VNINDEX vs EMA50 + breadth** (% cổ phiếu trên SMA20):

| Chế độ | Điều kiện | Khuyến nghị |
|---|---|---|
| 🟢 RISK_ON | index > EMA50 & breadth ≥ 40% | ưu tiên Momentum |
| 🔴 PANIC | index < EMA50 | cảnh báo thị trường yếu; không tự tắt momentum |
| 🟡 GRIND | index > EMA50 nhưng breadth yếu | thận trọng, ưu tiên tiền mặt |

Banner chỉ là ngữ cảnh giao diện, không thay thế rule hoặc điều kiện kiểm định chiến lược.

---

## 2. Kết quả backtest legacy (không được chứng nhận out-of-sample)

Bảng dưới giữ để truy vết nghiên cứu cũ, không chứng minh hai engine bù nhau hoặc đọc đúng regime:

| Năm | Regime hệ phát hiện | VNINDEX | 🚀 Momentum | 🎯 Bắt đáy |
|---|---|---|---|---|
| **2022** crash | PANIC 66% | −34% | −40% (bẫy) | ~0% (cứu vốn) |
| **2024** hồi phục | RISK_ON 63% | +12% | +19% ✅ | +0.1% |
| **2025** bull mạnh | RISK_ON 59% | +40% | +24% | −0.6% |
| **2026** crash+choppy | PANIC 31% | +4% | −11% | +6.7% ✅ |

> ⚠️ **Cột Momentum tính trên rổ VN30 hiện tại áp ngược → survivorship-inflated.** Trên rổ point-in-time (membership xoay) các con số này yếu hơn đáng kể và momentum chỉ xấp xỉ index. Đọc cùng §6.

Các giả thuyết market gate, high52w và khối ngoại đã có kết quả không ổn định giữa các giai đoạn. Điều đó không chứng minh chúng luôn vô dụng hoặc baseline luôn tốt; dùng protocol hiện tại, đối chứng và phí thực tế để đánh giá. Không có bảo đảm cứu vốn trong crash hoặc luôn vượt buy-and-hold.

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
