# 📈 VN30 Stock Technical Analysis Agent

> Hệ thống tự động phân tích kỹ thuật top 5 cổ phiếu tăng trưởng mạnh nhất trong rổ VN30, sử dụng LLM để tổng hợp nhận định và gửi báo cáo qua email hàng ngày.

---

## 🧠 Kiến trúc hệ thống

```
Yahoo Finance (VN30 data)
        │
        ▼
┌───────────────────────┐
│  Bước 1: Screening    │  → Lọc top 5 cổ phiếu tăng trưởng mạnh nhất 14 ngày
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Bước 2: Feature Eng  │  → Tính EMA, MACD, RSI, Bollinger Bands, AO, Breakout
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Bước 3: LLM Analysis │  → Groq API phân tích xu hướng, điểm mua/bán
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  Bước 4: Report + Mail│  → Xuất .docx + gửi email tự động
└───────────────────────┘
```

---

## ✨ Tính năng

| Tính năng | Mô tả |
|---|---|
| 📊 Screening tự động | Lọc top 5 cổ phiếu VN30 tăng mạnh nhất 14 ngày gần nhất |
| 📐 Technical indicators | EMA (20/50/100), MACD, RSI, Bollinger Bands, Awesome Oscillator |
| 🚨 Breakout detection | Cảnh báo khi volume > 1.5× trung bình 20 ngày |
| 🤖 LLM analysis | Groq API (GPT-class model) phân tích và nhận định chuyên sâu |
| 📄 Word report | Xuất báo cáo định dạng `.docx` có cấu trúc |
| 📧 Auto email | Gửi báo cáo tự động qua Gmail |

---

## 🛠️ Technical Indicators

### EMA (Exponential Moving Average)
```
EMA_t = α × Price_t + (1 − α) × EMA_{t−1}
α = 2 / (period + 1)
```
Dùng 3 chu kỳ: EMA20 (ngắn hạn), EMA50 (trung hạn), EMA100 (dài hạn). Giao cắt giữa các EMA là tín hiệu xu hướng.

### MACD
```
MACD     = EMA(12) − EMA(26)
Signal   = EMA(MACD, 9)
Histogram = MACD − Signal
```
Histogram dương và tăng → momentum tăng. Giao cắt MACD/Signal → điểm mua/bán.

### RSI (Relative Strength Index)
```
RSI = 100 − 100 / (1 + RS)
RS  = avg_gain / avg_loss  (period=14)
```
RSI > 70 → overbought. RSI < 30 → oversold.

### Bollinger Bands
```
Middle = SMA(20)
Upper  = SMA(20) + 2σ
Lower  = SMA(20) − 2σ
```
Giá chạm Upper → kháng cự. Giá chạm Lower → hỗ trợ. Bands thu hẹp → chuẩn bị breakout.

### Awesome Oscillator (AO)
```
AO = SMA(Median Price, 5) − SMA(Median Price, 34)
Median Price = (High + Low) / 2
```

---

## ⚙️ Cài đặt

### Yêu cầu
- Python 3.9+
- Groq API key ([đăng ký miễn phí](https://console.groq.com))
- Gmail App Password (bật 2FA → tạo App Password)

### Cài thư viện
```bash
pip install yfinance pandas numpy python-docx groq yagmail
```

### Thiết lập biến môi trường
```bash
# Linux / macOS
export GROQ_API_KEY="gsk_..."
export EMAIL_USER="your_email@gmail.com"
export EMAIL_PASS="your_app_password"

# Windows
set GROQ_API_KEY=gsk_...
set EMAIL_USER=your_email@gmail.com
set EMAIL_PASS=your_app_password
```

### Chạy
```bash
python main.py
```

---

## 📁 Cấu trúc project

```
vn30-analysis/
├── main.py              # Pipeline chính
├── outputs/             # Báo cáo Word được lưu tại đây
│   └── Bao_cao_top5_VN30.docx
└── README.md
```

---

## 📊 Nguồn dữ liệu

Dữ liệu lấy từ **Yahoo Finance** qua thư viện `yfinance`, với suffix `.VN` cho cổ phiếu niêm yết tại HOSE.

```python
yf.download("ACB.VN", start=start_date, end=end_date)
```

> **Lưu ý:** Yahoo Finance cung cấp dữ liệu delay ~15–20 phút và có thể thiếu một số phiên giao dịch so với HOSE thực tế. Phù hợp cho mục đích học tập và nghiên cứu. Với hệ thống production, nên dùng SSI iBoard API hoặc VNDirect data feed.

---

## 🔄 Tự động hóa (chạy hàng ngày)

### Linux/macOS — cron job
```bash
# Chạy lúc 17:30 mỗi ngày (sau khi HOSE đóng cửa)
30 17 * * 1-5 cd /path/to/project && python main.py
```

### GitHub Actions
```yaml
name: Daily VN30 Analysis
on:
  schedule:
    - cron: '30 10 * * 1-5'  # 17:30 ICT = 10:30 UTC
jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install yfinance pandas numpy python-docx groq yagmail
      - run: python main.py
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          EMAIL_USER: ${{ secrets.EMAIL_USER }}
          EMAIL_PASS: ${{ secrets.EMAIL_PASS }}
```

---

## ⚠️ Disclaimer

> Dự án này được xây dựng cho mục đích **học tập và nghiên cứu**. Các phân tích từ hệ thống không phải là khuyến nghị đầu tư tài chính. Mọi quyết định giao dịch đều là trách nhiệm của người dùng.

---

## 👤 Tác giả

**Lê Nguyễn Minh Tân** — Data Science, HCMUT  
`tan.le28092k6@hcmut.edu.vn`
