import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import glob
from docx import Document
from groq import Groq
import yagmail

# ==== Thiết lập biến môi trường ====
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise Exception("Bạn chưa set biến môi trường GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

# ==== Hàm chỉ báo kỹ thuật ====
def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def SMA(series, period):
    return series.rolling(window=period).mean()

def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd = ema_fast - ema_slow
    signal_line = EMA(macd, signal)
    hist = macd - signal_line
    return macd, signal_line, hist

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def BollingerBands(series, period=20, num_std=2):
    mid = SMA(series, period)
    std = series.rolling(window=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower

def AO(high, low, short=5, long=34):
    median_price = (high + low) / 2
    return SMA(median_price, short) - SMA(median_price, long)

# ==== Thư mục lưu file ====
save_path = "outputs"
os.makedirs(save_path, exist_ok=True)

# ==== Danh sách VN30 ====
VN30_TICKERS = [
    "ACB","BCM","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG","KDH","LPB",
    "MBB","MSN","MWG","NVL","PDR","PLX","POW","SAB","SHB","SSI","STB","TCB",
    "TPB","VCB","VHM","VIC","VJC","VNM","VPB","VRE"
]

# ==== Bước 1: Tính tăng trưởng 14 ngày ====
growth_data = []
end_date_14 = datetime.today()
start_date_14 = end_date_14 - timedelta(days=14)

for ticker in VN30_TICKERS:
    df_temp = yf.download(ticker + ".VN", start=start_date_14, end=end_date_14)
    if df_temp.empty:
        continue
    growth_pct = ((df_temp["Close"].iloc[-1] - df_temp["Close"].iloc[0]) / df_temp["Close"].iloc[0]) * 100
    growth_data.append((ticker, growth_pct))

# Sắp xếp giảm dần theo % tăng trưởng
growth_data.sort(key=lambda x: x[1], reverse=True)
top_5 = [x[0] for x in growth_data[:5]]

print("📈 Top 5 cổ phiếu VN30 tăng trưởng cao nhất 14 ngày gần đây:")
for t, g in growth_data[:5]:
    print(f"{t}: {g:.2f}%")

# ==== Bước 2: Lấy dữ liệu 28 ngày cho top 5 ====
end_date_28 = datetime.today()
start_date_28 = end_date_28 - timedelta(days=28)

for ticker in top_5:
    df = yf.download(ticker + ".VN", start=start_date_28, end=end_date_28)
    if df.empty:
        continue

    df["EMA20"] = EMA(df["Close"], 20)
    df["EMA50"] = EMA(df["Close"], 50)
    df["EMA100"] = EMA(df["Close"], 100)
    macd, signal, hist = MACD(df["Close"])
    df["MACD"] = macd
    df["Signal"] = signal
    df["MACD_Hist"] = hist
    df["RSI"] = RSI(df["Close"])
    upper, mid, lower = BollingerBands(df["Close"])
    df["BB_Upper"] = upper
    df["BB_Mid"] = mid
    df["BB_Lower"] = lower
    df["AO"] = AO(df["High"], df["Low"])
    vol_mean = df["Volume"].rolling(window=20).mean()
    df["Breakout"] = df["Volume"] > vol_mean * 1.5

    df_reset = df.reset_index()
    df_reset["Date"] = df_reset["Date"].dt.strftime("%Y-%m-%d")
    df_json_ready = df_reset.where(pd.notnull(df_reset), None)

    file_json = os.path.join(save_path, f"{ticker}.json")
    with open(file_json, "w") as f:
        json.dump(df_json_ready.to_dict(orient="records"), f, indent=2)

print("✅ Đã lưu dữ liệu 28 ngày cho top 5 vào thư mục outputs!")

# ==== Hàm phân tích bằng Groq ====
def analyze_data_with_groq(json_data):
    prompt = (
        "Bạn là chuyên gia phân tích kỹ thuật chứng khoán top 0,1%.\n"
        "Dưới đây là dữ liệu kỹ thuật của cổ phiếu (28 ngày gần nhất), "
        "hãy phân tích, nhận định xu hướng, điểm mua/bán, cảnh báo breakout, "
        "chỉ dùng đoạn văn bản, không dùng bảng.\n\n"
        f"Dữ liệu: {json_data}\n\n"
        "Phân tích chi tiết:"
    )
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_completion_tokens=3000
    )
    return completion.choices[0].message.content

# ==== Bước 3: Phân tích & gom báo cáo ====
json_files = glob.glob(os.path.join(save_path, "*.json"))
report_text = "Báo cáo phân tích kỹ thuật cho top 5 cổ phiếu VN30 tăng trưởng mạnh nhất:\n\n"

for file_path in json_files:
    with open(file_path, "r") as f:
        data_json = f.read()

    ticker_name = os.path.basename(file_path).replace(".json", "")
    print(f"🔍 Đang phân tích {ticker_name} ...")

    try:
        analysis_text = analyze_data_with_groq(data_json)
    except Exception as e:
        print(f"❌ Lỗi phân tích {ticker_name}: {e}")
        analysis_text = "Không có dữ liệu phân tích do lỗi API."

    report_text += f"--- Phân tích {ticker_name} ---\n{analysis_text}\n\n"
    os.remove(file_path)

# ==== Bước 4: Lưu báo cáo DOCX ====
doc = Document()
doc.add_heading("Báo cáo Phân tích Top 5 VN30", level=1)
for line in report_text.strip().split("\n"):
    doc.add_paragraph(line)

report_path = os.path.join(save_path, "Bao_cao_top5_VN30.docx")
doc.save(report_path)

print(f"✅ Đã lưu báo cáo vào {report_path}")

# ==== Bước 5: Gửi email ====
def send_email_report(receiver_email, subject, content, attachment_path):
    sender_email = os.getenv("EMAIL_USER")
    sender_password = os.getenv("EMAIL_PASS")
    if not sender_email or not sender_password:
        raise Exception("Chưa set EMAIL_USER và EMAIL_PASS")
    yag = yagmail.SMTP(user=sender_email, password=sender_password)
    yag.send(
        to=receiver_email,
        subject=subject,
        contents=content,
        attachments=attachment_path
    )
    print(f"📧 Đã gửi báo cáo tới {receiver_email}")

try:
    send_email_report(
        receiver_email="vanheminhtan@gmail.com",
        subject="Báo cáo phân tích Top 5 VN30",
        content=report_text,
        attachment_path=report_path
    )
except Exception as e:
    print(f"❌ Lỗi gửi email: {e}")
