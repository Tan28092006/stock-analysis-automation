import os
import pandas as pd
from datetime import datetime, timedelta
import json
import glob
from docx import Document
from groq import Groq
import yagmail
from vnstock import listing_companies, stock_historical_data

# =======================
# Cấu hình API & Email
# =======================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise Exception("Bạn chưa set GROQ_API_KEY trong môi trường")

client = Groq(api_key=GROQ_API_KEY)

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO = os.getenv("EMAIL_TO")

VN30_STATIC = [
    "ACB","BCM","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG",
    "KDH","MBB","MSN","MWG","NVL","PDR","PLX","POW","SAB","SHB",
    "SSB","SSI","STB","TCB","TPB","VCB","VHM","VIB","VIC","VJC","VNM"
]

# =======================
# Lấy danh sách VN30
# =======================
def get_vn30_tickers():
    try:
        df = listing_companies()
        print("Cột hiện có:", df.columns)

        if "group_code" in df.columns:
            lst = df[df["group_code"] == "VN30"]["ticker"].tolist()
            if lst:
                return lst

        print("Không thấy group_code → dùng danh sách tĩnh")
    except Exception as e:
        print(f"Lỗi listing_companies(): {e}")
    return VN30_STATIC

# =======================
# Top cổ phiếu tăng mạnh
# =======================
def get_top_gainers(days=14, top_n=5):
    tickers = get_vn30_tickers()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    changes = []
    for t in tickers:
        try:
            df = stock_historical_data(symbol=t,
                                       start_date=start_date.strftime("%Y-%m-%d"),
                                       end_date=end_date.strftime("%Y-%m-%d"))
            if df.empty: 
                continue
            pct = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
            changes.append((t, pct))
        except Exception as e:
            print(f"Lỗi {t}: {e}")
    df2 = pd.DataFrame(changes, columns=["Ticker", "Change"]).sort_values(by="Change", ascending=False)
    return df2.head(top_n)["Ticker"].tolist()

# =======================
# Phân tích kỹ thuật
# =======================
def analyze_stock(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    df = stock_historical_data(symbol=ticker,
                               start_date=start_date.strftime("%Y-%m-%d"),
                               end_date=end_date.strftime("%Y-%m-%d"))
    if df.empty:
        return {"ticker": ticker, "error": "No data"}
    
    df["MA20"] = df["close"].rolling(window=20).mean()
    df["MA50"] = df["close"].rolling(window=50).mean()
    
    ma_signal = "Bullish" if df["MA20"].iloc[-1] > df["MA50"].iloc[-1] else "Bearish"
    pct_change = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100

    return {
        "ticker": ticker,
        "pct_change": round(pct_change, 2),
        "MA20": round(df["MA20"].iloc[-1], 2),
        "MA50": round(df["MA50"].iloc[-1], 2),
        "signal": ma_signal
    }

# =======================
# Lưu kết quả JSON
# =======================
def save_to_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# =======================
# Gọi Groq AI tạo nhận định
# =======================
def generate_ai_summary(data):
    prompt = f"Phân tích và nhận định các cổ phiếu sau dựa trên dữ liệu:\n{json.dumps(data, ensure_ascii=False)}"
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content

# =======================
# Xuất DOCX
# =======================
def export_to_docx(summary, filename):
    doc = Document()
    doc.add_heading("Báo cáo phân tích cổ phiếu", level=1)
    doc.add_paragraph(summary)
    doc.save(filename)

# =======================
# Gửi email
# =======================
def send_email(subject, body, attachments=None):
    yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASS)
    yag.send(to=EMAIL_TO, subject=subject, contents=body, attachments=attachments)

# =======================
# Main workflow
# =======================
if __name__ == "__main__":
    top_tickers = get_top_gainers()
    analysis = [analyze_stock(t) for t in top_tickers]
    
    json_file = "analysis.json"
    save_to_json(analysis, json_file)
    
    summary = generate_ai_summary(analysis)
    docx_file = "report.docx"
    export_to_docx(summary, docx_file)
    
    send_email("Báo cáo phân tích cổ phiếu", summary, attachments=[json_file, docx_file])
    print("Báo cáo đã gửi thành công!")
