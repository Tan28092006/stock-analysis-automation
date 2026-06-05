import yfinance as yf
from datetime import date, timedelta
from vnstock import Vnstock
import io
from contextlib import redirect_stdout

end = date.today()
start = end - timedelta(days=5)
print("Querying yfinance for ACB.VN...")
try:
    ticker = yf.Ticker("ACB.VN")
    df_yf = ticker.history(start=str(start), end=str(end + timedelta(days=1)))
    print("yfinance returned:")
    print(df_yf)
except Exception as e:
    print("yfinance error:", e)

print("\nQuerying vnstock for ACB...")
try:
    with redirect_stdout(io.StringIO()):
        stock = Vnstock().stock(symbol="ACB", source="VCI")
    df_vn = stock.quote.history(start=str(start), end=str(end), interval="1D")
    print("vnstock returned:")
    print(df_vn)
except Exception as e:
    print("vnstock error:", e)
