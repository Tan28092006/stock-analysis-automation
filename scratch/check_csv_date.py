import pandas as pd
from pathlib import Path

p = Path("d:/Chungkhoan/data/raw/prices/ACB.csv")
if p.exists():
    df = pd.read_csv(p)
    print("ACB.csv last 5 rows:")
    print(df.tail(5))
else:
    print("ACB.csv not found")
