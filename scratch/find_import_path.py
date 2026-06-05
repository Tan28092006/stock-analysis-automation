import sys
try:
    import stock_agent
    print("stock_agent path:", stock_agent.__file__)
except Exception as e:
    print("Error:", e)
