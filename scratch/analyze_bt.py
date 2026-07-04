import json
from collections import Counter

def analyze_trades():
    try:
        with open('bt_30.json', 'r') as f:
            data = json.load(f)
        
        trades = data.get('trades', [])
        if not trades:
            print("No trades found in the report.")
            return

        print(f"Total Trades: {len(trades)}")
        
        wins = [t for t in trades if t['net_return_pct'] > 0]
        losses = [t for t in trades if t['net_return_pct'] <= 0]
        
        print(f"Wins: {len(wins)} (Avg PnL: {sum(t['net_return_pct'] for t in wins)/len(wins) if wins else 0:.2f}%)")
        print(f"Losses: {len(losses)} (Avg PnL: {sum(t['net_return_pct'] for t in losses)/len(losses) if losses else 0:.2f}%)")
        
        exit_reasons = Counter(t['exit_reason'] for t in trades)
        print("\nExit Reasons:")
        for reason, count in exit_reasons.items():
            print(f"  {reason}: {count}")
            
        symbols = Counter(t['symbol'] for t in trades)
        print("\nSymbols traded:")
        for symbol, count in symbols.items():
            print(f"  {symbol}: {count}")
            
        print("\nFirst 10 Trades:")
        for t in trades[:10]:
            print(f"  {t['symbol']} | Entry: {t['entry_date']} | Exit: {t['exit_date']} | PnL: {t['net_return_pct']:.2f}% | Reason: {t['exit_reason']}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    analyze_trades()
