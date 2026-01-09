import yfinance as yf
import json

def test_jpm():
    # Candidates for JPM-C
    candidates = ["JPM-PC", "JPM-C", "JPM.PC", "JPM.C", "JPM-PRC"]
    
    print(f"Testing candidates: {candidates}")
    
    # Batch download to see what works
    data = yf.download(candidates, period="5d", progress=False)
    
    print("\n--- Data Columns ---")
    print(data.columns)
    
    print("\n--- Close Prices ---")
    if 'Close' in data.columns:
        print(data['Close'].tail())
    else:
        print(data.tail())

    # Check individual info
    print("\n--- Ticker Info ---")
    for t in candidates:
        try:
            tick = yf.Ticker(t)
            hist = tick.history(period="5d")
            if not hist.empty:
                print(f"SUCCESS: {t} - Last Price: {hist['Close'].iloc[-1]}")
            else:
                print(f"FAILED: {t} - No history")
        except Exception as e:
            print(f"ERROR: {t} - {e}")

if __name__ == "__main__":
    test_jpm()
