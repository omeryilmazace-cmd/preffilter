from flask import Flask, render_template, jsonify, request
from bot import run_full_analysis, TICKER_FILE, scan_logs
import threading
import os

app = Flask(__name__)

# Global state
modes_data = {
    "preferred": {"all_data": [], "status": "idle"},
    "cef": {"all_data": [], "status": "idle"}
}
current_settings = {"threshold": 0.015}

def background_scan(mode):
    global modes_data
    modes_data[mode]["status"] = "scanning"
    try:
        res = run_full_analysis(threshold=current_settings["threshold"], mode=mode)
        if "error" in res:
            modes_data[mode]["status"] = f"Error: {res['error']}"
        else:
            # We only update all_data if no error
            modes_data[mode]["all_data"] = res.get("all_data", [])
            modes_data[mode]["status"] = "done"
    except Exception as e:
        modes_data[mode]["status"] = f"error: {str(e)}"

def scheduled_scan_loop():
    """Runs scan for both modes every 15 minutes."""
    while True:
        try:
            print("[SCHEDULER] Starting background scan...")
            scan_logs.clear()
            
            # 1. Preferred
            background_scan("preferred")
            
            # 2. CEFs
            background_scan("cef")
            
            print("[SCHEDULER] Scan complete. Sleeping for 15 mins.")
            time.sleep(900) # 15 minutes
        except Exception as e:
            print(f"[SCHEDULER] Error: {e}")
            time.sleep(60) # Retry in 1 min if error

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    return jsonify(modes_data)

@app.route('/api/logs')
def get_logs():
    return jsonify(scan_logs)

@app.route('/api/scan', methods=['POST'])
def start_scan():
    # If any is scanning, check logic
    data = request.json
    if data and "threshold" in data:
        current_settings["threshold"] = float(data["threshold"]) / 100

    scan_logs.clear()
    
    # Run both Preferred and CEF scans
    for m in ["preferred", "cef"]:
        if modes_data[m]["status"] != "scanning":
            threading.Thread(target=background_scan, args=(m,)).start()

    return jsonify({"message": "Scans started for both modes"})

@app.route('/api/tickers', methods=['GET'])
def get_tickers():
    if not os.path.exists(TICKER_FILE): return jsonify([])
    with open(TICKER_FILE, "r") as f:
        content = f.read()
        tickers = [t.strip() for t in content.replace('\n', ',').split(',') if t.strip()]
    return jsonify(sorted(list(set(tickers))))

@app.route('/api/tickers', methods=['POST'])
def add_ticker():
    ticker = request.json.get('ticker', '').strip().upper()
    if not ticker: return jsonify({"message": "Invalid"}), 400
    with open(TICKER_FILE, "a") as f: f.write(f", {ticker}")
    return jsonify({"message": "Success"})

@app.route('/api/tickers/<ticker>', methods=['DELETE'])
def remove_ticker(ticker):
    ticker = ticker.strip().upper()
    if not os.path.exists(TICKER_FILE): return jsonify({"message": "404"}), 404
    with open(TICKER_FILE, "r") as f:
        content = f.read()
        tickers = [t.strip() for t in content.replace('\n', ',').split(',') if t.strip()]
    if ticker in tickers:
        tickers.remove(ticker); 
        with open(TICKER_FILE, "w") as f: f.write(", ".join(tickers))
        return jsonify({"message": "OK"})
    return jsonify({"message": "404"}), 404

@app.route('/api/calc-stats', methods=['POST'])
def calc_stats():
    """
    On-demand calculation for historical index stats.
    Wraps bot.calculate_historical_index.
    """
    data = request.json
    target = data.get('target', '')
    peers = data.get('peers', [])
    
    if not target or not peers:
        return jsonify({"error": "Missing params"}), 400
        
    stats = bot.calculate_historical_index(target, peers)
    return jsonify(stats)

if __name__ == '__main__':
    # Start background scheduler
    t = threading.Thread(target=scheduled_scan_loop, daemon=True)
    t.start()
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
