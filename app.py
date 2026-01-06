from flask import Flask, render_template, jsonify, request
from bot import run_full_analysis, TICKER_FILE, scan_logs
import threading
import os

app = Flask(__name__)

# Global state
latest_results = {"all_data": [], "status": "idle"}
current_settings = {"threshold": 0.015}

def background_scan():
    global latest_results
    latest_results["status"] = "scanning"
    try:
        res = run_full_analysis(threshold=current_settings["threshold"])
        if "error" in res:
            latest_results["status"] = f"Error: {res['error']}"
        else:
            latest_results.update(res)
            latest_results["status"] = "done"
    except Exception as e:
        latest_results["status"] = f"error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    return jsonify(latest_results)

@app.route('/api/logs')
def get_logs():
    return jsonify(scan_logs)

@app.route('/api/scan', methods=['POST'])
def start_scan():
    if latest_results["status"] == "scanning":
        return jsonify({"message": "Scan already in progress"}), 400
    
    data = request.json
    if data and "threshold" in data:
        current_settings["threshold"] = float(data["threshold"]) / 100

    scan_logs.clear() # Reset logs for new scan
    thread = threading.Thread(target=background_scan)
    thread.start()
    return jsonify({"message": "Scan started"})

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

if __name__ == '__main__':
    app.run(debug=True, port=5000)
