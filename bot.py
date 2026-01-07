import yfinance as yf
import pandas as pd
import time
import requests
import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, Days_Lookback, Alert_Threshold_Pct

# Ensure terminal output handles UTF-8
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except Exception: pass

TICKER_FILE = "tickers.txt"
METADATA_FILE = "metadata.json"
SYMBOL_CACHE_FILE = "symbol_cache.json"
MASTER_METADATA_FILE = "master_metadata.json"

# Global progress list for Web API
scan_logs = []

def log_msg(msg):
    ts = time.strftime("%H:%M:%S")
    full_msg = f"[{ts}] {msg}"
    print(full_msg)
    scan_logs.append(full_msg)
    if len(scan_logs) > 100: scan_logs.pop(0)

def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f: return json.load(f)
        except: return {}
    return {}

def save_json(path, data):
    with open(path, "w") as f: json.dump(data, f, indent=4)

def load_tickers():
    if not os.path.exists(TICKER_FILE): return {}
    with open(TICKER_FILE, "r") as f:
        content = f.read()
        raw_tickers = [t.strip().upper() for t in content.replace('\n', ',').split(',') if t.strip()]
    
    ticker_map = {}
    for t in sorted(list(set(raw_tickers))):
        possible = [t]
        if "-" in t:
            parts = t.split("-")
            if len(parts) == 2 and len(parts[1]) == 1:
                possible.append(f"{parts[0]}-P{parts[1]}")
                possible.append(f"{parts[0]}-PR{parts[1]}")
                possible.append(f"{parts[0]}.P{parts[1]}")
        ticker_map[t] = list(set(possible))
    return ticker_map

def infer_metadata(ticker_symbol, info):
    name = info.get("longName", "").upper()
    sector = info.get("sector", "Other")
    if any(x in name for x in ["BANK", "FINANCIAL", "CAPITAL", "TRUST"]): sector = "Financials/Banking"
    elif "REIT" in name or any(x in name for x in ["REAL ESTATE", "HOUSING"]): sector = "REITs"
    elif any(x in name for x in ["ENERGY", "POWER", "GAS", "ELECTRIC"]): sector = "Utilities/Energy"
    sec_type = "Preferred Stock"
    if any(x in name for x in ["NOTE", "SENIOR NOTE", "DEBENTURE", "ETD"]): sec_type = "Note/ETD"
    elif any(x in name for x in ["ETF", "ISHARES", "VANGUARD"]): sec_type = "ETF"
    rate_type = "Fixed"
    if any(x in name for x in ["FLOAT", "VARIABLE", "LIBOR", "SOFR"]): rate_type = "Floating"
    elif any(x in name for x in ["TO FLOAT", "TO VARIABLE", "RESET"]): rate_type = "Fixed-to-Float"
    return {"name": info.get("longName", ticker_symbol), "sector": sector, "type": sec_type, "rate": rate_type}

def run_full_analysis(threshold=None, mode="preferred"):
    global scan_logs
    if threshold is None: threshold = Alert_Threshold_Pct
    results = {"all_data": []}
    
    log_msg(f"--- Scan Started ({mode.upper()} mode) ---")
    
    if mode == "cef":
        cef_master = load_json("cef_masterlist.json")
        ticker_map = {t: [t] for t in cef_master.keys()}
        master_metadata = cef_master
    else:
        ticker_map = load_tickers()
        master_metadata = load_json(MASTER_METADATA_FILE)

    metadata_cache = load_json(METADATA_FILE)
    symbol_cache = load_json(SYMBOL_CACHE_FILE)
    
    if not ticker_map:
        log_msg("Error: No tickers found.")
        return {"error": "No tickers"}

    original_tickers = list(ticker_map.keys())
    log_msg(f"Targeting {len(original_tickers)} unique positions.")

    # Step 1: Resolve Symbols & Metadata
    resolved_map = {} # orig -> yahoo
    needs_resolution = []

    for orig in original_tickers:
        if orig in symbol_cache:
            resolved_map[orig] = symbol_cache[orig]
        else:
            needs_resolution.append(orig)

    if needs_resolution:
        log_msg(f"Resolving {len(needs_resolution)} new or uncached tickers...")
        m_step = 100
        chunks = [needs_resolution[i:i+m_step] for i in range(0, len(needs_resolution), m_step)]
        
        def resolve_chunk(chunk_orig):
            to_test = []
            for o in chunk_orig: to_test.extend(ticker_map[o])
            try:
                m_data = yf.Tickers(" ".join(to_test))
                for o in chunk_orig:
                    for v in ticker_map[o]:
                        try:
                            info = m_data.tickers[v].info
                            if info and info.get("longName"):
                                m_item = infer_metadata(v, info)
                                # Capture dividend info for CEFs or any stock
                                div_rate = info.get("dividendRate", info.get("trailingAnnualDividendRate", 0.0))
                                m_item["dividendRate"] = div_rate
                                metadata_cache[v] = m_item
                                symbol_cache[o] = v
                                resolved_map[o] = v
                                break
                        except: pass
            except: pass
            return

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(resolve_chunk, chunk) for chunk in chunks]
            for future in as_completed(futures):
                future.result()
        
        save_json(METADATA_FILE, metadata_cache)
        save_json(SYMBOL_CACHE_FILE, symbol_cache)
        log_msg("Resolution caches updated.")
    else:
        log_msg("All tickers resolved from cache. Skipping trials.")

    # Step 2: Price Fetching
    symbols_to_download = list(set(resolved_map.values()))
    log_msg(f"Fetching prices for {len(symbols_to_download)} active symbols...")
    
    chunk_size = 200
    all_prices = []
    chunks = [symbols_to_download[i:i+chunk_size] for i in range(0, len(symbols_to_download), chunk_size)]
    
    def fetch_chunk(chunk):
        try:
            return yf.download(chunk, period="4mo", progress=False, threads=True, auto_adjust=False)
        except Exception:
            return pd.DataFrame()

    log_msg(f"Starting parallel fetch in {len(chunks)} chunks...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_chunk = {executor.submit(fetch_chunk, c): c for c in chunks}
        for i, future in enumerate(as_completed(future_to_chunk)):
            data = future.result()
            if not data.empty:
                all_prices.append(data)
            pct = ((i + 1) / len(chunks)) * 100
            log_msg(f"Fetch Progress: {pct:.1f}%...")

    if not all_prices:
        log_msg("Critical: Price fetch failed.")
        return {"error": "No data"}

    log_msg("Crunching range analytics...")
    combined_prices = pd.concat(all_prices, axis=1)
    # v3.9.5: Use both Open and Close for streak detection
    closes = combined_prices['Close']
    opens = combined_prices['Open']

    for orig, v in resolved_map.items():
        if isinstance(closes, pd.DataFrame) and v in closes.columns:
            series = closes[v]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0] # Handle duplicates
            series = series.dropna()
            
            if len(series) < 5: continue
            current = float(series.iloc[-1])
            
            # Streak Detection
            streak_type = "Neutral"
            streak_count = 0
            if isinstance(opens, pd.DataFrame) and v in opens.columns:
                open_v = opens[v]
                if isinstance(open_v, pd.DataFrame): open_v = open_v.iloc[:, 0]
                close_v = closes[v]
                if isinstance(close_v, pd.DataFrame): close_v = close_v.iloc[:, 0]
                
                # Align on date index
                aligned = pd.concat([open_v, close_v], axis=1, keys=['O', 'C']).dropna()
                if not aligned.empty:
                    for i in range(len(aligned)-1, -1, -1):
                        row = aligned.iloc[i]
                        try:
                            c_val = float(row['C'])
                            o_val = float(row['O'])
                            g = c_val > o_val
                            r = c_val < o_val
                            
                            if i == len(aligned)-1:
                                if g: streak_type = "Green"
                                elif r: streak_type = "Red"
                                else: break
                                streak_count = 1
                            else:
                                if (streak_type == "Green" and g) or (streak_type == "Red" and r):
                                    streak_count += 1
                                else: break
                        except: break

            l60, h60 = float(series.tail(60).min()), float(series.tail(60).max())
            l30, h30 = float(series.tail(30).min()), float(series.tail(30).max())
            l7, h7 = float(series.tail(7).min()), float(series.tail(7).max())
            if l60 <= 0: continue
            
            # Master Metadata enrichment
            m_data = master_metadata.get(orig, {})
            coupon = m_data.get("coupon", 0.0)
            sector = m_data.get("sector", "Other")
            sp_rating = m_data.get("sp_rating", "NR")
            mid_rating = m_data.get("moody_rating", "NR")
            rate_type = m_data.get("rate", "Fix")
            asset_type = m_data.get("type", "Trad")
            call_date_str = m_data.get("call_date", "")
            
            # Determine if currently floating (Fix-Float with call date passed)
            is_currently_floating = False
            if rate_type == "Fix-Float" and call_date_str:
                try:
                    from datetime import datetime
                    call_date = datetime.strptime(call_date_str, "%m/%d/%Y")
                    if datetime.now() > call_date:
                        is_currently_floating = True
                except:
                    pass
            
            # Yield & Display
            if mode == "cef":
                m_info = metadata_cache.get(v, {})
                div_rate = m_info.get("dividendRate", 0.0)
                cur_yield = (div_rate / current) if current > 0 else 0.0
                display_coupon = f"${div_rate:.2f}"
            else:
                # Preferred logic: (Coupon * FaceValue) / CurrentPrice. Assuming $25 face value.
                cur_yield = (coupon * 25.0 / current) if current > 0 and coupon > 0 else 0.0
                display_coupon = f"{coupon*100:.2f}%" if coupon > 0 else "N/A"

            results["all_data"].append({
                "ticker": orig,
                "name": metadata_cache.get(v, {}).get("longName", orig),
                "sector": sector,
                "sp_rating": sp_rating,
                "moody_rating": mid_rating,
                "rate": rate_type,
                "type": asset_type,
                "is_floating": is_currently_floating,
                "call_date": call_date_str,
                "maturity": m_data.get("maturity", ""),
                "coupon": display_coupon,
                "price": current,
                "yield": f"{cur_yield*100:.2f}%" if cur_yield > 0 else "N/A",
                "raw_yield": float(cur_yield),
                "streak_type": streak_type,
                "streak_count": streak_count,
                "l60": l60, "h60": h60, "l30": l30, "h30": h30, "l7": l7, "h7": h7,
                "dL60": (current-l60)/l60, "dH60": (h60-current)/h60, 
                "dL30": (current-l30)/l30, "dH30": (h30-current)/h30, 
                "dL7": (current-l7)/l7, "dH7": (h7-current)/h7
            })

    log_msg(f"Scan Finished. {len(results['all_data'])} items analyzed.")
    return results
