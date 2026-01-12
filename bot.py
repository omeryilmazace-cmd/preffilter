import yfinance as yf
import pandas as pd
import time
import requests
import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, Days_Lookback, Alert_Threshold_Pct

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

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
        master_metadata = load_json(MASTER_METADATA_FILE)
        # Scan ALL tickers in master metadata, not just the watchlist
        raw_tickers = list(master_metadata.keys())
        ticker_map = {}
        for t in raw_tickers:
            t = t.strip().upper()
            possible = [t]
            if "-" in t:
                parts = t.split("-")
                if len(parts) == 2 and len(parts[1]) == 1:
                    possible.append(f"{parts[0]}-P{parts[1]}")
                    possible.append(f"{parts[0]}-PR{parts[1]}")
                    possible.append(f"{parts[0]}.P{parts[1]}")
            ticker_map[t] = list(set(possible))

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
            v = symbol_cache[orig]
            # Check if we have valid metadata. If incomplete, force retry.
            if metadata_cache.get(v, {}).get("incomplete", False):
                needs_resolution.append(orig)
            else:
                resolved_map[orig] = v
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
                            else:
                                raise Exception("Empty info")
                        except:
                            # Fallback: trust the ticker exists, use defaults
                            # Only do this if we haven't found a better match yet
                            if o not in resolved_map:
                                # Use incomplete flag so we retry next time
                                metadata_cache[v] = {"name": v, "sector": "Other", "type": "Unknown", "rate": "Fixed", "dividendRate": 0.0, "incomplete": True}
                                symbol_cache[o] = v
                                resolved_map[o] = v
                                break
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

    # Step 2: Price Fetching & Analysis (Chunk-based for memory efficiency)
    import gc
    
    SECTOR_ETFS = {
        "Municipal Bond": ["XMPT", "MUB"],
        "Equity/Core": ["QQQ", "SPY"],
        "Utility": ["XLU"],
        "Utilities": ["XLU"],
        "Real Estate": ["XLRE", "VNQ"],
        "Mixed/Debt": ["SPY"],
        "Other": ["SPY"]
    }
    
    symbols_to_download = list(set(resolved_map.values()))
    # Add benchmark symbols for CEF mode
    benchmark_symbols = []
    if mode == "cef":
        for etf_list in SECTOR_ETFS.values():
            benchmark_symbols.extend(etf_list)
        benchmark_symbols = list(set(benchmark_symbols))
        symbols_to_download = list(set(symbols_to_download) | set(benchmark_symbols))

    log_msg(f"Fetching data for {len(symbols_to_download)} symbols in chunks...")
    
    chunk_size = 75
    chunks = [symbols_to_download[i:i+chunk_size] for i in range(0, len(symbols_to_download), chunk_size)]
    adj_flag = True if mode == "cef" else False
    
    benchmarks_data = {} # Only used in CEF mode

    def process_chunk_data(chunk_df, chunk_orig_map):
        chunk_results = []
        if chunk_df.empty: return []
        
        # Downcast to float32 immediately to save memory
        chunk_df = chunk_df.astype('float32')
        
        try:
            closes = chunk_df['Close']
            opens = chunk_df['Open']
            volumes = chunk_df['Volume']
        except:
            closes = chunk_df
            opens = chunk_df
            volumes = pd.DataFrame()

        if isinstance(closes, pd.Series): closes = closes.to_frame()
        if isinstance(opens, pd.Series): opens = opens.to_frame()
        if isinstance(volumes, pd.Series): volumes = volumes.to_frame()

        # Extract bench data if needed
        if mode == "cef":
            for b in benchmark_symbols:
                if b in closes.columns:
                    s = closes[b].dropna()
                    if not s.empty:
                        curr = float(s.iloc[-1])
                        h60 = float(s.tail(60).max())
                        if h60 > 0: benchmarks_data[b] = (h60 - curr) / h60

        # Run analysis for each ticker in this chunk
        for orig, v in chunk_orig_map.items():
            if v not in closes.columns: continue
            
            series = closes[v].dropna()
            if len(series) < 15: continue
            
            current = float(series.iloc[-1])
            l60, h60 = float(series.tail(60).min()), float(series.tail(60).max())
            l30, h30 = float(series.tail(30).min()), float(series.tail(30).max())
            l7, h7 = float(series.tail(7).min()), float(series.tail(7).max())
            if l60 <= 0: continue

            # RSI
            try:
                rsi_s = calculate_rsi(series).dropna()
                current_rsi = float(rsi_s.iloc[-1]) if not rsi_s.empty else 50.0
            except: current_rsi = 50.0

            # Vol
            try:
                vol_s = volumes[v].dropna()
                avg_vol = float(vol_s.tail(10).mean()) if not vol_s.empty else 0.0
            except: avg_vol = 0.0

            # Streak
            streak_type, streak_count = "Neutral", 0
            if v in opens.columns:
                o_v = opens[v]
                c_v = closes[v]
                aligned = pd.concat([o_v, c_v], axis=1, keys=['O', 'C']).dropna()
                if not aligned.empty:
                    for i in range(len(aligned)-1, -1, -1):
                        row = aligned.iloc[i]
                        c_val, o_val = float(row['C']), float(row['O'])
                        g, r = c_val > o_val, c_val < o_val
                        if i == len(aligned)-1:
                            if g: streak_type = "Green"
                            elif r: streak_type = "Red"
                            else: break
                            streak_count = 1
                        else:
                            if (streak_type == "Green" and g) or (streak_type == "Red" and r): streak_count += 1
                            else: break

            m_data = master_metadata.get(orig, {})
            raw_coupon = m_data.get("raw_coupon", 0.0)
            sector = m_data.get("sector", "Other").strip()
            
            # Simplified result item to save memory
            res_item = {
                "ticker": orig,
                "name": m_data.get("name", metadata_cache.get(v, {}).get("longName", orig)),
                "sector": sector,
                "sp_rating": m_data.get("sp_rating", "NR"),
                "moody_rating": m_data.get("moody_rating", "NR"),
                "rate": m_data.get("rate", "Fix"),
                "type": m_data.get("type", "Trad"),
                "coupon": m_data.get("coupon", "N/A"),
                "price": round(current, 2),
                "yield": f"{(raw_coupon * 25.0 / current)*100:.2f}%" if current > 0 and raw_coupon > 0 and mode != "cef" else "N/A",
                "raw_yield": float(raw_coupon * 25.0 / current) if current > 0 and raw_coupon > 0 else 0.0,
                "raw_coupon": raw_coupon,
                "streak_type": streak_type,
                "streak_count": streak_count,
                "dL60": (current-l60)/l60, "dH60": (h60-current)/h60, 
                "dL30": (current-l30)/l30, "dH30": (h30-current)/h30, 
                "dL7": (current-l7)/l7, "dH7": (h7-current)/h7,
                "rsi": round(current_rsi, 1),
                "avg_volume": int(avg_vol)
            }
            if mode == "cef":
                # Special CEF fields
                m_info = metadata_cache.get(v, {})
                div_rate = m_info.get("dividendRate", 0.0)
                res_item["yield"] = f"{(div_rate/current)*100:.2f}%" if current > 0 else "N/A"
                res_item["raw_yield"] = float(div_rate/current) if current > 0 else 0.0
                res_item["coupon"] = f"${div_rate:.2f}"
                # Divergence will be calculated in post-processing
                res_item["_cef_h60"] = h60
                res_item["_cef_current"] = current
            
            chunk_results.append(res_item)
            
        return chunk_results

    all_data_flattened = []
    for i, chunk in enumerate(chunks):
        log_msg(f"Processing chunk {i+1}/{len(chunks)}...")
        chunk_df = yf.download(chunk, period="4mo", progress=False, threads=True, auto_adjust=adj_flag)
        
        # Create a reverse map for this chunk
        chunk_orig_map = {orig: v for orig, v in resolved_map.items() if v in chunk}
        
        chunk_results = process_chunk_data(chunk_df, chunk_orig_map)
        all_data_flattened.extend(chunk_results)
        
        # Immediate memory cleanup
        del chunk_df
        gc.collect()

    # Step 3: Post-processing (CEF Divergence)
    if mode == "cef" and benchmarks_data:
        log_msg("Calculating CEF Divergences...")
        for item in all_data_flattened:
            relevant_etfs = SECTOR_ETFS.get(item["sector"], [])
            used_etfs = [e for e in relevant_etfs if e in benchmarks_data]
            if used_etfs:
                avg_etf_dip = sum(benchmarks_data[e] for e in used_etfs) / len(used_etfs)
                cef_dip = (item["_cef_h60"] - item["_cef_current"]) / item["_cef_h60"]
                raw_div = (cef_dip - avg_etf_dip)
                item["divergence"] = f"{raw_div*100:+.1f}%"
                item["raw_divergence"] = raw_div
                item["benchmark_str"] = f"vs {', '.join(used_etfs)}"
            else:
                item["divergence"] = "-"
                item["raw_divergence"] = 0
            
            # Clean up temporary CEF helper fields
            item.pop("_cef_h60", None)
            item.pop("_cef_current", None)

    results["all_data"] = all_data_flattened
    log_msg(f"Scan Complete. Found {len(results['all_data'])} items.")
    return results

def calculate_historical_index(target_ticker, peer_tickers):
    """
    Fetches 90-day history for target and peers, computes daily index ratio,
    and returns stats (L7/H7, L30/H30, etc. for the INDEX VALUE).
    """
    if not target_ticker or not peer_tickers:
        return {"error": "Missing target or peers"}
    
    symbol_cache = load_json(SYMBOL_CACHE_FILE)
    
    def resolve_t(t):
        if t in symbol_cache:
            return symbol_cache[t]
        # Fallback heuristic
        if "-" in t:
            parts = t.split("-")
            return f"{parts[0]}-P{parts[1]}"
        return t

    target_y = resolve_t(target_ticker)
    peers_y = [resolve_t(p) for p in peer_tickers]
    all_tickers = list(set([target_y] + peers_y))

    try:
        # Fetch 2 years of data for extensive historical context
        df = yf.download(all_tickers, period="2y", progress=False, threads=True, group_by='ticker')
        
        if df.empty:
             return {"error": "No data found for these symbols"}
        
        # Extract Close prices
        closes = pd.DataFrame()
        for t in all_tickers:
            try:
                if t in df.columns.levels[0]:
                    s = df[t]['Close'].dropna()
                    if not s.empty:
                        closes[t] = s
            except:
                if 'Close' in df.columns:
                    closes = df['Close']
                    break

        if target_y not in closes.columns:
            return {"error": f"Target {target_ticker} ({target_y}) data missing"}

        valid_peers = [p for p in peers_y if p in closes.columns]
        if not valid_peers:
            return {"error": "No valid peer data found for index calculation"}

        # Calculate Index: Target / Average(Peers)
        peer_avg = closes[valid_peers].mean(axis=1)
        target_price = closes[target_y]
        
        index_series = (target_price / peer_avg).dropna()
        
        if len(index_series) < 5:
            return {"error": "Not enough overlapping price data"}

        current_val = float(index_series.iloc[-1])
        
        # stats calculation (keep existing windows)
        def get_stats(days):
            n = int(days * 0.72) 
            subset = index_series.tail(n)
            if subset.empty: return None, None
            return round(float(subset.min()), 3), round(float(subset.max()), 3)

        l7, h7 = get_stats(10)
        l30, h30 = get_stats(30)
        l60, h60 = get_stats(60)
        l90, h90 = get_stats(90)

        # Prepare series for Chart.js
        # We'll send labels (dates) and values
        labels = [d.strftime('%Y-%m-%d') for d in index_series.index]
        values = [round(float(v), 3) for v in index_series.values]

        return {
            "current": round(current_val, 3),
            "l7": l7, "h7": h7,
            "l30": l30, "h30": h30,
            "l60": l60, "h60": h60,
            "l90": l90, "h90": h90,
            "series": {
                "labels": labels,
                "values": values
            }
        }

    except Exception as e:
        return {"error": f"Calculation error: {str(e)}"}
