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

    # Step 2: Price Fetching
    SECTOR_ETFS = {
        "Municipal Bond": ["XMPT", "MUB"],
        "Equity/Core": ["QQQ", "SPY"],
        "Utility": ["XLU"],
        "Utilities": ["XLU"],
        "Real Estate": ["XLRE", "VNQ"],
        "Mixed/Debt": ["SPY"],
        "Other": ["SPY"]
    }
    
    benchmark_symbols = []
    if mode == "cef":
        for etf_list in SECTOR_ETFS.values():
            benchmark_symbols.extend(etf_list)
    
    symbols_to_download = list(set(resolved_map.values()) | set(benchmark_symbols))
    log_msg(f"Benchmarks: {benchmark_symbols}")
    log_msg(f"Download Sample: {symbols_to_download[:10]}")
    log_msg(f"Fetching prices for {len(symbols_to_download)} symbols (Mode: {mode})...")
    
    chunk_size = 200
    all_prices = []
    chunks = [symbols_to_download[i:i+chunk_size] for i in range(0, len(symbols_to_download), chunk_size)]
    
    # CEFs need adjusted Close (IDV adjusted)
    adj_flag = True if mode == "cef" else False

    def fetch_chunk(chunk):
        try:
            return yf.download(chunk, period="4mo", progress=False, threads=True, auto_adjust=adj_flag)
        except Exception:
            return pd.DataFrame()

    log_msg(f"Parallel fetch (Auto-Adjust={adj_flag}) in {len(chunks)} chunks...")
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
    # Debug info
    # log_msg(f"Combined Cols: {list(combined_prices.columns)[:5]}")
    try:
        closes = combined_prices['Close']
        opens = combined_prices['Open']
        volumes = combined_prices['Volume']
    except Exception as e:
        log_msg(f"Error accessing Close/Open: {e}")
        # Fallback if flat index (rare but possible with 1 ticker)
        closes = combined_prices
        opens = combined_prices 
        volumes = pd.DataFrame() # Fallback

    # Ensure DataFrame format
    if isinstance(closes, pd.Series): closes = closes.to_frame()
    if isinstance(opens, pd.Series): opens = opens.to_frame()
    if isinstance(volumes, pd.Series): volumes = volumes.to_frame()

    # --- FALLBACK FOR MISSING BENCHMARKS ---
    if mode == "cef" and benchmark_symbols:
        available = []
        if hasattr(closes, 'columns'):
            available = [str(c).upper() for c in closes.columns]
        
        missing_benchmarks = [b for b in benchmark_symbols if b not in available]
        
        if missing_benchmarks:
            log_msg(f"Attempting fallback fetch for missing benchmarks: {missing_benchmarks}")
            for mb in missing_benchmarks:
                try:
                    # Individual fetch
                    mb_data = yf.download(mb, period="4mo", progress=False, threads=False, auto_adjust=adj_flag)
                    if not mb_data.empty and 'Close' in mb_data.columns:
                        mb_close = mb_data['Close']
                        # Handle if it comes back as DataFrame or Series
                        if isinstance(mb_close, pd.DataFrame):
                             # Often comes as (Date, Ticker) -> we want just the series
                             # If column name matches ticker, grab it, else take first col
                             if mb in mb_close.columns:
                                 mb_close = mb_close[mb]
                             else:
                                 mb_close = mb_close.iloc[:, 0]
                        
                        # Rename Series to ticker name to ensure it aligns
                        mb_close.name = mb
                        
                        # Merge into closes
                        # We use join (outer) to keep existing dates, or concat
                        # Concat is safer to append a new column
                        closes = pd.concat([closes, mb_close], axis=1)
                        log_msg(f"Fallback success for {mb}")
                except Exception as e:
                    log_msg(f"Fallback failed for {mb}: {e}")
    # ---------------------------------------
    
    if hasattr(closes, 'columns'):
         log_msg(f"Closes columns (first 10): {list(closes.columns)[:10]}")

    # Pre-calculate benchmark dips if in CEF mode
    benchmarks_dL60 = {}
    if mode == "cef":
        log_msg(f"Calculating benchmarks for {len(benchmark_symbols)} ETFs...")
        # Clean columns for easier matching
        available_cols = []
        if isinstance(closes, pd.DataFrame):
            available_cols = [str(c).upper() for c in closes.columns]
        
        log_msg(f"Available Cols for Benchmarks: {available_cols[:20]}... (Total: {len(available_cols)})")

        for sym in benchmark_symbols:
            s_upper = sym.upper()
            if s_upper in available_cols:
                # Get the actual column (could be string or MultiIndex)
                col_idx = closes.columns[available_cols.index(s_upper)]
                s = closes[col_idx]
                if isinstance(s, pd.DataFrame): 
                    s = s.iloc[:, 0]
                s = s.dropna()
                if not s.empty:
                    current = float(s.iloc[-1])
                    h60 = float(s.tail(60).max())
                    if h60 > 0:
                        benchmarks_dL60[sym] = (h60 - current) / h60
                        # log_msg(f"Benchmark {sym}: {benchmarks_dL60[sym]*100:.2f}% dip")
        
        log_msg(f"Benchmarks ready: {list(benchmarks_dL60.keys())}")

    analysis_count = 0
    for orig, v in resolved_map.items():
        if v in closes.columns:
            analysis_count += 1
            series = closes[v]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0] # Handle duplicates
            series = series.dropna()
            
            if len(series) < 15: continue # Need enough data for RSI
            current = float(series.iloc[-1])
            
            # RSI Calculation
            try:
                rsi_s = calculate_rsi(series)
                rsi_sq = rsi_s.dropna()
                current_rsi = float(rsi_sq.iloc[-1]) if not rsi_sq.empty else 50.0
            except:
                current_rsi = 50.0

            # Volume Calculation
            try:
                vol_s = volumes[v]
                if isinstance(vol_s, pd.DataFrame): vol_s = vol_s.iloc[:, 0]
                vol_s = vol_s.dropna()
                avg_vol = float(vol_s.tail(10).mean()) if not vol_s.empty else 0.0
            except:
                avg_vol = 0.0
            
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
            sector = m_data.get("sector", "Other").strip()
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
            raw_divergence = 0.0
            display_divergence = "-"
            
            if mode == "cef":
                m_info = metadata_cache.get(v, {})
                div_rate = m_info.get("dividendRate", 0.0)
                cur_yield = (div_rate / current) if current > 0 else 0.0
                display_coupon = f"${div_rate:.2f}"
                
                # Divergence calculation
                relevant_etfs = SECTOR_ETFS.get(sector, [])
                used_etfs = [e for e in relevant_etfs if e in benchmarks_dL60]
                etf_dips = [benchmarks_dL60[e] for e in used_etfs]
                
                benchmark_str = ""
                if used_etfs:
                    benchmark_str = f"vs {', '.join(used_etfs)}"
                
                if etf_dips:
                    avg_etf_dip = sum(etf_dips) / len(etf_dips)
                    cef_dip = (h60 - current) / h60 # How much it dipped from 60D high
                    raw_divergence = (cef_dip - avg_etf_dip)
                    display_divergence = f"{raw_divergence*100:+.1f}%"
                else:
                    log_msg(f"Warning: No benchmarks for {v} (Sector: '{sector}'). Expected: {relevant_etfs}, Found: {list(benchmarks_dL60.keys())}")
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
                "divergence": display_divergence,
                "raw_divergence": raw_divergence,
                "raw_yield": float(cur_yield),
                "raw_coupon": coupon, # Add raw numeric coupon for frontend index calc
                "streak_type": streak_type,
                "streak_count": streak_count,
                "l60": l60, "h60": h60, "l30": l30, "h30": h30, "l7": l7, "h7": h7,
                "dL60": (current-l60)/l60, "dH60": (h60-current)/h60, 
                "dL30": (current-l30)/l30, "dH30": (h30-current)/h30, 
                "dL7": (current-l7)/l7, "dH7": (h7-current)/h7,
                "rsi": round(current_rsi, 1),
                "avg_volume": int(avg_vol),
                "benchmark_str": benchmark_str if mode == "cef" else ""
            })

    log_msg(f"Scan Complete. Found {len(results['all_data'])} items.")
    return results

def calculate_historical_index(target_ticker, peer_tickers):
    """
    Fetches 90-day history for target and peers, computes daily index ratio,
    and returns stats (L7/H7, L30/H30, etc. for the INDEX VALUE).
    """
    if not target_ticker or not peer_tickers:
        return {"error": "Missing target or peers"}
    
    # 1. Resolve Yahoo Tickers
    # The frontend sends "JPM-L" or "C-J".
    # Need to convert to Yahoo format: "JPM-PL", "C-PJ".
    # Since I don't have the `load_tickers` map here readily without re-loading,
    # I'll rely on a simple heuristic or pass the Yahoo tickers from frontend?
    # Actually, `bot.py` has `load_tickers` but `run_full_analysis` builds the map.
    # Simple heuristic: If it has hyphen, try inserting 'P'. 
    # BUT, frontend already has mapped tickers in `rawData`? 
    # Let's assume frontend sends RAW tickers (e.g. JPM-L) and we convert here or frontend sends Yahoo tickers.
    # Better: Frontend sends what it has.
    # In `run_full_analysis`, I use `orig` (JPM-L) and map to `v` (JPM-PL).
    # I'll reuse `load_tickers()` logic if needed, but let's try a direct map first.
    
    def to_yahoo(t):
        if "-" in t:
            parts = t.split("-")
            # Try -P first
            return f"{parts[0]}-P{parts[1]}"
        return t

    target_y = to_yahoo(target_ticker)
    peers_y = [to_yahoo(p) for p in peer_tickers]
    all_tickers = [target_y] + peers_y

    try:
        # 2. Fetch History (Batch)
        # 3mo = ~65 trading days. 6mo might be safer for 90d lookback? Index usually needs 90d?
        # User asked for "Last 90 day low". Let's fetch "6mo" to be safe.
        data = yf.download(all_tickers, period="6mo", progress=False)['Close']
        
        if data.empty:
             return {"error": "No data found"}
        
        # 3. Process
        # Ensure target column exists
        if target_y not in data.columns:
            # Try alternate formatting? JPM-PL vs JPM-PRL?
            # Creating a robust fallback is hard without the cache.
            # Let's try basic fallback
            alt = target_ticker.replace("-", "-PR")
            if alt in data.columns: target_y = alt
            else: return {"error": f"Target {target_ticker} not found in Yahoo"}

        # Calculate Peer Average Daily
        # Filter peers that exist in columns
        valid_peers = [p for p in peers_y if p in data.columns]
        if not valid_peers:
            return {"error": "No valid peer data found"}

        peer_avg = data[valid_peers].mean(axis=1)
        target_price = data[target_y]
        
        # Index Ratio Series
        index_series = target_price / peer_avg
        index_series = index_series.dropna()
        
        if index_series.empty:
            return {"error": "Not enough overlapping data"}

        # 4. Compute Stats
        current_val = index_series.iloc[-1]
        
        def get_stats(days):
            # Last 'days' trading days. (Approx days * 5/7?). 
            # Or just take last N rows? 
            # User said "90 day low". Usually implies calendar days? 
            # In finance, usually "Last 30 bars". 
            # Let's assume trading days for simplicity or limit by timestamp.
            # window = ~days * 0.7 for trading days? 
            # Let's just use slicing by index for now: last N rows.
            # 90 calendar days ~= 63 trading days.
            n = int(days * 0.7) 
            subset = index_series.tail(n)
            if subset.empty: return None, None
            return round(subset.min(), 3), round(subset.max(), 3)

        l7, h7 = get_stats(10) # ~ 7 calendar days
        l30, h30 = get_stats(30)
        l60, h60 = get_stats(60)
        l90, h90 = get_stats(90)

        return {
            "current": round(current_val, 3),
            "l7": l7, "h7": h7,
            "l30": l30, "h30": h30,
            "l60": l60, "h60": h60,
            "l90": l90, "h90": h90
        }

    except Exception as e:
        return {"error": str(e)}
