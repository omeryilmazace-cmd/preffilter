import json
import os

def parse_line(line):
    # Exclusion list for inactive stocks (based on price column)
    EXCLUDE_STATUSES = ['BK', 'CONVERT', 'CONVERTED', 'DELIST', 'EXCHANGE', 'REDEEM', 'REDEEMED', 'CALLED']
    
    # Manual exclusion list for specific tickers
    EXCLUDE_TICKERS = [
        'GMLPF', 'RILYP', 'TFINP', 'AUVIP', 'CSSEP', 'CSSEN', 'SCCC', 'AJXA', 'CCLDP', 'OPINL',
        'AIC', 'AAIN', 'HMLP-A', 'LUXHP', 'FOSLL', 'CNFRL', 'AL-A', 'AIG-A', 'ANG-A', 'ANG-B',
        'ARGD', 'ARGO-A', 'AHL-C', 'TBC', 'ATH-C', 'ATCO-D', 'ATCOL', 'RILYO', 'RILYM', 'BWSN',
        'BC-B', 'BC-A', 'CSWCZ', 'CCIA', 'CSR-C', 'C-J', 'C-K', 'CFG-D', 'CMRE-E', 'COWNL',
        'CUBI-E', 'CUBI-F', 'DLNG-B', 'EICB', 'EFC-E', 'ET-D', 'ET-C', 'ET-E', 'EQC-D', 'AGM-C',
        'FHN-D', 'FHN-B', 'FNB-E', 'FTAIP', 'FTAIO', 'GLOG-A', 'GLADZ', 'GAINL', 'GLP-A', 'GS-K',
        'GS-J', 'GECCM', 'GECCN', 'GECCZ', 'HWCPL', 'HROWL', 'HROWM', 'HT-C', 'HT-D', 'HT-E',
        'IVR-B', 'MDRRP', 'MBNKP', 'MBINP', 'MBINO', 'NEWTL', 'NI-B', 'NS-C', 'NS-A', 'NS-B',
        'NSS', 'OCFCP', 'OFSSI', 'OXLCM', 'OXSQL', 'OXSQZ', 'PRIF-F', 'PRIF-G', 'PRIF-H', 'PRIF-I',
        'PXSAP', 'METCL', 'O-', 'RF-B', 'SCCB', 'SACC', 'SCE-H', 'SITC-A', 'STT-D', 'SPLPP',
        'SNCRL', 'TELZ', 'TGH-B', 'TGH-A', 'TRINL', 'UMBFP', 'UCB-I', 'WFC-R', 'WFC-Q', 'WSBCP',
        'WCC-A', 'WTFCM', 'WTFCP', 'XFLT-A', 'ZIONL', 'ZIONO', 'RMPL-', 'SBBA', 'ASBA', 'ESGOF',
        'ESGRF', 'MFICL'
    ]
    
    parts = line.split('\t')
    if len(parts) < 15:
        # Try splitting by multiple spaces if tabs aren't present
        import re
        parts = re.split(r'\t| {2,}', line.strip())
    
    if len(parts) < 15:
        return None
    
    # Check ticker against manual exclusion list
    ticker = parts[6].strip().upper()
    if ticker in EXCLUDE_TICKERS:
        return None
    
    # Check if price column (index 8) contains an exclusion status
    price_col = parts[8].strip().upper()
    for status in EXCLUDE_STATUSES:
        if status in price_col:
            return None  # Skip this stock

    
    ticker = parts[6].strip().upper()
    coupon_str = parts[7].strip().replace('%', '')
    sector = parts[3].strip()
    rate_type = parts[4].strip()
    asset_type = parts[5].strip()
    sp_rating = parts[13].strip()
    moody_rating = parts[14].strip()
    
    # Call date at index 16, maturity at index 17
    call_date = ""
    maturity = ""
    if len(parts) > 16:
        call_date = parts[16].strip()
    if len(parts) > 17:
        maturity = parts[17].strip()
    
    try:
        coupon = float(coupon_str) / 100
    except:
        coupon = 0.0
        
    return {
        "ticker": ticker,
        "coupon": coupon,
        "sector": sector,
        "rate": rate_type,
        "type": asset_type,
        "call_date": call_date,
        "maturity": maturity,
        "sp_rating": sp_rating,
        "moody_rating": moody_rating
    }

def main():
    metadata = {}
    if not os.path.exists('masterlist_raw.txt'):
        print("Raw masterlist not found.")
        return
        
    with open('masterlist_raw.txt', 'r', encoding='utf-8') as f:
        for line in f:
            res = parse_line(line)
            if res:
                metadata[res['ticker']] = res
    
    with open('master_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved {len(metadata)} entries to master_metadata.json")

if __name__ == "__main__":
    main()
