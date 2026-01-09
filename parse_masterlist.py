import pandas as pd
import json
import re

# Read the CSV file (skip blank rows, handle irregular header)
df = pd.read_csv("masterlist.csv", skiprows=1, encoding='utf-8')

# Fix column names (split if necessary)
# From viewing the file:
# Columns: [None], Issuer, Cum, QDI, Sector, Fix/Float, Type, Ticker, Coupon  Percent, Current Price, Change, Current Yield, Ann Yield to Worst , IG, S&P Rating, Moody Rating, Quarterly Int / Div, 1st Call, Maturity Date, Pay Dates, [blank], All

# Rename Columns Explicitly for clarity
df.columns = [
    "_blank_", "Issuer", "Cum", "QDI", "Sector", "Fix_Float", "Type", "Ticker", "Coupon", 
    "Price", "Change", "Current_Yield", "Yield_to_Worst", "IG", "SP_Rating", "Moody_Rating",
    "Quarterly_Div", "Call_Date", "Maturity", "Pay_Dates", "_blank2_", "_All_"
]

# The Ticker column is what we use as the key
# We need to normalize tickers:
# - The CSV often has tickers like "JPM-C" which is what we want.
# - Some may have .PX or -PX format, but we standardize to Original format

output = {}
for idx, row in df.iterrows():
    ticker = row.get("Ticker")
    if pd.isna(ticker) or ticker is None or str(ticker).strip() == "":
        continue
    ticker = str(ticker).strip().upper()
    
    # Clean it (e.g. "JPM-C" stays)
    # Yahoo uses JPM-PC but our original ticker is JPM-C, so we store as JPM-C
    
    issuer = str(row.get("Issuer", "")).strip()
    
    # Coupon: "9.875%" -> raw value 0.09875
    coupon_str = str(row.get("Coupon", "0%")).replace(",", "").strip()
    coupon_match = re.search(r'([\d.]+)', coupon_str)
    raw_coupon = float(coupon_match.group(1)) / 100 if coupon_match else 0.0  # e.g. 9.875% -> 0.09875
    
    # Yield is already formatted like "9.52%"
    yield_str = str(row.get("Current_Yield", "0%")).strip()
    if "%" not in yield_str:
        yield_str = yield_str + "%"
    
    # Price
    price_str = str(row.get("Price", "$0")).replace("$", "").replace(",", "").strip()
    try:
        price = float(price_str)
    except:
        price = 0.0
    
    # Call Date (e.g. "2/15/2027")
    call_date = str(row.get("Call_Date", "")).strip()
    if call_date.lower() == "nan" or call_date == "": call_date = "NONE"
    
    # Maturity
    maturity = str(row.get("Maturity", "")).strip()
    if maturity.lower() == "nan" or maturity == "" or "NONE" in maturity.upper(): maturity = "NONE"
    
    # Ratings
    sp_rating = str(row.get("SP_Rating", "NR")).strip()
    if sp_rating.lower() == "nan" or sp_rating == "": sp_rating = "NR"
    
    moody_rating = str(row.get("Moody_Rating", "NR")).strip()
    if moody_rating.lower() == "nan" or moody_rating == "": moody_rating = "NR"
    
    # Sector
    sector = str(row.get("Sector", "Other")).strip()
    if sector.lower() == "nan": sector = "Other"
    
    # Type (Trad, BB, Trust P)
    sec_type = str(row.get("Type", "Preferred")).strip()
    if sec_type.lower() == "nan": sec_type = "Preferred"
    
    # Rate Type (Fix, Float, Fix-Float)
    rate = str(row.get("Fix_Float", "Fix")).strip()
    if rate.lower() == "nan": rate = "Fix"
    
    # IG
    ig = str(row.get("IG", "N")).strip().upper()
    is_ig = ig == "Y"
    
    # Cum
    cum = str(row.get("Cum", "No")).strip().upper()
    is_cum = cum == "YES"
    
    # QDI
    qdi = str(row.get("QDI", "No")).strip().upper()
    is_qdi = qdi == "YES" or qdi == "VAR"
    
    # Is Floating if rate contains FLOAT
    is_floating = "FLOAT" in rate.upper()
    
    output[ticker] = {
        "name": issuer,
        "sector": sector,
        "type": sec_type,
        "rate": rate,
        "coupon": coupon_str,
        "raw_coupon": raw_coupon,
        "yield": yield_str,
        "price": price,
        "call_date": call_date,
        "maturity": maturity,
        "sp_rating": sp_rating,
        "moody_rating": moody_rating,
        "is_ig": is_ig,
        "is_cum": is_cum,
        "is_qdi": is_qdi,
        "is_floating": is_floating
    }

# Write to master_metadata.json
with open("master_metadata.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"Processed {len(output)} tickers. Saved to master_metadata.json")
