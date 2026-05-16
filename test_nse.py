import requests
import json

def get_option_chain(symbol):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/option-chain",
        "Connection": "keep-alive",
    }
    
    session = requests.Session()
    
    # First hit NSE homepage to get cookies
    session.get("https://www.nseindia.com", headers=headers, timeout=10)
    session.get("https://www.nseindia.com/option-chain", headers=headers, timeout=10)
    
    # Now fetch option chain
    url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
    response = session.get(url, headers=headers, timeout=10)
    
    data = response.json()
    return data

data = get_option_chain("SBIN")

if data:
    records = data.get('records', {})
    print(f"Expiry dates: {records.get('expiryDates', [])}")
    print(f"Underlying value: {records.get('underlyingValue', 'N/A')}")
    print(f"Total records: {len(records.get('data', []))}")
else:
    print("No data returned")
