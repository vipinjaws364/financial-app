from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import yfinance as yf
from dotenv import load_dotenv
from kiteconnect import KiteConnect

load_dotenv()

IST = ZoneInfo("Asia/Kolkata")
CAPITAL_RS = 1_000_000
TOKEN_FILE = Path("access_token.txt")


def _get_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _load_access_token() -> str:
    if not TOKEN_FILE.exists():
        raise RuntimeError("access_token.txt not found. Authenticate Kite first.")
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("access_token.txt is empty.")
    return token


def _kite() -> KiteConnect:
    kite = KiteConnect(api_key=_get_env("KITE_API_KEY"))
    kite.set_access_token(_load_access_token())
    return kite


def _next_tuesday(from_date: date) -> date:
    # Monday=0 ... Tuesday=1
    delta = (1 - from_date.weekday()) % 7
    return from_date + timedelta(days=delta)


def _time_guard(now: datetime, expiry: date) -> Optional[str]:
    if now.hour > 14 or (now.hour == 14 and now.minute >= 30):
        return "No signal after 2:30 PM IST due to gamma risk."
    if now.date() == expiry and (now.hour > 13 or (now.hour == 13 and now.minute >= 0)):
        return "No signal on expiry day after 1:00 PM IST due to accelerated theta decay."
    return None


def _india_vix_info() -> Dict[str, Any]:
    hist = yf.Ticker("^INDIAVIX").history(period="10d", interval="1d")
    if hist.empty or len(hist) < 2:
        return {"value": 0.0, "direction": "unknown", "condition": "unknown"}
    latest = float(hist["Close"].iloc[-1])
    prev = float(hist["Close"].iloc[-2])
    direction = "rising" if latest > prev else "falling"

    if latest > 15:
        condition = "high volatility, premium expensive"
    elif latest < 12:
        condition = "low volatility, premium cheap"
    else:
        condition = "moderate volatility"
    return {"value": round(latest, 2), "direction": direction, "condition": condition}


def _nifty_spot_and_gap() -> Dict[str, Any]:
    hist = yf.Ticker("^NSEI").history(period="7d", interval="1d")
    if hist.empty or len(hist) < 2:
        raise RuntimeError("Unable to fetch Nifty history from yfinance.")
    latest = hist.iloc[-1]
    prev = hist.iloc[-2]
    spot = float(latest["Close"])
    prev_close = float(prev["Close"])
    gift_proxy_gap_pct = ((spot - prev_close) / prev_close * 100.0) if prev_close else 0.0
    return {"spot": round(spot, 2), "previous_close": round(prev_close, 2), "gift_nifty_proxy_gap_pct": round(gift_proxy_gap_pct, 2)}


def _atm_strike(spot: float) -> int:
    return int(round(spot / 50.0) * 50)


def _extract_strike_and_type(ts: str) -> Tuple[Optional[int], Optional[str]]:
    if ts.endswith("CE") or ts.endswith("PE"):
        opt_type = ts[-2:]
        digits = "".join(ch for ch in ts if ch.isdigit())
        if digits:
            return int(digits[-5:]), opt_type
    return None, None


def _fetch_nifty_chain(expiry: date, spot: float) -> Dict[str, Any]:
    kite = _kite()
    instruments = kite.instruments("NFO")
    lo = spot - 500
    hi = spot + 500
    options = []
    quote_keys = []

    for ins in instruments:
        if ins.get("segment") != "NFO-OPT":
            continue
        if ins.get("name") != "NIFTY":
            continue
        if ins.get("expiry") != expiry:
            continue
        ts = ins["tradingsymbol"]
        strike = int(ins.get("strike") or 0)
        if strike <= 0:
            parsed, _ = _extract_strike_and_type(ts)
            if parsed:
                strike = parsed
        opt_type = ins.get("instrument_type") or ("CE" if ts.endswith("CE") else "PE")
        if strike < lo or strike > hi:
            continue
        options.append(
            {"tradingsymbol": ts, "strike": strike, "type": opt_type, "lot_size": int(ins.get("lot_size") or 0)}
        )
        quote_keys.append(f"NFO:{ts}")

    if not options:
        raise RuntimeError("No NIFTY weekly options found for requested expiry range.")

    quotes = kite.quote(quote_keys)
    rows: Dict[int, Dict[str, Any]] = {}
    for opt in options:
        q = quotes.get(f"NFO:{opt['tradingsymbol']}", {})
        leg = {
            "oi": int(q.get("oi") or 0),
            "change_in_oi": q.get("oi_day_change") if q.get("oi_day_change") is not None else 0,
            "volume": int(q.get("volume") or 0),
            "ltp": float(q.get("last_price") or 0.0),
            "iv": q.get("implied_volatility"),
            "open": float((q.get("ohlc") or {}).get("open") or 0.0),
            "lot_size": opt["lot_size"],
            "tradingsymbol": opt["tradingsymbol"],
        }
        row = rows.setdefault(opt["strike"], {"strike": opt["strike"], "CE": None, "PE": None})
        row[opt["type"]] = leg
    return {"strikes": sorted(rows.values(), key=lambda x: x["strike"])}


def _calc_max_pain(chain: Dict[str, Any]) -> int:
    strikes = chain["strikes"]
    strike_vals = [int(s["strike"]) for s in strikes]
    best = None
    for settle in strike_vals:
        pain = 0.0
        for row in strikes:
            k = int(row["strike"])
            ce_oi = float((row.get("CE") or {}).get("oi") or 0)
            pe_oi = float((row.get("PE") or {}).get("oi") or 0)
            pain += max(0, k - settle) * ce_oi
            pain += max(0, settle - k) * pe_oi
        if best is None or pain < best[1]:
            best = (settle, pain)
    return int(best[0]) if best else strike_vals[len(strike_vals) // 2]


def _calc_pcr(chain: Dict[str, Any]) -> float:
    put_oi = 0.0
    call_oi = 0.0
    for row in chain["strikes"]:
        put_oi += float((row.get("PE") or {}).get("oi") or 0)
        call_oi += float((row.get("CE") or {}).get("oi") or 0)
    return round((put_oi / call_oi), 3) if call_oi > 0 else 0.0


def _find_row(strikes: List[Dict[str, Any]], strike: int) -> Dict[str, Any]:
    for row in strikes:
        if int(row["strike"]) == int(strike):
            return row
    return {"strike": strike, "CE": {}, "PE": {}}


def _oi_change_direction(atm_row: Dict[str, Any]) -> str:
    ce_ch = float((atm_row.get("CE") or {}).get("change_in_oi") or 0)
    pe_ch = float((atm_row.get("PE") or {}).get("change_in_oi") or 0)
    if ce_ch > 0 and pe_ch > 0:
        return "both_sides_fresh_buildup"
    if ce_ch < 0 and pe_ch < 0:
        return "both_sides_unwinding"
    if pe_ch > ce_ch:
        return "put_side_fresh_buildup"
    if ce_ch > pe_ch:
        return "call_side_fresh_buildup"
    return "mixed"


def _derive_direction(
    pcr: float,
    max_pain: int,
    spot: float,
    atm_row: Dict[str, Any],
    gap_pct: float,
    vix: Dict[str, Any],
) -> Tuple[str, int, float, str]:
    ce_oi = float((atm_row.get("CE") or {}).get("oi") or 0)
    pe_oi = float((atm_row.get("PE") or {}).get("oi") or 0)
    ce_ch = float((atm_row.get("CE") or {}).get("change_in_oi") or 0)
    pe_ch = float((atm_row.get("PE") or {}).get("change_in_oi") or 0)

    bullish = 0
    bearish = 0
    reasons = []

    if pcr < 0.8:
        bullish += 2
        reasons.append("PCR below 0.8 supports bullish bias")
    elif pcr > 1.2:
        bearish += 2
        reasons.append("PCR above 1.2 supports bearish bias")

    if max_pain > spot:
        bullish += 1
        reasons.append("Max pain above spot suggests upward gravity")
    elif max_pain < spot:
        bearish += 1
        reasons.append("Max pain below spot suggests bearish gravity")

    if pe_oi > ce_oi:
        bullish += 1
        reasons.append("ATM put OI heavier than call OI")
    elif ce_oi > pe_oi:
        bearish += 1
        reasons.append("ATM call OI heavier than put OI")

    if pe_ch > ce_ch:
        bullish += 1
        reasons.append("Fresh OI adding more on put side")
    elif ce_ch > pe_ch:
        bearish += 1
        reasons.append("Fresh OI adding more on call side")

    if gap_pct > 0:
        bullish += 1
        reasons.append("Gift Nifty proxy gap positive")
    elif gap_pct < 0:
        bearish += 1
        reasons.append("Gift Nifty proxy gap negative")

    if vix["direction"] == "rising":
        reasons.append("VIX rising favors option buying")
    else:
        reasons.append("VIX falling favors premium selling")

    atm = int(atm_row["strike"])
    if bullish >= bearish:
        direction = "BUY call"
        rec = atm if bullish - bearish <= 1 else atm + 50
        confidence = min(90.0, 55.0 + (bullish - bearish) * 7.0)
    else:
        direction = "BUY put"
        rec = atm if bearish - bullish <= 1 else atm - 50
        confidence = min(90.0, 55.0 + (bearish - bullish) * 7.0)
    return direction, rec, round(confidence, 1), "; ".join(reasons)


def generate_nifty_weekly_signal() -> Dict[str, Any]:
    now = datetime.now(IST)
    expiry = _next_tuesday(now.date())

    block = _time_guard(now, expiry)
    if block:
        raise RuntimeError(block)

    vix = _india_vix_info()
    if float(vix["value"]) > 20.0:
        raise RuntimeError("VIX above 20: market too chaotic, no signal generated.")

    spot_data = _nifty_spot_and_gap()
    spot = float(spot_data["spot"])
    atm = _atm_strike(spot)

    chain = _fetch_nifty_chain(expiry, spot)
    strikes = chain["strikes"]
    atm_row = _find_row(strikes, atm)
    max_pain = _calc_max_pain(chain)
    pcr = _calc_pcr(chain)
    oi_dir = _oi_change_direction(atm_row)

    direction, rec_strike, confidence, reason_base = _derive_direction(
        pcr=pcr,
        max_pain=max_pain,
        spot=spot,
        atm_row=atm_row,
        gap_pct=float(spot_data["gift_nifty_proxy_gap_pct"]),
        vix=vix,
    )

    rec_row = _find_row(strikes, rec_strike)
    leg = rec_row.get("CE") if direction == "BUY call" else rec_row.get("PE")
    if not leg:
        rec_strike = atm
        rec_row = _find_row(strikes, rec_strike)
        leg = rec_row.get("CE") if direction == "BUY call" else rec_row.get("PE")
    if not leg:
        raise RuntimeError("No tradable option leg found for recommended strike.")

    entry = float(leg.get("ltp") or 0.0)
    if entry <= 0:
        raise RuntimeError("Recommended option LTP unavailable.")
    stop = round(entry * 0.5, 2)
    target = round(entry * 2.0, 2)
    lot_unit = int(leg.get("lot_size") or 50)
    qty = int(CAPITAL_RS // entry)
    lots = max(1, qty // max(lot_unit, 1))
    total_qty = lots * lot_unit

    atm_ce_oi = int((atm_row.get("CE") or {}).get("oi") or 0)
    atm_pe_oi = int((atm_row.get("PE") or {}).get("oi") or 0)
    atm_ce_ch = float((atm_row.get("CE") or {}).get("change_in_oi") or 0)
    atm_pe_ch = float((atm_row.get("PE") or {}).get("change_in_oi") or 0)

    vix_condition = "favorable" if float(vix["value"]) <= 15 else "unfavorable"
    reasoning = (
        f"Max Pain at {max_pain} vs spot {spot:.2f}; overall PCR {pcr}. "
        f"ATM OI CE={atm_ce_oi}, PE={atm_pe_oi}, OI change CE={atm_ce_ch:.0f}, PE={atm_pe_ch:.0f} ({oi_dir}). "
        f"VIX {vix['value']} ({vix['direction']}, {vix['condition']}). "
        f"{reason_base}"
    )

    return {
        "symbol": "NIFTY",
        "direction": direction,
        "atm_strike": atm,
        "recommended_strike": rec_strike,
        "expiry": expiry.isoformat(),
        "entry_premium": round(entry, 2),
        "stop_loss_premium": stop,
        "target_premium": target,
        "lot_size": int(total_qty),
        "confidence_score": confidence,
        "vix_condition": vix_condition,
        "nifty_spot_price": round(spot, 2),
        "max_pain_strike": max_pain,
        "overall_pcr": pcr,
        "atm_call_oi": atm_ce_oi,
        "atm_put_oi": atm_pe_oi,
        "atm_call_change_oi": round(atm_ce_ch, 2),
        "atm_put_change_oi": round(atm_pe_ch, 2),
        "india_vix": vix,
        "reasoning": reasoning,
    }

