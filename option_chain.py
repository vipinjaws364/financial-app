from __future__ import annotations

import math
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from kiteconnect import KiteConnect

load_dotenv()

TOKEN_FILE = Path("access_token.txt")


def _get_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _load_access_token() -> str:
    if not TOKEN_FILE.exists():
        raise RuntimeError("access_token.txt not found. Run Kite auth first.")
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("access_token.txt is empty. Re-authenticate with Kite.")
    return token


def _kite() -> KiteConnect:
    kite = KiteConnect(api_key=_get_env("KITE_API_KEY"))
    kite.set_access_token(_load_access_token())
    return kite


def _parse_expiry(expiry: str) -> date:
    return datetime.strptime(expiry.strip(), "%Y-%m-%d").date()


def _extract_strike_and_type(tradingsymbol: str) -> Tuple[Optional[float], Optional[str]]:
    match = re.search(r"(\d+(?:\.\d+)?)(CE|PE)$", tradingsymbol)
    if not match:
        return None, None
    return float(match.group(1)), match.group(2)


def get_option_chain(symbol: str, expiry: str) -> Dict[str, Any]:
    """
    Returns option chain for an NSE F&O symbol for a given expiry (YYYY-MM-DD).
    Includes strike-level CE/PE OI, change in OI (if available), volume, LTP, IV.
    """
    try:
        kite = _kite()
        symbol = symbol.strip().upper()
        target_expiry = _parse_expiry(expiry)

        instruments = kite.instruments("NFO")
        options = []
        quote_keys = []

        for ins in instruments:
            if ins.get("segment") != "NFO-OPT":
                continue
            if ins.get("name") != symbol:
                continue
            exp = ins.get("expiry")
            if not exp or exp != target_expiry:
                continue
            ts = ins["tradingsymbol"]
            strike, opt_type = _extract_strike_and_type(ts)
            if strike is None or opt_type is None:
                strike = float(ins.get("strike") or 0)
                opt_type = ins.get("instrument_type")
            options.append(
                {
                    "tradingsymbol": ts,
                    "strike": float(strike),
                    "type": opt_type,
                    "lot_size": int(ins.get("lot_size") or 0),
                    "instrument_token": ins.get("instrument_token"),
                }
            )
            quote_keys.append(f"NFO:{ts}")

        if not options:
            raise RuntimeError(f"No option contracts found for {symbol} {expiry}.")

        quotes = kite.quote(quote_keys)
        spot_quote = kite.quote([f"NSE:{symbol}"]).get(f"NSE:{symbol}", {})
        spot_price = float(spot_quote.get("last_price") or 0.0)

        strike_map: Dict[float, Dict[str, Any]] = {}
        for opt in options:
            key = f"NFO:{opt['tradingsymbol']}"
            q = quotes.get(key, {})
            data = {
                "oi": int(q.get("oi") or 0),
                # Kite quote API may not always expose previous OI intraday.
                "change_in_oi": q.get("oi_day_change") if q.get("oi_day_change") is not None else None,
                "volume": int(q.get("volume") or 0),
                "ltp": float(q.get("last_price") or 0.0),
                "iv": q.get("implied_volatility"),
                "open": float((q.get("ohlc") or {}).get("open") or 0.0),
                "close": float((q.get("ohlc") or {}).get("close") or 0.0),
                "lot_size": opt["lot_size"],
                "tradingsymbol": opt["tradingsymbol"],
            }
            row = strike_map.setdefault(opt["strike"], {"strike": opt["strike"], "CE": None, "PE": None})
            row[opt["type"]] = data

        strikes = sorted(strike_map.values(), key=lambda x: x["strike"])
        return {
            "symbol": symbol,
            "expiry": expiry,
            "spot_price": spot_price,
            "strikes": strikes,
        }
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to fetch option chain: {exc}") from exc


def calculate_max_pain(option_chain_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        strikes = option_chain_data.get("strikes", [])
        spot = float(option_chain_data.get("spot_price") or 0.0)
        if not strikes:
            raise RuntimeError("Option chain is empty.")

        pain_values: List[Tuple[float, float]] = []
        strike_list = [float(s["strike"]) for s in strikes]

        for settle in strike_list:
            total_pain = 0.0
            for row in strikes:
                k = float(row["strike"])
                ce_oi = float((row.get("CE") or {}).get("oi") or 0.0)
                pe_oi = float((row.get("PE") or {}).get("oi") or 0.0)
                total_pain += max(0.0, settle - k) * pe_oi
                total_pain += max(0.0, k - settle) * ce_oi
            pain_values.append((settle, total_pain))

        max_pain_strike, _ = min(pain_values, key=lambda x: x[1])
        distance = spot - max_pain_strike
        return {
            "max_pain_strike": max_pain_strike,
            "distance_from_spot": round(distance, 2),
        }
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed max pain calculation: {exc}") from exc


def calculate_pcr(option_chain_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        strikes = option_chain_data.get("strikes", [])
        if not strikes:
            raise RuntimeError("Option chain is empty.")

        total_put_oi = 0.0
        total_call_oi = 0.0
        strikewise = []
        highest_oi_rows = []

        for row in strikes:
            ce = row.get("CE") or {}
            pe = row.get("PE") or {}
            call_oi = float(ce.get("oi") or 0.0)
            put_oi = float(pe.get("oi") or 0.0)
            total_call_oi += call_oi
            total_put_oi += put_oi
            strike_pcr = (put_oi / call_oi) if call_oi > 0 else None
            combined_oi = put_oi + call_oi
            strikewise.append(
                {
                    "strike": row["strike"],
                    "put_oi": int(put_oi),
                    "call_oi": int(call_oi),
                    "strike_pcr": round(strike_pcr, 3) if strike_pcr is not None else None,
                }
            )
            highest_oi_rows.append(
                {"strike": row["strike"], "combined_oi": int(combined_oi), "put_oi": int(put_oi), "call_oi": int(call_oi)}
            )

        overall_pcr = (total_put_oi / total_call_oi) if total_call_oi > 0 else None
        top_5 = sorted(highest_oi_rows, key=lambda x: x["combined_oi"], reverse=True)[:5]
        return {
            "overall_pcr": round(overall_pcr, 3) if overall_pcr is not None else None,
            "strikewise_pcr": strikewise,
            "top_5_oi_strikes": top_5,
        }
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed PCR calculation: {exc}") from exc


def _classify_leg(leg: Dict[str, Any], opt_type: str) -> Optional[str]:
    ltp = float(leg.get("ltp") or 0.0)
    open_px = float(leg.get("open") or 0.0)
    price_change = ((ltp - open_px) / open_px * 100.0) if open_px > 0 else 0.0
    change_oi = leg.get("change_in_oi")
    if change_oi is None:
        return None
    try:
        change_oi_f = float(change_oi)
    except Exception:  # noqa: BLE001
        return None

    if change_oi_f <= 0:
        return "OI_UNWINDING"

    if opt_type == "CE":
        if price_change >= 0:
            return "CALL_BUYING_BULLISH"
        return "CALL_WRITING_BEARISH"

    if opt_type == "PE":
        if price_change >= 0:
            return "PUT_BUYING_BEARISH"
        return "PUT_WRITING_BULLISH"
    return None


def detect_oi_buildup(symbol: str, option_chain_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        rows = option_chain_data.get("strikes", [])
        if not rows:
            raise RuntimeError("Option chain is empty.")

        counts = {
            "call_writing_bearish": 0,
            "put_writing_bullish": 0,
            "call_buying_bullish": 0,
            "put_buying_bearish": 0,
            "oi_unwinding": 0,
        }
        notes = []
        for row in rows:
            strike = row.get("strike")
            for opt_type in ("CE", "PE"):
                leg = row.get(opt_type)
                if not leg:
                    continue
                label = _classify_leg(leg, opt_type)
                if not label:
                    continue
                if label == "CALL_WRITING_BEARISH":
                    counts["call_writing_bearish"] += 1
                elif label == "PUT_WRITING_BULLISH":
                    counts["put_writing_bullish"] += 1
                elif label == "CALL_BUYING_BULLISH":
                    counts["call_buying_bullish"] += 1
                elif label == "PUT_BUYING_BEARISH":
                    counts["put_buying_bearish"] += 1
                elif label == "OI_UNWINDING":
                    counts["oi_unwinding"] += 1
                notes.append(f"{strike} {opt_type}: {label}")

        bullish = counts["put_writing_bullish"] + counts["call_buying_bullish"]
        bearish = counts["call_writing_bearish"] + counts["put_buying_bearish"]

        if bullish > bearish:
            summary = "Institutional positioning tilts bullish (put writing / call buying dominance)."
        elif bearish > bullish:
            summary = "Institutional positioning tilts bearish (call writing / put buying dominance)."
        else:
            summary = "Institutional positioning is balanced / mixed."

        if counts["oi_unwinding"] > max(3, math.floor(len(rows) * 0.3)):
            summary += " OI unwinding is visible, suggesting potential trend reversal."

        return {"symbol": symbol, "classification_counts": counts, "institutional_summary": summary, "samples": notes[:12]}
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed OI buildup detection: {exc}") from exc
