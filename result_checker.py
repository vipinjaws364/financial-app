"""
Run daily after market close (3:30 PM IST): evaluate PENDING signals vs day's OHLC.
"""
from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import schedule
import yfinance as yf
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

IST = ZoneInfo("Asia/Kolkata")

_last_check_date: Optional[date] = None


def _get_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v.strip()


def _supabase():
    return create_client(_get_env("SUPABASE_URL"), _get_env("SUPABASE_KEY"))


def _parse_signal_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    if isinstance(value, datetime):
        return value.date()
    raise ValueError(f"Bad signal_date: {value!r}")


def _day_ohlc(yahoo_symbol: str, d: date) -> Optional[Dict[str, float]]:
    """Daily OHLC for `d` from recent history (handles exchange timezone index)."""
    t = yf.Ticker(yahoo_symbol)
    hist = t.history(period="30d", interval="1d")
    if hist.empty:
        return None
    for idx, row in hist.iterrows():
        ts = pd.Timestamp(idx)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("Asia/Kolkata")
        row_date = ts.date()
        if row_date == d:
            return {
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
            }
    return None


def resolve_outcome(
    direction: str,
    target_price: float,
    stop_loss: float,
    high: float,
    low: float,
) -> str:
    direction = direction.upper().strip()
    if direction == "BUY":
        if low <= stop_loss:
            return "LOSS"
        if high >= target_price:
            return "WIN"
        return "EXPIRED"
    if direction == "SELL":
        if high >= stop_loss:
            return "LOSS"
        if low <= target_price:
            return "WIN"
        return "EXPIRED"
    raise ValueError(f"Unknown direction: {direction}")


def check_pending_signals() -> int:
    """Fetch all PENDING rows, update WIN/LOSS/EXPIRED. Returns count updated."""
    sb = _supabase()
    res = sb.table("daily_signals").select("*").eq("result", "PENDING").execute()
    rows: List[Dict[str, Any]] = res.data or []
    updated = 0
    for row in rows:
        rid = row["id"]
        ticker = str(row["ticker"]).upper().replace(".NS", "")
        yahoo = f"{ticker}.NS"
        d = _parse_signal_date(row["signal_date"])
        target = float(row["target_price"])
        stop = float(row["stop_loss"])
        direction = str(row["direction"])

        ohlc = _day_ohlc(yahoo, d)
        if ohlc is None:
            continue

        outcome = resolve_outcome(direction, target, stop, ohlc["high"], ohlc["low"])
        sb.table("daily_signals").update({"result": outcome}).eq("id", rid).execute()
        updated += 1
    return updated


def run_scheduled_loop() -> None:
    """
    Poll every 60s; once per IST calendar day when time is >= 15:30 IST,
    run the checker (market close batch).
    """
    global _last_check_date

    def tick() -> None:
        global _last_check_date
        now = datetime.now(IST)
        today = now.date()
        if now.hour < 15 or (now.hour == 15 and now.minute < 30):
            return
        if _last_check_date == today:
            return
        check_pending_signals()
        _last_check_date = today

    schedule.every(60).seconds.do(tick)
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    run_scheduled_loop()
