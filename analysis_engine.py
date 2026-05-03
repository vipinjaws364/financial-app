"""
NSE intraday signal analysis: technical indicators, NewsAPI, Claude, Supabase.
"""
from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import anthropic
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

IST = ZoneInfo("Asia/Kolkata")

# Fixed liquid NSE large-caps (Yahoo suffix .NS)
BASE_TICKERS: List[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "HINDUNILVR.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "AXISBANK.NS",
    "ASIANPAINT.NS",
    "MARUTI.NS",
    "SUNPHARMA.NS",
    "TITAN.NS",
    "BAJFINANCE.NS",
    "WIPRO.NS",
    "ULTRACEMCO.NS",
    "NESTLEIND.NS",
]

ANTHROPIC_MODEL = "claude-sonnet-4-5"


def _get_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v.strip()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _short_name(yahoo_symbol: str) -> str:
    return yahoo_symbol.replace(".NS", "")


def fetch_sorted_by_volume() -> List[str]:
    """Return the 20 tickers sorted by latest daily volume (desc)."""
    volumes: List[Tuple[str, float]] = []
    for sym in BASE_TICKERS:
        t = yf.Ticker(sym)
        hist = t.history(period="5d", interval="1d")
        if hist.empty or "Volume" not in hist.columns:
            volumes.append((sym, 0.0))
            continue
        vol = float(hist["Volume"].iloc[-1])
        if pd.isna(vol):
            vol = 0.0
        volumes.append((sym, vol))
    volumes.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in volumes]


def _compute_stock_metrics(symbol: str) -> Dict[str, Any]:
    short = _short_name(symbol)
    t = yf.Ticker(symbol)
    hist = t.history(period="6mo", interval="1d")
    if hist.empty or len(hist) < 30:
        raise ValueError(f"Insufficient history for {symbol}")

    close = hist["Close"]
    vol = hist["Volume"]

    rsi_series = _rsi(close, 14)
    macd_line, signal_line, histogram = _macd(close)

    last_close = float(close.iloc[-1])
    rsi_last = float(rsi_series.iloc[-1])
    macd_last = float(macd_line.iloc[-1])
    macd_sig_last = float(signal_line.iloc[-1])
    macd_hist_last = float(histogram.iloc[-1])

    vol_last = float(vol.iloc[-1])
    vol_ma20 = float(vol.tail(20).mean())
    vol_spike_ratio = vol_last / vol_ma20 if vol_ma20 > 0 else 1.0

    ma20 = float(close.tail(20).mean())
    price_vs_ma20_pct = ((last_close - ma20) / ma20 * 100.0) if ma20 else 0.0

    return {
        "ticker": short,
        "yahoo": symbol,
        "last_close": round(last_close, 2),
        "rsi_14": round(rsi_last, 2),
        "macd": round(macd_last, 4),
        "macd_signal": round(macd_sig_last, 4),
        "macd_histogram": round(macd_hist_last, 4),
        "volume_today": int(vol_last),
        "volume_ma20": int(vol_ma20),
        "volume_spike_vs_ma20": round(vol_spike_ratio, 3),
        "ma20": round(ma20, 2),
        "price_vs_ma20_pct": round(price_vs_ma20_pct, 2),
    }


def fetch_news_headlines(ticker_short: str, max_articles: int = 5) -> List[Dict[str, str]]:
    try:
        key = _get_env("NEWSAPI_KEY")
        q = f'"{ticker_short}" OR "{ticker_short} stock" India NSE'
        today_ist = datetime.now(IST).date()
        params = {
            "q": q,
            "apiKey": key,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": max_articles,
            "from": today_ist.isoformat(),
        }
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        out: List[Dict[str, str]] = []
        for art in data.get("articles", [])[:max_articles]:
            title = (art.get("title") or "").strip()
            desc = (art.get("description") or "").strip()
            src = (art.get("source") or {}).get("name", "")
            if title:
                out.append({"title": title, "description": desc[:280], "source": src})
        return out
    except Exception:
        return []


def _build_analysis_payload(
    ordered_tickers: List[str],
) -> Tuple[List[Dict[str, Any]], str]:
    rows: List[Dict[str, Any]] = []
    summary_lines: List[str] = []
    for sym in ordered_tickers:
        try:
            m = _compute_stock_metrics(sym)
            news = fetch_news_headlines(m["ticker"])
            m["news"] = news
            rows.append(m)
            n_txt = "; ".join(n["title"] for n in news[:3]) if news else "No headlines today."
            summary_lines.append(
                f"- {m['ticker']}: close={m['last_close']}, RSI={m['rsi_14']}, "
                f"MACD={m['macd']}, MACD_signal={m['macd_signal']}, hist={m['macd_histogram']}, "
                f"vol_spike_x={m['volume_spike_vs_ma20']} vs 20d avg, "
                f"price_vs_MA20={m['price_vs_ma20_pct']}%. News: {n_txt}"
            )
        except Exception:
            continue
    if not rows:
        raise RuntimeError("Could not compute metrics for any NSE ticker (check network/data).")
    return rows, "\n".join(summary_lines)


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if block:
        text = block.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Claude response did not contain a JSON object.")
    return json.loads(text[start : end + 1])


def _claude_pick_signal(analysis_summary: str, raw_rows_json: str) -> Dict[str, Any]:
    api_key = _get_env("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""You are an expert Indian equity intraday trader focused on NSE.

Below is technical + news context for 20 large-cap NSE stocks (volume-ranked order matters for liquidity).
Each line includes: RSI(14), MACD components, volume spike vs 20-day average, price distance from 20-day MA, and today's news snippets.

{analysis_summary}

Full structured rows (JSON array for your reference, do not echo verbatim):
{raw_rows_json}

Task:
1) Pick exactly ONE stock that has the highest realistic probability of a favorable intraday move TODAY for an intraday trade (not investment advice; probabilistic).
2) Output strict JSON only (no markdown) with these keys:
   - ticker: NSE symbol WITHOUT .NS (e.g. RELIANCE)
   - direction: either BUY or SELL
   - entry_price: number (INR, 2 decimals)
   - target_price: number (INR, 2 decimals)
   - stop_loss: number (INR, 2 decimals)
   - confidence_score: number from 0 to 100
   - reasoning: concise string explaining the edge (max ~400 chars)

Rules:
- For BUY: target_price > entry_price > stop_loss
- For SELL: stop_loss > entry_price > target_price
- Use realistic NSE-style levels based on the last_close in the data.
- JSON only, double quotes, no trailing commentary."""

    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = ""
    for chunk in msg.content:
        if chunk.type == "text":
            text += chunk.text
    return _extract_json_object(text)


def _validate_signal(s: Dict[str, Any]) -> Dict[str, Any]:
    required = [
        "ticker",
        "direction",
        "entry_price",
        "target_price",
        "stop_loss",
        "confidence_score",
        "reasoning",
    ]
    for k in required:
        if k not in s:
            raise ValueError(f"Missing key in signal: {k}")

    direction = str(s["direction"]).upper().strip()
    if direction not in ("BUY", "SELL"):
        raise ValueError("direction must be BUY or SELL")

    entry = float(s["entry_price"])
    target = float(s["target_price"])
    stop = float(s["stop_loss"])
    conf = float(s["confidence_score"])
    conf = max(0.0, min(100.0, conf))

    if direction == "BUY" and not (target > entry > stop):
        raise ValueError("BUY requires target > entry > stop_loss")
    if direction == "SELL" and not (stop > entry > target):
        raise ValueError("SELL requires stop_loss > entry > target")

    return {
        "ticker": str(s["ticker"]).upper().replace(".NS", ""),
        "direction": direction,
        "entry_price": round(entry, 2),
        "target_price": round(target, 2),
        "stop_loss": round(stop, 2),
        "confidence_score": round(conf, 1),
        "reasoning": str(s["reasoning"])[:2000],
    }


def _supabase():
    url = _get_env("SUPABASE_URL")
    key = _get_env("SUPABASE_KEY")
    return create_client(url, key)


def save_signal_to_supabase(signal: Dict[str, Any], signal_date: date) -> Dict[str, Any]:
    sb = _supabase()
    row = {
        "signal_date": signal_date.isoformat(),
        "ticker": signal["ticker"],
        "direction": signal["direction"],
        "entry_price": signal["entry_price"],
        "target_price": signal["target_price"],
        "stop_loss": signal["stop_loss"],
        "confidence_score": signal["confidence_score"],
        "reasoning": signal["reasoning"],
        "result": "PENDING",
        "acted": False,
    }
    res = sb.table("daily_signals").insert(row).execute()
    if not res.data:
        raise RuntimeError("Supabase insert returned no data.")
    saved = res.data[0]
    return {**signal, "id": saved.get("id"), "signal_date": signal_date.isoformat()}


def run_analysis() -> Dict[str, Any]:
    """
    Full pipeline: rank by volume, compute metrics + news, Claude pick, Supabase save.
    Returns the signal dict including id and signal_date.
    """
    ordered = fetch_sorted_by_volume()
    rows, summary = _build_analysis_payload(ordered)
    raw_json = json.dumps(rows, default=str)
    claude_raw = _claude_pick_signal(summary, raw_json)
    validated = _validate_signal(claude_raw)
    signal_date = datetime.now(IST).date()
    return save_signal_to_supabase(validated, signal_date)


if __name__ == "__main__":
    out = run_analysis()
    print(json.dumps(out, indent=2))
