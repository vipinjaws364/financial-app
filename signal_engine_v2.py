from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import anthropic
import requests
import yfinance as yf
from dotenv import load_dotenv
from supabase import create_client

from beta_screener import get_top_momentum_stocks
from option_chain import (
    calculate_max_pain,
    calculate_pcr,
    detect_oi_buildup,
    get_option_chain,
)

load_dotenv()

IST = ZoneInfo("Asia/Kolkata")
ANTHROPIC_MODEL = "claude-sonnet-4-5"
CAPITAL_RS = 1_000_000


def _get_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _supabase():
    return create_client(_get_env("SUPABASE_URL"), _get_env("SUPABASE_KEY"))


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if block:
        text = block.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Claude response did not contain JSON.")
    return json.loads(text[start : end + 1])


def _fetch_today_news(symbol: str, max_articles: int = 6) -> List[Dict[str, str]]:
    try:
        key = _get_env("NEWSAPI_KEY")
        today = datetime.now(IST).date().isoformat()
        params = {
            "q": f'"{symbol}" OR "{symbol} stock" India NSE',
            "apiKey": key,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": max_articles,
            "from": today,
        }
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        out = []
        for a in data.get("articles", [])[:max_articles]:
            out.append(
                {
                    "title": (a.get("title") or "").strip(),
                    "description": (a.get("description") or "").strip(),
                    "source": ((a.get("source") or {}).get("name") or "").strip(),
                }
            )
        return [x for x in out if x["title"]]
    except Exception:
        return []


def _stock_technicals(symbol: str) -> Dict[str, Any]:
    ticker = yf.Ticker(f"{symbol}.NS")
    hist = ticker.history(period="6mo", interval="1d")
    if hist.empty or len(hist) < 40:
        raise RuntimeError(f"Insufficient technical history for {symbol}.")

    close = hist["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    day = hist.iloc[-1]
    prev = hist.iloc[-2]
    today_change_pct = ((float(day["Close"]) - float(prev["Close"])) / float(prev["Close"]) * 100.0) if float(prev["Close"]) else 0.0
    avg_vol_20 = float(hist["Volume"].tail(20).mean())
    vol_ratio = float(day["Volume"]) / avg_vol_20 if avg_vol_20 > 0 else 1.0
    support = float(close.tail(20).min())
    resistance = float(close.tail(20).max())

    return {
        "last_price": round(float(day["Close"]), 2),
        "today_change_pct": round(today_change_pct, 2),
        "volume_ratio_20d": round(vol_ratio, 3),
        "rsi_14": round(float(rsi.iloc[-1]), 2),
        "macd": round(float(macd.iloc[-1]), 4),
        "macd_signal": round(float(macd_signal.iloc[-1]), 4),
        "support_20d": round(support, 2),
        "resistance_20d": round(resistance, 2),
    }


def _india_vix() -> float:
    vix = yf.Ticker("^INDIAVIX").history(period="10d", interval="1d")
    if vix.empty:
        return 0.0
    return round(float(vix["Close"].iloc[-1]), 2)


def _next_expiry_for_symbol(symbol: str) -> str:
    # Uses yfinance option expiries as practical fallback.
    expiries = yf.Ticker(f"{symbol}.NS").options
    if not expiries:
        raise RuntimeError(f"No option expiries from yfinance for {symbol}")
    return expiries[0]


def _build_candidate_payload() -> Dict[str, Any]:
    screened = get_top_momentum_stocks()
    candidates = []
    for row in screened[:5]:
        symbol = row["ticker"]
        expiry = _next_expiry_for_symbol(symbol)
        chain = get_option_chain(symbol, expiry)
        pcr = calculate_pcr(chain)
        pain = calculate_max_pain(chain)
        oi_view = detect_oi_buildup(symbol, chain)
        tech = _stock_technicals(symbol)
        news = _fetch_today_news(symbol, max_articles=6)
        candidates.append(
            {
                "ticker": symbol,
                "expiry": expiry,
                "beta_momentum": row,
                "option_chain_snapshot": chain,
                "pcr": pcr,
                "max_pain": pain,
                "oi_buildup": oi_view,
                "technicals": tech,
                "news_today": news,
            }
        )
    return {
        "india_vix": _india_vix(),
        "generated_at_ist": datetime.now(IST).isoformat(),
        "candidates": candidates,
    }


def _call_claude(payload: Dict[str, Any]) -> Dict[str, Any]:
    client = anthropic.Anthropic(api_key=_get_env("ANTHROPIC_API_KEY"))
    prompt = f"""You are selecting ONE intraday options trade for NSE F&O.

Use only these 4 weighted layers:
Layer 1 Options Positioning (35%)
- PCR >1.2 bearish, <0.8 bullish
- Max pain below spot = bearish gravity
- Fresh call writing bearish; fresh put writing bullish
- OI unwinding can mean reversal

Layer 2 Price + Volume Action (30%)
- Price direction and momentum
- Volume vs 20-day average
- High volume + price up => accumulation
- High volume + price down => distribution

Layer 3 Technical Momentum (20%)
- RSI, MACD
- Beta adjusted move expectations
- Support/resistance context

Layer 4 Stock Specific News (15%)
- Only same-day news
- If results day is indicated in news, skip stock
- Block deals / board meetings should be explicitly flagged

Universe payload:
{json.dumps(payload, default=str)}

Return strict JSON only with:
ticker, direction(BUY/SELL), recommended_strike, expiry,
entry_premium, stop_loss_premium, target_premium,
lot_size, confidence_score,
layer_scores{{layer1_options_positioning, layer2_price_volume_action, layer3_technical_momentum, layer4_news}},
institutional_footprint, reasoning

Rules:
- Use realistic option premium values from chain snapshot.
- Compute lot_size as total quantity deployable close to Rs 10L capital.
- For BUY: target_premium > entry_premium > stop_loss_premium
- For SELL: stop_loss_premium > entry_premium > target_premium
- Keep reasoning concise but specific about options positioning and smart money.
"""
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1600,
        messages=[{"role": "user", "content": prompt}],
    )
    text = ""
    for block in msg.content:
        if block.type == "text":
            text += block.text
    return _extract_json_object(text)


def _validate_signal(signal: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    required = [
        "ticker",
        "direction",
        "recommended_strike",
        "expiry",
        "entry_premium",
        "stop_loss_premium",
        "target_premium",
        "lot_size",
        "confidence_score",
        "layer_scores",
        "institutional_footprint",
        "reasoning",
    ]
    for k in required:
        if k not in signal:
            raise ValueError(f"Missing key: {k}")

    direction = str(signal["direction"]).upper()
    if direction not in ("BUY", "SELL"):
        raise ValueError("direction must be BUY or SELL")

    entry = float(signal["entry_premium"])
    stop = float(signal["stop_loss_premium"])
    target = float(signal["target_premium"])
    if direction == "BUY" and not (target > entry > stop):
        raise ValueError("BUY premium levels invalid")
    if direction == "SELL" and not (stop > entry > target):
        raise ValueError("SELL premium levels invalid")

    ticker = str(signal["ticker"]).upper().replace(".NS", "")
    candidate = next((c for c in payload["candidates"] if c["ticker"] == ticker), None)
    lot_unit = 1
    if candidate:
        strikes = candidate["option_chain_snapshot"]["strikes"]
        for row in strikes:
            if float(row["strike"]) == float(signal["recommended_strike"]):
                ce = row.get("CE") or {}
                pe = row.get("PE") or {}
                lot_unit = int((ce.get("lot_size") or pe.get("lot_size") or 1))
                break

    desired_qty = int(CAPITAL_RS // max(entry, 0.01))
    lots = max(1, desired_qty // max(lot_unit, 1))
    final_qty = lots * lot_unit

    return {
        "ticker": ticker,
        "direction": direction,
        "recommended_strike": float(signal["recommended_strike"]),
        "expiry": str(signal["expiry"]),
        "entry_premium": round(entry, 2),
        "stop_loss_premium": round(stop, 2),
        "target_premium": round(target, 2),
        "lot_size": int(final_qty),
        "confidence_score": round(float(signal["confidence_score"]), 1),
        "layer_scores": signal["layer_scores"],
        "institutional_footprint": str(signal["institutional_footprint"])[:1200],
        "reasoning": str(signal["reasoning"])[:2500],
    }


def _save_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
    sb = _supabase()
    row = {
        "signal_date": datetime.now(IST).date().isoformat(),
        "ticker": signal["ticker"],
        "direction": signal["direction"],
        "entry_price": signal["entry_premium"],
        "target_price": signal["target_premium"],
        "stop_loss": signal["stop_loss_premium"],
        "confidence_score": signal["confidence_score"],
        "reasoning": json.dumps(
            {
                "reasoning": signal["reasoning"],
                "recommended_strike": signal["recommended_strike"],
                "expiry": signal["expiry"],
                "layer_scores": signal["layer_scores"],
                "institutional_footprint": signal["institutional_footprint"],
                "lot_size": signal["lot_size"],
            }
        ),
        "result": "PENDING",
        "acted": False,
    }
    res = sb.table("daily_signals").insert(row).execute()
    if not res.data:
        raise RuntimeError("Supabase insert failed.")
    saved = res.data[0]
    return {**signal, "id": saved.get("id"), "signal_date": row["signal_date"]}


def run_signal_generation() -> Dict[str, Any]:
    payload = _build_candidate_payload()
    raw = _call_claude(payload)
    valid = _validate_signal(raw, payload)
    return _save_signal(valid)


def run_signal_generation_with_nifty() -> Dict[str, Any]:
    """
    Combined capability: stock signal + Nifty weekly options signal.
    """
    from nifty_signal import generate_nifty_weekly_signal

    return {
        "stock_signal": run_signal_generation(),
        "nifty_signal": generate_nifty_weekly_signal(),
    }


if __name__ == "__main__":
    print(json.dumps(run_signal_generation(), indent=2))
