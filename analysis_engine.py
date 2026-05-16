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
LAYER_WEIGHT = 25.0

SECTOR_PEERS: Dict[str, Dict[str, Any]] = {
    "RELIANCE": {"sector": "Energy", "peers": ["ONGC.NS", "BPCL.NS", "IOC.NS"]},
    "TCS": {"sector": "IT", "peers": ["INFY.NS", "WIPRO.NS", "HCLTECH.NS"]},
    "HDFCBANK": {"sector": "Banking", "peers": ["ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS"]},
    "INFY": {"sector": "IT", "peers": ["TCS.NS", "WIPRO.NS", "HCLTECH.NS"]},
    "ICICIBANK": {"sector": "Banking", "peers": ["HDFCBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS"]},
    "HINDUNILVR": {"sector": "FMCG", "peers": ["ITC.NS", "NESTLEIND.NS", "DABUR.NS"]},
    "SBIN": {"sector": "Banking", "peers": ["BANKBARODA.NS", "PNB.NS", "CANBK.NS"]},
    "BHARTIARTL": {"sector": "Telecom", "peers": ["IDEA.NS", "INDUSTOWER.NS"]},
    "ITC": {"sector": "FMCG", "peers": ["HINDUNILVR.NS", "NESTLEIND.NS", "DABUR.NS"]},
    "KOTAKBANK": {"sector": "Banking", "peers": ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS"]},
    "LT": {"sector": "Infrastructure", "peers": ["ULTRACEMCO.NS", "SIEMENS.NS", "ABB.NS"]},
    "AXISBANK": {"sector": "Banking", "peers": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS"]},
    "ASIANPAINT": {"sector": "Paints", "peers": ["BERGEPAINT.NS", "KANSAINER.NS", "INDIGO.NS"]},
    "MARUTI": {"sector": "Auto", "peers": [ "M&M.NS", "EICHERMOT.NS"]},
    "SUNPHARMA": {"sector": "Pharma", "peers": ["DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"]},
    "TITAN": {"sector": "Consumer", "peers": ["TRENT.NS", "VBL.NS", "DMART.NS"]},
    "BAJFINANCE": {"sector": "NBFC", "peers": ["CHOLAFIN.NS", "SHRIRAMFIN.NS", "MUTHOOTFIN.NS"]},
    "WIPRO": {"sector": "IT", "peers": ["TCS.NS", "INFY.NS", "HCLTECH.NS"]},
    "ULTRACEMCO": {"sector": "Cement", "peers": ["SHREECEM.NS", "ACC.NS", "AMBUJACEM.NS"]},
    "NESTLEIND": {"sector": "FMCG", "peers": ["HINDUNILVR.NS", "ITC.NS", "DABUR.NS"]},
}


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


def _safe_pct_change(symbol: str) -> Optional[float]:
    t = yf.Ticker(symbol)
    hist = t.history(period="7d", interval="1d")
    if hist.empty or len(hist) < 2:
        return None
    last = float(hist["Close"].iloc[-1])
    prev = float(hist["Close"].iloc[-2])
    if prev == 0:
        return None
    return (last - prev) / prev * 100.0


def _intraday_price_change_pct(symbol: str) -> Tuple[float, float, float]:
    """
    Return (open_price, current_price, intraday_pct_change).
    Uses intraday candles when available; falls back to latest daily bar.
    """
    t = yf.Ticker(symbol)
    intraday = t.history(period="1d", interval="5m")
    if not intraday.empty:
        open_price = float(intraday["Open"].iloc[0])
        current_price = float(intraday["Close"].iloc[-1])
        if open_price > 0:
            return open_price, current_price, ((current_price - open_price) / open_price) * 100.0

    daily = t.history(period="5d", interval="1d")
    if daily.empty:
        raise ValueError(f"Unable to get intraday/open data for {symbol}")
    open_price = float(daily["Open"].iloc[-1])
    current_price = float(daily["Close"].iloc[-1])
    if open_price <= 0:
        return open_price, current_price, 0.0
    return open_price, current_price, ((current_price - open_price) / open_price) * 100.0


def _news_sentiment_score(headlines: List[Dict[str, str]]) -> float:
    positive_words = ("gain", "surge", "rise", "bull", "beat", "up", "record", "strong")
    negative_words = ("fall", "drop", "bear", "miss", "down", "weak", "cut", "slump")
    score = 0.0
    for item in headlines:
        txt = f"{item.get('title', '')} {item.get('description', '')}".lower()
        score += sum(1 for w in positive_words if w in txt)
        score -= sum(1 for w in negative_words if w in txt)
    return score


def _score_to_sentiment(score: float) -> str:
    if score >= 65:
        return "bullish"
    if score <= 40:
        return "bearish"
    return "neutral"


def _clamp_score(v: float) -> float:
    return max(0.0, min(100.0, v))


def _fetch_global_macro_layer() -> Dict[str, Any]:
    spx = _safe_pct_change("^GSPC") or 0.0
    crude = _safe_pct_change("CL=F") or 0.0
    dxy = _safe_pct_change("DX-Y.NYB") or 0.0

    # Risk-on: equities up, dollar softer, crude not spiking hard.
    score = 50.0 + (spx * 7.0) - (dxy * 5.0) - (max(crude, 0.0) * 2.0)
    score = _clamp_score(score)

    return {
        "score": round(score, 1),
        "sentiment": _score_to_sentiment(score),
        "inputs": {
            "sp500_change_pct": round(spx, 2),
            "crude_change_pct": round(crude, 2),
            "dxy_change_pct": round(dxy, 2),
        },
    }


def _fetch_india_market_layer() -> Dict[str, Any]:
    nifty = _safe_pct_change("^NSEI") or 0.0
    banknifty = _safe_pct_change("^NSEBANK") or 0.0
    india_news = fetch_news_headlines("India stock market NSE Nifty", max_articles=6)
    news_bias = _news_sentiment_score(india_news)

    # FII/DII often unavailable free intraday APIs; if unavailable, keep neutral contribution.
    fii_dii_note = "FII/DII flow data not directly available from yfinance; treated as neutral."
    fii_dii_bias = 0.0

    score = 50.0 + (nifty * 8.0) + (banknifty * 7.0) + (news_bias * 1.8) + fii_dii_bias
    score = _clamp_score(score)
    return {
        "score": round(score, 1),
        "sentiment": _score_to_sentiment(score),
        "inputs": {
            "nifty_change_pct": round(nifty, 2),
            "banknifty_change_pct": round(banknifty, 2),
            "india_news_count": len(india_news),
            "india_news_top": [n.get("title", "") for n in india_news[:3]],
            "fii_dii_note": fii_dii_note,
        },
    }


def _compute_stock_specific_layer(metrics: Dict[str, Any], stock_news: List[Dict[str, str]]) -> Dict[str, Any]:
    score = 50.0
    rsi = float(metrics["rsi_14"])
    macd_hist = float(metrics["macd_histogram"])
    vol_spike = float(metrics["volume_vs_20d_avg"])
    price_vs_ma = float(metrics["price_vs_ma20_pct"])
    intraday_pct = float(metrics["intraday_price_change_pct"])
    high_volume = vol_spike >= 1.5
    volume_price_score = 50.0
    volume_interpretation = "Balanced volume-price behavior."
    distribution_warning = False

    if high_volume and intraday_pct > 0.5:
        volume_price_score = min(100.0, 90.0 + min(10.0, intraday_pct * 4.0))
        volume_interpretation = "Accumulation confirmed (high volume with strong upside)."
    elif high_volume and 0.0 <= intraday_pct <= 0.5:
        volume_price_score = min(70.0, 60.0 + (intraday_pct * 20.0))
        volume_interpretation = "High volume but weak upside; conviction is mixed."
    elif high_volume and -0.2 <= intraday_pct <= 0.2:
        volume_price_score = 30.0
        volume_interpretation = "Distribution warning: heavy volume with flat price."
    elif high_volume and intraday_pct < 0.0:
        volume_price_score = max(0.0, 20.0 + (intraday_pct * 10.0))
        volume_interpretation = (
            "Warning: high volume with price decline suggests distribution."
        )
        distribution_warning = True
    elif (not high_volume) and intraday_pct > 0.0:
        volume_price_score = 50.0
        volume_interpretation = "Price up on low volume; weak conviction."
    else:
        volume_price_score = 45.0
        volume_interpretation = "Low volume move; confirmation is limited."

    # Prefer trending but not too overbought conditions.
    if 45 <= rsi <= 65:
        score += 10.0
    elif rsi > 75 or rsi < 25:
        score -= 10.0

    score += 10.0 if macd_hist > 0 else -10.0
    # Volume is never isolated: use combined volume + intraday price action.
    score += (volume_price_score - 50.0) * 0.9
    score += max(-12.0, min(12.0, price_vs_ma * 1.3))
    score += _news_sentiment_score(stock_news) * 1.5
    if distribution_warning:
        score -= 20.0

    score = _clamp_score(score)
    volume_story = (
        f"{vol_spike:.1f}x average volume; price {'up' if intraday_pct >= 0 else 'down'} "
        f"{abs(intraday_pct):.2f}% intraday (vs open). {volume_interpretation}"
    )
    return {
        "score": round(score, 1),
        "sentiment": _score_to_sentiment(score),
        "inputs": {
            "rsi_14": metrics["rsi_14"],
            "macd_histogram": metrics["macd_histogram"],
            "volume_vs_20d_avg": metrics["volume_vs_20d_avg"],
            "intraday_price_change_pct": metrics["intraday_price_change_pct"],
            "volume_price_action_score": round(volume_price_score, 1),
            "high_volume_with_price_decline": distribution_warning,
            "volume_story": volume_story,
            "price_vs_ma20_pct": metrics["price_vs_ma20_pct"],
            "stock_news_count": len(stock_news),
        },
    }


def _compute_sector_layer(stock_ticker: str, india_layer: Dict[str, Any]) -> Dict[str, Any]:
    mapping = SECTOR_PEERS.get(stock_ticker, {"sector": "Unknown", "peers": []})
    sector = mapping["sector"]
    peers: List[str] = mapping["peers"][:3]
    nifty_today = float(india_layer["inputs"]["nifty_change_pct"])

    peer_changes: List[Tuple[str, float]] = []
    for peer in peers:
        ch = _safe_pct_change(peer)
        if ch is not None:
            peer_changes.append((peer, ch))

    if peer_changes:
        avg_peer = sum(v for _, v in peer_changes) / len(peer_changes)
    else:
        avg_peer = 0.0

    relative = avg_peer - nifty_today
    score = _clamp_score(50.0 + (relative * 10.0))

    return {
        "score": round(score, 1),
        "sentiment": _score_to_sentiment(score),
        "inputs": {
            "sector": sector,
            "peers": [p for p, _ in peer_changes] or peers,
            "peer_avg_change_pct": round(avg_peer, 2),
            "nifty_change_pct": round(nifty_today, 2),
            "relative_vs_nifty_pct": round(relative, 2),
        },
    }


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
    open_price, current_price, intraday_price_change_pct = _intraday_price_change_pct(symbol)

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
        "volume_vs_20d_avg": round(vol_spike_ratio, 3),
        "open_price": round(open_price, 2),
        "current_price": round(current_price, 2),
        "intraday_price_change_pct": round(intraday_price_change_pct, 2),
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
    ordered_tickers: List[str], global_layer: Dict[str, Any], india_layer: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], str]:
    rows: List[Dict[str, Any]] = []
    summary_lines: List[str] = []
    for sym in ordered_tickers:
        try:
            m = _compute_stock_metrics(sym)
            news = fetch_news_headlines(m["ticker"])
            m["news"] = news
            m["layer4_stock_specific"] = _compute_stock_specific_layer(m, news)
            m["layer3_sector_momentum"] = _compute_sector_layer(m["ticker"], india_layer)
            rows.append(m)
            n_txt = "; ".join(n["title"] for n in news[:3]) if news else "No headlines today."
            summary_lines.append(
                f"- {m['ticker']}: close={m['last_close']}, RSI={m['rsi_14']}, "
                f"MACD={m['macd']}, MACD_signal={m['macd_signal']}, hist={m['macd_histogram']}, "
                f"intraday_change_vs_open={m['intraday_price_change_pct']}%, "
                f"vol_x20d={m['volume_vs_20d_avg']}, "
                f"price_vs_MA20={m['price_vs_ma20_pct']}%, "
                f"L3_sector_score={m['layer3_sector_momentum']['score']}, "
                f"L4_stock_score={m['layer4_stock_specific']['score']}, "
                f"VolumeStory={m['layer4_stock_specific']['inputs']['volume_story']}. News: {n_txt}"
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


def _claude_pick_signal(
    analysis_summary: str, raw_rows_json: str, global_layer: Dict[str, Any], india_layer: Dict[str, Any]
) -> Dict[str, Any]:
    api_key = _get_env("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""You are an expert Indian equity intraday trader focused on NSE.

Use this 4-layer confidence framework (equal 25% each):
- Layer 1 Global Macro
- Layer 2 India Market Sentiment
- Layer 3 Sector Momentum
- Layer 4 Stock Specific Signals

{analysis_summary}

Full structured rows (JSON array for your reference, do not echo verbatim):
{raw_rows_json}

Layer 1 (Global Macro) precomputed:
{json.dumps(global_layer, default=str)}

Layer 2 (India Sentiment) precomputed:
{json.dumps(india_layer, default=str)}

Task:
1) Pick exactly ONE stock that has the highest realistic probability of a favorable intraday move TODAY for an intraday trade (not investment advice; probabilistic).
2) Output strict JSON only (no markdown) with these keys:
   - ticker: NSE symbol WITHOUT .NS (e.g. RELIANCE)
   - direction: either BUY or SELL
   - entry_price: number (INR, 2 decimals)
   - target_price: number (INR, 2 decimals)
   - stop_loss: number (INR, 2 decimals)
   - confidence_score: number from 0 to 100 (from weighted 4-layer framework before caps)
   - reasoning: concise string mentioning all 4 layers and a clear "Volume Story" line (max ~800 chars)
   - layer_breakdown: object with keys layer1_global_macro, layer2_india_sentiment, layer3_sector_momentum, layer4_stock_specific; each has score (0-100), sentiment (bullish/neutral/bearish), summary
     and layer4_stock_specific must include inputs with:
       volume_vs_20d_avg, intraday_price_change_pct, high_volume_with_price_decline, volume_story

Rules:
- For BUY: target_price > entry_price > stop_loss
- For SELL: stop_loss > entry_price > target_price
- Use realistic NSE-style levels based on the last_close in the data.
- Volume must never be treated in isolation; combine volume with intraday move (current vs open).
- If high volume and intraday price decline for selected stock, include warning:
  "⚠️ High volume with price decline - possible distribution, avoid BUY"
- SELL logic must be reversed: high volume + strong intraday downside supports SELL;
  high volume + intraday upside should reduce SELL conviction.
- Confidence formula: weighted average, 25% each.
- Cap rule A: if any single layer sentiment is bearish strongly (or score <= 35), final confidence must be capped at 50.
- Cap rule B: if both global macro and India sentiment are bearish strongly (or score <= 35), final confidence must be capped at 35.
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
        "layer_breakdown",
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
    layers = s.get("layer_breakdown", {})
    needed_layers = [
        "layer1_global_macro",
        "layer2_india_sentiment",
        "layer3_sector_momentum",
        "layer4_stock_specific",
    ]
    for lk in needed_layers:
        if lk not in layers:
            raise ValueError(f"Missing layer breakdown key: {lk}")
        if "score" not in layers[lk]:
            raise ValueError(f"Missing score in {lk}")
    if "inputs" not in layers["layer4_stock_specific"]:
        raise ValueError("Missing inputs in layer4_stock_specific")

    l1 = float(layers["layer1_global_macro"]["score"])
    l2 = float(layers["layer2_india_sentiment"]["score"])
    l3 = float(layers["layer3_sector_momentum"]["score"])
    l4 = float(layers["layer4_stock_specific"]["score"])
    l4_inputs = layers["layer4_stock_specific"].get("inputs", {})
    hv_down = bool(l4_inputs.get("high_volume_with_price_decline", False))
    intraday_pct = float(l4_inputs.get("intraday_price_change_pct", 0.0) or 0.0)
    vol_x = float(l4_inputs.get("volume_vs_20d_avg", 1.0) or 1.0)

    weighted_conf = (
        (l1 * LAYER_WEIGHT)
        + (l2 * LAYER_WEIGHT)
        + (l3 * LAYER_WEIGHT)
        + (l4 * LAYER_WEIGHT)
    ) / 100.0
    conf = _clamp_score(weighted_conf)

    strong_bearish = [lk for lk in needed_layers if float(layers[lk]["score"]) <= 35.0]
    if strong_bearish:
        conf = min(conf, 50.0)
    if l1 <= 35.0 and l2 <= 35.0:
        conf = min(conf, 35.0)
    if direction == "BUY" and hv_down:
        conf = min(conf, 45.0)
    if direction == "SELL" and vol_x >= 1.5 and intraday_pct > 0:
        # Reverse logic for SELL: avoid shorting high-volume upside.
        conf = min(conf, 35.0)

    if direction == "BUY" and not (target > entry > stop):
        raise ValueError("BUY requires target > entry > stop_loss")
    if direction == "SELL" and not (stop > entry > target):
        raise ValueError("SELL requires stop_loss > entry > target")

    reason = str(s["reasoning"])[:2000]
    for token in ("Layer 1", "Layer 2", "Layer 3", "Layer 4"):
        if token.lower() not in reason.lower():
            reason = (
                "Layer 1 global macro, Layer 2 India sentiment, Layer 3 sector momentum, "
                "Layer 4 stock-specific signals are jointly considered. "
            ) + reason
            break
    volume_story = str(l4_inputs.get("volume_story", "")).strip()
    if volume_story and "volume story" not in reason.lower():
        reason = f"{reason} Volume Story: {volume_story}"
    if direction == "BUY" and hv_down and "possible distribution" not in reason.lower():
        reason = (
            f"{reason} ⚠️ High volume with price decline - possible distribution, avoid BUY"
        )

    return {
        "ticker": str(s["ticker"]).upper().replace(".NS", ""),
        "direction": direction,
        "entry_price": round(entry, 2),
        "target_price": round(target, 2),
        "stop_loss": round(stop, 2),
        "confidence_score": round(conf, 1),
        "reasoning": reason[:2000],
        "layer_breakdown": {
            lk: {
                "score": round(float(layers[lk]["score"]), 1),
                "sentiment": str(layers[lk].get("sentiment", "")).lower(),
                "summary": str(layers[lk].get("summary", ""))[:400],
                "inputs": layers[lk].get("inputs", {}),
            }
            for lk in needed_layers
        },
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
    global_layer = _fetch_global_macro_layer()
    india_layer = _fetch_india_market_layer()
    rows, summary = _build_analysis_payload(ordered, global_layer, india_layer)
    raw_json = json.dumps(rows, default=str)
    claude_raw = _claude_pick_signal(summary, raw_json, global_layer, india_layer)
    validated = _validate_signal(claude_raw)
    signal_date = datetime.now(IST).date()
    return save_signal_to_supabase(validated, signal_date)


if __name__ == "__main__":
    out = run_analysis()
    print(json.dumps(out, indent=2))
