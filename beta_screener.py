from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

TOP_BETA_STOCKS = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "INFY",
    "ICICIBANK",
    "AXISBANK",
    "KOTAKBANK",
    "SBIN",
    "BAJFINANCE",
    "MARUTI",
    "TATAMOTORS",
    "ADANIENT",
    "WIPRO",
    "SUNPHARMA",
    "TITAN",
    "HINDUNILVR",
    "LT",
    "ULTRACEMCO",
    "ASIANPAINT",
    "BHARTIARTL",
]


def _safe_symbol(symbol: str) -> str:
    return symbol if symbol.endswith(".NS") else f"{symbol}.NS"


def calculate_beta(symbol: str) -> float:
    """
    30-day beta vs Nifty 50 using daily returns.
    """
    try:
        ticker = _safe_symbol(symbol.upper())
        s = yf.Ticker(ticker).history(period="90d", interval="1d")
        b = yf.Ticker("^NSEI").history(period="90d", interval="1d")
        if s.empty or b.empty:
            raise RuntimeError("No price history.")

        s_ret = s["Close"].pct_change().dropna().tail(30)
        b_ret = b["Close"].pct_change().dropna().tail(30)
        joined = pd.concat([s_ret, b_ret], axis=1, join="inner").dropna()
        if len(joined) < 15:
            raise RuntimeError("Insufficient overlap for beta.")

        cov = np.cov(joined.iloc[:, 0], joined.iloc[:, 1], ddof=1)[0][1]
        var = np.var(joined.iloc[:, 1], ddof=1)
        if var == 0:
            return 1.0
        return float(cov / var)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Beta calc failed for {symbol}: {exc}") from exc


def _today_momentum(symbol: str) -> Dict[str, float]:
    """
    Momentum score uses today's % change * (today volume / avg volume 20d).
    """
    t = yf.Ticker(_safe_symbol(symbol))
    hist = t.history(period="3mo", interval="1d")
    if hist.empty or len(hist) < 25:
        raise RuntimeError(f"Insufficient data for momentum: {symbol}")

    today = hist.iloc[-1]
    prev_close = float(hist["Close"].iloc[-2])
    close = float(today["Close"])
    vol = float(today["Volume"])
    avg_vol = float(hist["Volume"].tail(20).mean())
    pct = ((close - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0
    vol_ratio = (vol / avg_vol) if avg_vol > 0 else 1.0
    momentum_score = pct * vol_ratio
    return {
        "price_change_pct": round(pct, 2),
        "volume_ratio_20d": round(vol_ratio, 3),
        "momentum_score": round(momentum_score, 3),
    }


def get_top_momentum_stocks() -> List[Dict[str, Any]]:
    """
    Screens 20 high-beta names and returns top 5 using combined
    beta rank and today's momentum rank.
    """
    rows: List[Dict[str, Any]] = []
    for sym in TOP_BETA_STOCKS:
        try:
            beta = calculate_beta(sym)
            mom = _today_momentum(sym)
            rows.append({"ticker": sym, "beta_30d": round(beta, 3), **mom})
        except Exception:
            continue

    if not rows:
        raise RuntimeError("No stocks passed beta/momentum screening.")

    # Rank-based blend: beta rank (descending) + momentum rank (descending)
    beta_sorted = sorted(rows, key=lambda x: x["beta_30d"], reverse=True)
    mom_sorted = sorted(rows, key=lambda x: x["momentum_score"], reverse=True)
    beta_rank = {r["ticker"]: i + 1 for i, r in enumerate(beta_sorted)}
    mom_rank = {r["ticker"]: i + 1 for i, r in enumerate(mom_sorted)}

    for r in rows:
        br = beta_rank[r["ticker"]]
        mr = mom_rank[r["ticker"]]
        r["rank_score"] = round((1 / br) * 0.5 + (1 / mr) * 0.5, 6)

    top5 = sorted(rows, key=lambda x: x["rank_score"], reverse=True)[:5]
    return top5
