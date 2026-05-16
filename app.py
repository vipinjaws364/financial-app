from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, request, send_from_directory
from supabase import create_client

from kite_auth import exchange_request_token, get_login_url
from nifty_signal import generate_nifty_weekly_signal
from option_chain import get_option_chain
from signal_engine_v2 import run_signal_generation

load_dotenv()

IST = ZoneInfo("Asia/Kolkata")

app = Flask(__name__, static_folder=".", static_url_path="")


def _get_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v.strip()


def _supabase():
    return create_client(_get_env("SUPABASE_URL"), _get_env("SUPABASE_KEY"))


def _parse_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    if isinstance(value, datetime):
        return value.date()
    raise ValueError(f"Bad date: {value!r}")


@app.route("/")
def index() -> Any:
    return send_from_directory(".", "index.html")


@app.get("/auth")
def kite_auth() -> Any:
    try:
        return redirect(get_login_url())
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.get("/callback")
def kite_callback() -> Any:
    try:
        request_token = (request.args.get("request_token") or "").strip()
        if not request_token:
            return jsonify({"error": "Missing request_token in callback URL."}), 400
        session_data = exchange_request_token(request_token)
        return jsonify(
            {
                "message": "Authentication successful",
                "user_name": session_data.get("user_name"),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.post("/generate")
def generate() -> Any:
    try:
        return jsonify(run_signal_generation())
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.post("/nifty-signal")
def nifty_signal() -> Any:
    try:
        return jsonify(generate_nifty_weekly_signal())
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.get("/option-chain")
def option_chain() -> Any:
    try:
        symbol = (request.args.get("symbol") or "").strip().upper()
        expiry = (request.args.get("expiry") or "").strip()
        if not symbol or not expiry:
            return jsonify({"error": "Provide query params: symbol, expiry(YYYY-MM-DD)."}), 400
        data = get_option_chain(symbol, expiry)
        return jsonify(data)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.get("/signals")
def signals() -> Any:
    try:
        sb = _supabase()
        cutoff = (datetime.now(IST).date() - timedelta(days=7)).isoformat()
        res = sb.table("daily_signals").select("*").gte("signal_date", cutoff).execute()
        rows: List[Dict[str, Any]] = list(res.data or [])
        rows.sort(
            key=lambda r: (
                _parse_date(r.get("signal_date")),
                str(r.get("created_at") or ""),
            ),
            reverse=True,
        )
        today = datetime.now(IST).date()
        today_rows = [r for r in rows if _parse_date(r.get("signal_date")) == today]
        today_signal: Optional[Dict[str, Any]] = None
        if today_rows:
            today_signal = sorted(
                today_rows,
                key=lambda r: str(r.get("created_at") or r.get("id")),
                reverse=True,
            )[0]
        wins = sum(1 for r in rows if r.get("result") == "WIN")
        losses = sum(1 for r in rows if r.get("result") == "LOSS")
        resolved = wins + losses
        win_rate_pct = round(100.0 * wins / resolved, 1) if resolved > 0 else None

        return jsonify(
            {
                "signals": rows,
                "today_signal": today_signal,
                "win_rate_pct": win_rate_pct,
                "stats": {"wins": wins, "losses": losses, "resolved": resolved},
            }
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.post("/update-acted")
def update_acted() -> Any:
    try:
        payload = request.get_json(force=True, silent=True) or {}
        sid = payload.get("id")
        acted = payload.get("acted")
        if sid is None or not isinstance(acted, bool):
            return jsonify({"error": "Expected JSON body with 'id' and boolean 'acted'."}), 400
        sb = _supabase()
        sb.table("daily_signals").update({"acted": acted}).eq("id", sid).execute()
        return jsonify({"ok": True, "id": sid, "acted": acted})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", "8080")))
