from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv
from flask import jsonify, request
from kiteconnect import KiteConnect

load_dotenv()

TOKEN_FILE = Path("access_token.txt")


def _get_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def create_kite_client() -> KiteConnect:
    return KiteConnect(api_key=_get_env("KITE_API_KEY"))


def get_login_url() -> str:
    client = create_kite_client()
    url = client.login_url()
    return url


def print_login_url() -> None:
    print(f"Kite Login URL: {get_login_url()}")


def exchange_request_token(request_token: str) -> Dict[str, Any]:
    client = create_kite_client()
    data = client.generate_session(
        request_token=request_token,
        api_secret=_get_env("KITE_API_SECRET"),
    )
    access_token = data.get("access_token")
    if not access_token:
        raise RuntimeError("Kite did not return access_token.")
    TOKEN_FILE.write_text(access_token, encoding="utf-8")
    print("Authentication successful")
    return data


def register_callback_route(app) -> None:
    @app.get("/callback")
    def kite_callback():  # type: ignore[no-redef]
        try:
            request_token = (request.args.get("request_token") or "").strip()
            if not request_token:
                return jsonify({"error": "Missing request_token in callback URL."}), 400
            session_data = exchange_request_token(request_token)
            return jsonify(
                {
                    "message": "Authentication successful",
                    "user_name": session_data.get("user_name"),
                    "access_token_saved_to": str(TOKEN_FILE.resolve()),
                }
            )
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    from flask import Flask
    app = Flask(__name__)
    register_callback_route(app)
    print_login_url()
    app.run(port=5000, debug=False)
