import os
import re
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# ------------ CONFIG ------------

TP_WEBHOOK_URL = os.getenv("TP_WEBHOOK_URL", "")
TP_DEFAULT_QTY = int(os.getenv("TP_DEFAULT_QTY", "1"))  # ENTRY SIZE (e.g., 2)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Latest EMA state per ticker
EMA_STATE: Dict[str, Dict[str, Any]] = {}

# Last trade / events (optional debugging)
LAST_TRADES: Dict[str, Dict[str, Any]] = {}

# Simple position state per ticker
POSITION_STATE: Dict[str, Dict[str, Any]] = {}

# Titan “New Trade Design” waiting for next EMA update
PENDING_TRADES: Dict[str, Dict[str, Any]] = {}


# ------------ HELPERS ------------

def utc_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def send_to_traderspost(payload: dict) -> dict:
    if not TP_WEBHOOK_URL:
        logger.error("TP_WEBHOOK_URL not set")
        return {"ok": False, "error": "TP_WEBHOOK_URL not set"}

    try:
        logger.info("Sending to TradersPost: %s", payload)
        resp = requests.post(TP_WEBHOOK_URL, json=payload, timeout=5)
        return {"ok": resp.ok, "status_code": resp.status_code, "body": resp.text}
    except Exception as e:
        logger.exception("Error sending to TradersPost: %s", e)
        return {"ok": False, "error": str(e)}


def update_ema_state_from_json(data: dict) -> Optional[str]:
    ticker = data.get("ticker")
    if not ticker:
        logger.warning("ema_update without ticker: %s", data)
        return None

    above13_raw = str(data.get("above13", data.get("above", ""))).lower()
    above13 = above13_raw in ("true", "1", "yes")

    try:
        ema13 = float(data.get("ema13", 0.0))
    except Exception:
        ema13 = 0.0

    try:
        close = float(data.get("close", 0.0))
    except Exception:
        close = 0.0

    EMA_STATE[ticker] = {
        "above13": above13,
        "ema13": ema13,
        "close": close,
        "time": data.get("time", ""),
        "received_at": utc_ts(),
    }

    logger.info("Updated EMA state for %s: %s", ticker, EMA_STATE[ticker])
    return ticker


# ------------ PARSERS ------------

TITAN_RE = re.compile(
    r"(?P<ticker>[A-Z0-9_]+)\s+New Trade Design\s*,\s*Price\s*=\s*(?P<price>[0-9.]+)"
)

EXIT_RE = re.compile(
    r"(?P<ticker>[A-Z0-9_]+)\s+Exit Signal\s*,?\s*Price\s*=\s*(?P<price>[0-9.]+)"
)


def parse_titan_new_trade(text: str) -> Dict[str, Any]:
    m = TITAN_RE.search(text)
    return {"ticker": m.group("ticker"), "price": float(m.group("price"))} if m else {}


def parse_exit_signal(text: str) -> Dict[str, Any]:
    m = EXIT_RE.search(text)
    return {"ticker": m.group("ticker"), "price": float(m.group("price"))} if m else {}


# ------------ TRADE HANDLING ------------

def handle_new_trade_for_ticker(ticker: str, price: Optional[float], time_str: Optional[str]):
    """
    ALWAYS determine direction from EMA:
        above13=True → buy
        above13=False → sell

    ENTRY:
        - Always enter qty=TP_DEFAULT_QTY (e.g., 2)

    EXIT:
        - If already in a trade:
              exit FULL remaining qty
              THEN re-enter fresh qty=TP_DEFAULT_QTY in EMA direction
    """

    ema_info = EMA_STATE.get(ticker)
    if not ema_info:
        logger.warning("No EMA state for %s; skipping", ticker)
        return {"ok": True, "skipped": "no_ema_state"}

    direction = "buy" if ema_info.get("above13", False) else "sell"
    entry_qty = TP_DEFAULT_QTY if TP_DEFAULT_QTY > 0 else 1

    pos = POSITION_STATE.get(ticker, {})
    current_qty = int(pos.get("qty", 0))
    is_open = bool(pos.get("open", False))

    # ---- If already in a position → FULL EXIT + fresh entry ----
    if is_open and current_qty > 0:
        logger.info(
            "%s already in position (%s %d). Exiting full qty and re-entering %d in EMA direction (%s).",
            ticker, pos.get("direction"), current_qty, entry_qty, direction
        )

        # FULL EXIT
        exit_payload = {"ticker": ticker, "action": "exit", "quantity": current_qty}
        if price is not None:
            exit_payload["price"] = price
        exit_result = send_to_traderspost(exit_payload)

        # FRESH ENTRY
        entry_payload = {"ticker": ticker, "action": direction, "quantity": entry_qty}
        if price is not None:
            entry_payload["price"] = price
        entry_result = send_to_traderspost(entry_payload)

        POSITION_STATE[ticker] = {
            "open": True,
            "direction": direction,
            "qty": entry_qty,
            "opened_time": time_str,
            "price": price,
        }

        LAST_TRADES[ticker] = {
            "event": "full_exit_and_reenter",
            "prev_qty": current_qty,
            "new_qty": entry_qty,
            "price": price,
            "direction": direction,
            "exit_result": exit_result,
            "entry_result": entry_result,
            "ema_snapshot": ema_info,
        }

        ok = exit_result.get("ok", False) and entry_result.get("ok", False)
        return {"ok": ok, "event": "full_exit_and_reenter"}

    # ---- Fresh Entry ----
    payload = {"ticker": ticker, "action": direction, "quantity": entry_qty}
    if price is not None:
        payload["price"] = price

    result = send_to_traderspost(payload)

    POSITION_STATE[ticker] = {
        "open": True,
        "direction": direction,
        "qty": entry_qty,
        "opened_time": time_str,
        "price": price,
    }

    LAST_TRADES[ticker] = {
        "event": "new_trade",
        "direction": direction,
        "qty": entry_qty,
        "price": price,
        "ema_snapshot": ema_info,
        "time": time_str,
        "tp_result": result,
    }

    return {"ok": result.get("ok", False), "event": "new_trade"}


def handle_exit_for_ticker(ticker: str, price: Optional[float], time_str: Optional[str]):
    """
    Exit 1 contract per exit signal.
    """

    pos = POSITION_STATE.get(ticker, {})
    current_qty = int(pos.get("qty", 0))
    is_open = bool(pos.get("open", False))

    exit_qty = 1

    logger.info(
        "Exit signal for %s: exiting 1 contract. Before exit qty=%d.", ticker, current_qty
    )

    payload = {"ticker": ticker, "action": "exit", "quantity": exit_qty}
    if price is not None:
        payload["price"] = price
    result = send_to_traderspost(payload)

    new_qty = max(current_qty - exit_qty, 0)
    new_open = new_qty > 0

    POSITION_STATE[ticker] = {
        "open": new_open,
        "direction": pos.get("direction") if new_open else None,
        "qty": new_qty,
        "closed_time": time_str if not new_open else None,
        "price": price,
    }

    LAST_TRADES[ticker] = {
        "event": "partial_exit" if new_open else "final_exit",
        "qty_before": current_qty,
        "qty_after": new_qty,
        "exit_qty": exit_qty,
        "price": price,
        "time": time_str,
        "tp_result": result,
    }

    return {"ok": result.get("ok", False), "event": LAST_TRADES[ticker]["event"]}


# ------------ ROUTES ------------

@app.route("/webhook", methods=["POST"])
def webhook():
    raw_body = request.get_data(as_text=True) or ""
    logger.info("Incoming body: %r", raw_body)

    # 1) Try JSON (EMA updates)
    data = None
    try:
        data = json.loads(raw_body)
    except:
        data = None

    if isinstance(data, dict) and data:
        if data.get("type") == "ema_update":
            ticker = update_ema_state_from_json(data)

            if ticker and ticker in PENDING_TRADES:
                pending = PENDING_TRADES.pop(ticker)
                result = handle_new_trade_for_ticker(
                    ticker, pending.get("price"), pending.get("time")
                )
                return jsonify(result), 200

            return jsonify({"ok": True, "event": "ema_update_only"}), 200

        return jsonify({"ok": False, "error": f"unknown json type {data.get('type')}"}), 400

    # 2) Titan New Trade Design
    titan = parse_titan_new_trade(raw_body)
    if titan:
        ticker = titan["ticker"]
        price = titan["price"]
        now = utc_ts()

        state = EMA_STATE.get(ticker)
        fresh = False
        if state and "received_at" in state:
            if now - state["received_at"] <= 5:
                fresh = True

        if fresh:
            result = handle_new_trade_for_ticker(ticker, price, None)
            return jsonify(result), 200

        PENDING_TRADES[ticker] = {"price": price, "time": None, "created_at": now}
        return jsonify({"ok": True, "event": "pending_trade_stored"}), 200

    # 3) Exit Signal
    exit_info = parse_exit_signal(raw_body)
    if exit_info:
        ticker = exit_info["ticker"]
        price = exit_info["price"]
        result = handle_exit_for_ticker(ticker, price, None)
        return jsonify(result), 200

    return jsonify({"ok": False, "error": "unrecognized payload"}), 400


@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "message": "Titan Bot webhook running"})


@app.route("/dashboard", methods=["GET"])
def dashboard():
    return jsonify(
        {
            "ema_state": EMA_STATE,
            "positions": POSITION_STATE,
            "pending_trades": PENDING_TRADES,
            "last_trades": LAST_TRADES,
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
