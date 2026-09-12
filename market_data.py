# ============================================================
# Multi-provider market data: Yahoo -> Polygon -> Finnhub
# NOTE: curl_cffi removed — it segfaults on Python 3.14
# ============================================================
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

import httpx
import yfinance as yf
from fastapi import HTTPException

logger = logging.getLogger(__name__)

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

TIMEFRAME_MAP = {
    "1D": {"yf": "1d",  "poly": ("5",  "minute", 1),   "finn": ("5", 1)},
    "5D": {"yf": "5d",  "poly": ("30", "minute", 5),   "finn": ("30", 5)},
    "1M": {"yf": "1mo", "poly": ("1",  "day",    30),  "finn": ("D", 30)},
    "3M": {"yf": "3mo", "poly": ("1",  "day",    90),  "finn": ("D", 90)},
    "6M": {"yf": "6mo", "poly": ("1",  "day",    180), "finn": ("D", 180)},
    "1Y": {"yf": "1y",  "poly": ("1",  "day",    365), "finn": ("D", 365)},
}


def _shape_result(rows: List[Dict[str, Any]], source: str) -> Dict[str, Any]:
    if not rows:
        raise ValueError("empty result set")
    closes = [r["price"] for r in rows]
    current = closes[-1]
    first = closes[0] if closes else 1
    change_pct = round(((current - first) / first) * 100, 2) if first else 0.0
    return {
        "current_price": current,
        "change_pct": change_pct,
        "data": rows,
        "closes": closes,
        "source": source,
    }


# ---------- Provider 1: Yahoo (plain yfinance) ----------
def _fetch_yahoo(symbol: str, timeframe: str) -> Dict[str, Any]:
    period = TIMEFRAME_MAP[timeframe]["yf"]
    df = yf.Ticker(symbol).history(period=period)
    if df is None or df.empty:
        raise ValueError("Yahoo returned no data")

    rows = []
    for d, r in df.iterrows():
        rows.append({
            "date": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d),
            "price": round(float(r["Close"]), 2),
        })
    return _shape_result(rows, "yahoo")


# ---------- Provider 2: Polygon ----------
def _fetch_polygon(symbol: str, timeframe: str) -> Dict[str, Any]:
    if not POLYGON_API_KEY:
        raise ValueError("POLYGON_API_KEY not set")

    mult, timespan, days = TIMEFRAME_MAP[timeframe]["poly"]
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/"
        f"{mult}/{timespan}/{start}/{end}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
    )
    r = httpx.get(url, timeout=10.0)
    r.raise_for_status()
    payload = r.json()

    results = payload.get("results") or []
    if not results:
        raise ValueError(f"Polygon returned no results (status={payload.get('status')})")

    rows = []
    for bar in results:
        ts = bar.get("t")
        if ts is None:
            continue
        d = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        rows.append({"date": d, "price": round(float(bar["c"]), 2)})
    return _shape_result(rows, "polygon")


# ---------- Provider 3: Finnhub ----------
def _fetch_finnhub(symbol: str, timeframe: str) -> Dict[str, Any]:
    if not FINNHUB_API_KEY:
        raise ValueError("FINNHUB_API_KEY not set")

    resolution, days = TIMEFRAME_MAP[timeframe]["finn"]
    days = min(days, 365)
    now = int(datetime.now(timezone.utc).timestamp())
    start = now - days * 86400

    url = (
        f"https://finnhub.io/api/v1/stock/candle"
        f"?symbol={symbol}&resolution={resolution}"
        f"&from={start}&to={now}&token={FINNHUB_API_KEY}"
    )
    r = httpx.get(url, timeout=10.0)
    r.raise_for_status()
    payload = r.json()

    if payload.get("s") != "ok":
        raise ValueError(f"Finnhub status={payload.get('s')}")

    rows = []
    for ts, close in zip(payload["t"], payload["c"]):
        d = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        rows.append({"date": d, "price": round(float(close), 2)})
    return _shape_result(rows, "finnhub")


# ---------- Orchestrator: try Yahoo, then Polygon, then Finnhub ----------
def fetch_asset_data(symbol: str, timeframe: str = "3M") -> Dict[str, Any]:
    if timeframe not in TIMEFRAME_MAP:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe: {timeframe}")

    symbol = symbol.upper()
    errors = []

    try:
        result = _fetch_yahoo(symbol, timeframe)
        logger.info(f"[yahoo] OK {symbol} {timeframe} -> {len(result['data'])} pts")
        return result
    except Exception as e:
        logger.warning(f"[yahoo] FAIL {symbol}: {e}")
        errors.append(f"yahoo: {e}")

    try:
        result = _fetch_polygon(symbol, timeframe)
        logger.info(f"[polygon] OK {symbol} {timeframe} -> {len(result['data'])} pts")
        return result
    except Exception as e:
        logger.warning(f"[polygon] FAIL {symbol}: {e}")
        errors.append(f"polygon: {e}")

    try:
        result = _fetch_finnhub(symbol, timeframe)
        logger.info(f"[finnhub] OK {symbol} {timeframe} -> {len(result['data'])} pts")
        return result
    except Exception as e:
        logger.warning(f"[finnhub] FAIL {symbol}: {e}")
        errors.append(f"finnhub: {e}")

    raise HTTPException(
        status_code=502,
        detail=f"All market data providers failed for {symbol}. " + " | ".join(errors),
    )
