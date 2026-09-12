# ============================================================
# Stock Dashboard Backend — FastAPI
# ============================================================

# ---------------- Imports ----------------
import os
from dotenv import load_dotenv
load_dotenv()

import logging
import traceback
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
import pandas as pd
import jwt
import bcrypt

from market_data import fetch_asset_data, TIMEFRAME_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- App ----------------
app = FastAPI(title="Stock Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.vercel.app",
        "https://*.onrender.com",
    ],
    allow_origin_regex=r"https://.*\.(vercel\.app|onrender\.com|run\.app)",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- API Keys / Config ----------------
JWT_SECRET = os.environ.get("JWT_SECRET", "change-this-in-production")
JWT_EXP_HOURS = 24
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MONGODB_URI = os.environ.get("MONGODB_URI")
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY")

# ---------------- MongoDB Setup ----------------
mongo_db = None
users_col = None
portfolio_col = None
alerts_col = None
shares_col = None

try:
    if MONGODB_URI:
        from pymongo import MongoClient
        _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        _client.admin.command("ping")
        mongo_db = _client["stock_dashboard"]
        users_col = mongo_db["users"]
        portfolio_col = mongo_db["portfolio"]
        alerts_col = mongo_db["alerts"]
        shares_col = mongo_db["shares"]
        users_col.create_index("email", unique=True)
        logger.info("MongoDB connected")
    else:
        logger.warning("MONGODB_URI not set — DB routes disabled")
except Exception as e:
    logger.warning(f"MongoDB unavailable: {e}")


def db_required():
    if mongo_db is None:
        raise HTTPException(status_code=503, detail="Database not connected")


# ---------------- Auth helpers ----------------
def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), hashed.encode())
    except Exception:
        return False


def make_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXP_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1]
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    if users_col is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    user = users_col.find_one({"email": data["email"]})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    user["_id"] = str(user["_id"])
    return user


# ---------------- Technical Indicators ----------------
def calculate_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if not closes or len(closes) < period:
        return None
    s = pd.Series(closes, dtype=float)
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    ll = float(loss.iloc[-1])
    if ll == 0:
        return 100.0
    rs = float(gain.iloc[-1]) / ll
    return round(100 - 100 / (1 + rs), 2)


def calculate_macd(closes: List[float]) -> Dict[str, Optional[float]]:
    if not closes or len(closes) < 26:
        return {"line": None, "signal": None, "histogram": None}
    s = pd.Series(closes, dtype=float)
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    return {
        "line": round(float(macd_line.iloc[-1]), 4),
        "signal": round(float(signal_line.iloc[-1]), 4),
        "histogram": round(float(hist.iloc[-1]), 4),
    }


def calculate_bollinger(closes: List[float], period: int = 20) -> Dict[str, Optional[float]]:
    if not closes or len(closes) < period:
        v = closes[-1] if closes else None
        return {"upper": v, "mid": v, "lower": v}
    s = pd.Series(closes, dtype=float)
    mid = s.rolling(period).mean()
    sd = s.rolling(period).std()
    return {
        "upper": round(float((mid + 2 * sd).iloc[-1]), 2),
        "mid": round(float(mid.iloc[-1]), 2),
        "lower": round(float((mid - 2 * sd).iloc[-1]), 2),
    }


# ---------------- AI Providers ----------------
def _ai_gemini(prompt: str) -> Optional[str]:
    """Gemini via the modern google-genai SDK."""
    if not GEMINI_API_KEY:
        return None
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    resp = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )
    text = getattr(resp, "text", None)
    return text.strip() if text and text.strip() else None


def _ai_claude(prompt: str) -> Optional[str]:
    """Claude via Anthropic REST API (no SDK needed)."""
    if not ANTHROPIC_API_KEY:
        return None
    import httpx
    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=15.0,
    )
    r.raise_for_status()
    data = r.json()
    for block in data.get("content", []):
        if block.get("type") == "text" and block.get("text"):
            return block["text"].strip()
    return None


def _ai_groq(prompt: str) -> Optional[str]:
    """Groq via OpenAI-compatible REST API."""
    if not GROQ_API_KEY:
        return None
    import httpx
    r = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.5,
        },
        timeout=15.0,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip() or None


def generate_ai_summary(symbol: str, metrics: Dict[str, Any]) -> tuple[Optional[str], str]:
    """
    Try Gemini → Claude → Groq in order.
    Returns (text, provider_name). text is None if all providers fail.
    """
    prompt = (
        f"Give a concise 2-sentence market analysis for {symbol}. "
        f"Current price: ${metrics.get('current_price')}, "
        f"change: {metrics.get('change_pct')}%, "
        f"RSI(14): {metrics.get('rsi')}. Be factual and brief."
    )

    providers = [
        ("Gemini", _ai_gemini),
        ("Claude", _ai_claude),
        ("Groq", _ai_groq),
    ]

    errors = []
    for name, fn in providers:
        try:
            text = fn(prompt)
            if text:
                logger.info(f"[{name.lower()}] summary generated for {symbol}")
                return text, name
            errors.append(f"{name}: empty response")
        except Exception as e:
            logger.warning(f"[{name.lower()}] failed: {type(e).__name__}: {str(e)[:150]}")
            logger.debug(traceback.format_exc())
            errors.append(f"{name}: {type(e).__name__}")

    logger.warning(f"All AI providers failed for {symbol}: {errors}")
    return None, "Unavailable"


# ---------------- Asset Classes ----------------
ASSET_CLASSES = {
    "stock": {
        "name": "Stocks",
        "symbols": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA",
            "BRK-B", "JPM", "V", "WMT", "JNJ", "PG", "MA", "HD",
            "CVX", "ABBV", "MRK", "PEP", "KO", "COST",
        ],
    },
    "sp500": {
        "name": "S&P 500",
        "symbols": [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B",
            "LLY", "AVGO", "JPM", "TSLA", "UNH", "XOM", "V", "PG",
            "MA", "JNJ", "HD", "COST", "MRK", "ABBV", "CVX", "WMT",
            "PEP", "KO", "ADBE", "CRM", "MCD", "TMO", "CSCO",
        ],
    },
    "nasdaq": {
        "name": "NASDAQ",
        "symbols": [
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
            "AVGO", "COST", "NFLX", "AMD", "ADBE", "PEP", "CSCO",
            "INTC", "QCOM", "TXN", "AMAT", "INTU", "AMGN",
        ],
    },
    "etf": {
        "name": "ETFs",
        "symbols": [
            "SPY", "QQQ", "VTI", "VOO", "IWM", "DIA",
            "VEA", "VWO", "AGG", "BND", "GLD", "SLV",
        ],
    },
    "crypto": {
        "name": "Crypto",
        "symbols": [
            "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD",
            "XRP-USD", "DOGE-USD", "AVAX-USD", "DOT-USD",
        ],
    },
    "forex": {
        "name": "Forex",
        "symbols": [
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
            "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURGBP=X",
        ],
    },
    "sector": {
        "name": "Sector ETFs",
        "symbols": [
            "XLK", "XLF", "XLV", "XLE", "XLY", "XLP",
            "XLI", "XLB", "XLU", "XLRE", "XLC",
        ],
    },
}

POPULAR_SYMBOLS = [
    {"symbol": "AAPL",    "name": "Apple Inc.",        "class": "stock"},
    {"symbol": "MSFT",    "name": "Microsoft Corp.",   "class": "stock"},
    {"symbol": "NVDA",    "name": "NVIDIA Corp.",      "class": "stock"},
    {"symbol": "SPY",     "name": "SPDR S&P 500 ETF",  "class": "etf"},
    {"symbol": "BTC-USD", "name": "Bitcoin USD",       "class": "crypto"},
    {"symbol": "EURUSD=X","name": "Euro / US Dollar",  "class": "forex"},
    {"symbol": "XLK",     "name": "Technology Select", "class": "sector"},
]

# ============================================================
# Pydantic models
# ============================================================
class ChartRequest(BaseModel):
    symbol: Optional[str] = None
    ticker: Optional[str] = None
    asset_type: str = "stock"
    timeframe: str = "3M"
    include_ai: bool = True


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    name: str = ""


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class PortfolioAddRequest(BaseModel):
    symbol: str
    shares: float = Field(gt=0)
    buy_price: Optional[float] = None


class AlertRequest(BaseModel):
    symbol: str
    target_price: float = Field(gt=0)
    condition: str = "above"


class ShareRequest(BaseModel):
    symbol: str
    note: str = ""


class CheckoutRequest(BaseModel):
    plan: str


PLANS = {
    "free": {"name": "Free", "price": 0},
    "pro": {"name": "Pro", "price": 999},
    "premium": {"name": "Premium", "price": 2999},
}


# ============================================================
# Health

# ============================================================
@app.get("/")
async def root():
    return FileResponse("stock-dashboard.html")


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "message": "Backend is running!",
        "gemini": bool(GEMINI_API_KEY),
        "claude": bool(ANTHROPIC_API_KEY),
        "groq": bool(GROQ_API_KEY),
        "mongodb": mongo_db is not None,
        "stripe": bool(STRIPE_SECRET_KEY),
    }


@app.get("/api/debug/providers/{symbol}")
async def debug_providers(symbol: str, timeframe: str = "3M"):
    from market_data import _fetch_yahoo, _fetch_polygon, _fetch_finnhub
    results = {}
    for name, fn in (
        ("yahoo", _fetch_yahoo),
        ("polygon", _fetch_polygon),
        ("finnhub", _fetch_finnhub),
    ):
        try:
            r = fn(symbol.upper(), timeframe)
            results[name] = {
                "ok": True,
                "points": len(r["data"]),
                "current_price": r["current_price"],
                "change_pct": r["change_pct"],
            }
        except Exception as e:
            results[name] = {"ok": False, "error": str(e)}
    return {"symbol": symbol.upper(), "timeframe": timeframe, "providers": results}


@app.get("/api/debug/ai/{symbol}")
async def debug_ai(symbol: str):
    """Test each AI provider independently."""
    prompt = f"Say 'test' and nothing else."
    results = {}
    for name, fn in (
        ("gemini", _ai_gemini),
        ("claude", _ai_claude),
        ("groq", _ai_groq),
    ):
        try:
            text = fn(prompt)
            results[name] = {"ok": bool(text), "response": text}
        except Exception as e:
            results[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return {"symbol": symbol.upper(), "providers": results}


# ============================================================
# Stock Routes
# ============================================================
@app.post("/api/chart/hybrid")
async def chart_hybrid(req: ChartRequest):
    symbol = (req.symbol or req.ticker or "").upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="Missing symbol")

    data = fetch_asset_data(symbol, req.timeframe)
    closes = data["closes"]

    rsi = calculate_rsi(closes)
    macd = calculate_macd(closes)
    boll = calculate_bollinger(closes)

    metrics = {
        "current_price": data["current_price"],
        "change_pct": data["change_pct"],
        "rsi": rsi,
    }

    ai_text: Optional[str] = None
    provider = "Unavailable"
    if req.include_ai:
        ai_text, provider = generate_ai_summary(symbol, metrics)

    if not ai_text:
        ai_text = "AI analysis is currently unavailable."

    return {
        "symbol": symbol,
        "asset_type": req.asset_type,
        "current_price": data["current_price"],
        "change_pct": data["change_pct"],
        "data": data["data"],
        "source": data.get("source", "unknown"),
        "technical_indicators": {"rsi": rsi, "macd": macd, "bollinger": boll},
        "ai_analysis": ai_text,
        "provider": provider,
        "alerts": {},
    }


@app.get("/api/stock/{symbol}")
async def get_stock(symbol: str):
    data = fetch_asset_data(symbol.upper(), "3M")
    closes = data["closes"]
    return {
        "symbol": symbol.upper(),
        "current_price": data["current_price"],
        "change_pct": data["change_pct"],
        "source": data.get("source", "unknown"),
        "technical_indicators": {
            "rsi": calculate_rsi(closes),
            "macd": calculate_macd(closes),
            "bollinger": calculate_bollinger(closes),
        },
    }


# ============================================================
# Search Routes
# ============================================================
@app.get("/api/search/popular")
async def popular():
    return {"popular": POPULAR_SYMBOLS, "categories": ASSET_CLASSES}


@app.get("/api/asset-classes")
async def asset_classes():
    return {"asset_classes": ASSET_CLASSES}


@app.get("/api/search")
async def search(q: str = ""):
    q = q.upper()
    matches = []
    for cls, info in ASSET_CLASSES.items():
        for sym in info["symbols"]:
            if q in sym:
                matches.append({"symbol": sym, "class": cls})
    return {"query": q, "results": matches[:20]}


# ============================================================
# Auth
# ============================================================
@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    db_required()
    if users_col.find_one({"email": req.email}):
        raise HTTPException(status_code=400, detail="User already exists")
    doc = {
        "email": req.email,
        "name": req.name,
        "password": hash_password(req.password),
        "plan": "free",
        "created_at": datetime.now(timezone.utc),
    }
    res = users_col.insert_one(doc)
    uid = str(res.inserted_id)
    return {
        "message": "User created",
        "user": {"id": uid, "email": req.email, "name": req.name, "plan": "free"},
        "token": make_token(uid, req.email),
    }


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    db_required()
    user = users_col.find_one({"email": req.email})
    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    uid = str(user["_id"])
    return {
        "message": "Login successful",
        "user": {
            "id": uid,
            "email": user["email"],
            "name": user.get("name", ""),
            "plan": user.get("plan", "free"),
        },
        "token": make_token(uid, user["email"]),
    }


@app.get("/api/auth/me")
async def me(user: Dict[str, Any] = Depends(current_user)):
    return {
        "user": {
            "id": user["_id"],
            "email": user["email"],
            "name": user.get("name", ""),
            "plan": user.get("plan", "free"),
        }
    }


# ============================================================
# Portfolio
# ============================================================
@app.post("/api/portfolio")
async def add_portfolio(req: PortfolioAddRequest, user: Dict[str, Any] = Depends(current_user)):
    db_required()
    data = fetch_asset_data(req.symbol.upper(), "1D")
    price = req.buy_price if req.buy_price is not None else data["current_price"]
    existing = portfolio_col.find_one({"user_id": user["_id"], "symbol": req.symbol.upper()})
    if existing:
        portfolio_col.update_one(
            {"_id": existing["_id"]},
            {"$inc": {"shares": req.shares}, "$set": {"updated_at": datetime.now(timezone.utc)}},
        )
    else:
        portfolio_col.insert_one({
            "user_id": user["_id"],
            "symbol": req.symbol.upper(),
            "shares": req.shares,
            "buy_price": price,
            "created_at": datetime.now(timezone.utc),
        })
    return {"message": f"Added {req.symbol.upper()} to portfolio"}


@app.get("/api/portfolio")
async def get_portfolio(user: Dict[str, Any] = Depends(current_user)):
    db_required()
    items = []
    total = 0.0
    for item in portfolio_col.find({"user_id": user["_id"]}):
        try:
            price_data = fetch_asset_data(item["symbol"], "1D")
            cur = price_data["current_price"]
        except HTTPException:
            cur = item.get("buy_price", 0)
        value = cur * item["shares"]
        total += value
        items.append({
            "symbol": item["symbol"],
            "shares": item["shares"],
            "buy_price": item["buy_price"],
            "current_price": cur,
            "value": round(value, 2),
        })
    return {"portfolio": items, "total_value": round(total, 2)}


@app.delete("/api/portfolio/{symbol}")
async def delete_portfolio(symbol: str, user: Dict[str, Any] = Depends(current_user)):
    db_required()
    res = portfolio_col.delete_one({"user_id": user["_id"], "symbol": symbol.upper()})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Not found")
    return {"message": f"Removed {symbol.upper()}"}


# ============================================================
# Alerts
# ============================================================
@app.post("/api/alerts")
async def create_alert(req: AlertRequest, user: Dict[str, Any] = Depends(current_user)):
    db_required()
    res = alerts_col.insert_one({
        "user_id": user["_id"],
        "symbol": req.symbol.upper(),
        "target_price": req.target_price,
        "condition": req.condition,
        "triggered": False,
        "created_at": datetime.now(timezone.utc),
    })
    return {"id": str(res.inserted_id), "message": "Alert created"}


@app.get("/api/alerts")
async def list_alerts(user: Dict[str, Any] = Depends(current_user)):
    db_required()
    items = []
    for a in alerts_col.find({"user_id": user["_id"]}):
        items.append({
            "id": str(a["_id"]),
            "symbol": a["symbol"],
            "target_price": a["target_price"],
            "condition": a["condition"],
            "triggered": a.get("triggered", False),
        })
    return {"alerts": items}


@app.delete("/api/alerts/{alert_id}")
async def delete_alert(alert_id: str, user: Dict[str, Any] = Depends(current_user)):
    db_required()
    from bson import ObjectId
    try:
        oid = ObjectId(alert_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id")
    res = alerts_col.delete_one({"_id": oid, "user_id": user["_id"]})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": "Alert deleted"}


# ============================================================
# Social Share
# ============================================================
@app.post("/api/share")
async def share(req: ShareRequest, user: Dict[str, Any] = Depends(current_user)):
    db_required()
    res = shares_col.insert_one({
        "user_id": user["_id"],
        "symbol": req.symbol.upper(),
        "note": req.note,
        "created_at": datetime.now(timezone.utc),
    })
    return {"id": str(res.inserted_id), "message": "Shared"}


@app.get("/api/share/{symbol}")
async def get_shares(symbol: str):
    db_required()
    items = []
    for s in shares_col.find({"symbol": symbol.upper()}).sort("created_at", -1).limit(50):
        items.append({
            "id": str(s["_id"]),
            "symbol": s["symbol"],
            "note": s.get("note", ""),
            "created_at": s["created_at"].isoformat() if hasattr(s["created_at"], "isoformat") else str(s["created_at"]),
        })
    return {"symbol": symbol.upper(), "shares": items}


# ============================================================
# Stripe
# ============================================================
@app.get("/api/plans")
async def plans():
    return {"plans": PLANS, "currency": "USD"}


@app.post("/api/create-checkout-session")
async def create_checkout(req: CheckoutRequest, user: Dict[str, Any] = Depends(current_user)):
    if req.plan not in PLANS:
        raise HTTPException(status_code=400, detail="Invalid plan")
    if req.plan == "free":
        return {"message": "Free plan — no payment needed"}
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Stripe not configured")
    try:
        import stripe
        stripe.api_key = STRIPE_SECRET_KEY
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {"name": f"Stock Dashboard {PLANS[req.plan]['name']}"},
                    "unit_amount": PLANS[req.plan]["price"],
                    "recurring": {"interval": "month"},
                },
                "quantity": 1,
            }],
            success_url="https://example.com/success",
            cancel_url="https://example.com/cancel",
        )
        return {"sessionId": session.id, "url": session.url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stripe error: {e}")


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
