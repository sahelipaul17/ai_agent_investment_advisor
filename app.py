# app.py
import os
import math
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

import yfinance as yf
from fastapi import FastAPI, Query
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI  # OpenAI client pointed to Gemini-compatible endpoint

# Load .env
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment variables (.env)")

#OpenAI client
client = OpenAI(api_key=API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

app = FastAPI(title="Invest Advisor Agent (SIP / MF / Stock / Crypto / Gold)")

#Utilities
US_TICKERS = {"AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META"}
CRYPTO_SUFFIX = "-USD"   # yfinance uses BTC-USD, ETH-USD etc.
GOLD_TICKERS = {"XAUUSD", "GC=F", "GLD"}  # common gold tickers

def normalize_ticker(symbol: str, asset_type: str) -> str:
    s = symbol.strip().upper()
    if asset_type == "crypto":
        # allow user to pass BTC, BTC-USD, BTC/USDT, normalize to BTC-USD
        s = s.replace("/", "-").replace("USDT", "USD")
        if "-" not in s:
            s = s + CRYPTO_SUFFIX
        return s
    if asset_type == "gold":
        # accept XAU, XAUUSD or GC=F; prefer GC=F then XAUUSD
        if s in GOLD_TICKERS:
            return s
        if s in ("XAU", "GOLD"):
            return "XAUUSD"
        return s
    if asset_type == "mutual_fund":
        # many mutual funds are listed with tickers on Yahoo 
        return s
    if asset_type == "stock":
        if "." in s:
            return s
        if s in US_TICKERS:
            return s
        # common heuristic: default to NSE if looks like Indian (3-5 letters)
        if len(s) >= 3 and len(s) <= 5 and s.isalpha():
            return f"{s}.NS"
        return s
    return s

#Market Data Fetching Tool 
def fetch_market_history(ticker: str, period: str = "6mo", interval: str = "1d"):
    """
    Returns pandas DataFrame of history or raises Exception
    """
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval)
    if df is None or df.shape[0] == 0:
        raise ValueError(f"No data found for {ticker}")
    return df

def fetch_latest_price(ticker: str):
    df = fetch_market_history(ticker, period="7d", interval="1d")
    last = df.tail(1)
    if last.empty:
        raise ValueError(f"No recent price for {ticker}")
    price = float(last["Close"].values[0])
    return price

#Indicators Tool
def calculate_basic_indicators(ticker: str, period: str = "6mo"):
    df = fetch_market_history(ticker, period=period, interval="1d")
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["RET_1D"] = df["Close"].pct_change()
    # RSI simple implementation
    delta = df["Close"].diff().dropna()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(span=14).mean()
    roll_down = down.ewm(span=14).mean()
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    df = df.assign(RSI=rsi.reindex(df.index, method="ffill"))
    latest = df.dropna().iloc[-1]
    return {
        "close": float(latest["Close"]),
        "sma20": float(latest["SMA20"]),
        "sma50": float(latest["SMA50"]),
        "rsi": float(latest["RSI"]),
        "vol": float(latest["Volume"]) if "Volume" in latest else None,
        "recent_return_1d_pct": float(latest["RET_1D"] * 100) if not np.isnan(latest["RET_1D"]) else None
    }

#Mutual Fund / NAV Tool
def fetch_mutual_fund_nav(symbol: str):
    """
    Try Yahoo via yfinance Ticker.history (some MFs are available), otherwise return error.
    For production you'd integrate with a mutual fund API (AMFI/third-party).
    """
    t = yf.Ticker(symbol)
    df = t.history(period="1y", interval="1d")
    if df is None or df.empty:
        raise ValueError(f"No NAV data available for mutual fund {symbol}")
    last = df.tail(1)
    nav = float(last["Close"].values[0])
    return {"nav": nav, "date": str(last.index[0].date())}

#SIP Calculator Tool
def sip_simulation(monthly_amount: float, annual_return_pct: float, years: int):
    """
    Simple SIP future value assuming monthly contributions and fixed annual return (compounded monthly).
    """
    r_month = annual_return_pct / 100.0 / 12.0
    months = years * 12
    if r_month == 0:
        fv = monthly_amount * months
        total_invested = monthly_amount * months
    else:
        fv = monthly_amount * (( (1 + r_month) ** months - 1) / r_month) * (1 + r_month)
        total_invested = monthly_amount * months
    gain = fv - total_invested
    return {"future_value": float(fv), "total_invested": float(total_invested), "gain": float(gain)}

#Simple Portfolio Recommender Tool
def portfolio_recommendation(risk_level: str, amount: float):
    """
    Very simple rule-based allocation:
     - conservative, moderate, aggressive
    """
    risk = risk_level.lower()
    if risk not in ("conservative", "moderate", "aggressive"):
        raise ValueError("risk_level must be one of conservative|moderate|aggressive")
    if risk == "conservative":
        alloc = {"debt_mf_pct": 0.70, "equity_pct": 0.15, "gold_pct": 0.10, "crypto_pct": 0.05}
    elif risk == "moderate":
        alloc = {"debt_mf_pct": 0.40, "equity_pct": 0.40, "gold_pct": 0.10, "crypto_pct": 0.10}
    else:  # aggressive
        alloc = {"debt_mf_pct": 0.10, "equity_pct": 0.65, "gold_pct": 0.10, "crypto_pct": 0.15}
    # compute amounts
    alloc_amounts = {k: round(v * amount, 2) for k, v in alloc.items()}
    return {"allocation_pct": alloc, "allocation_amounts": alloc_amounts}

#Gemini (LLM) wrapper
def call_gemini(prompt: str, model_name: str = "gemini-2.5-flash", max_tokens: int = 300):
    # Uses OpenAI client configured to Gemini base URL
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    # current SDK returns message object with .content
    try:
        return resp.choices[0].message.content
    except Exception:
        # Fallback to str(resp)
        return str(resp)

#APIs
class SIPRequest(BaseModel):
    monthly_amount: float
    annual_return_pct: float = 12.0
    years: int = 10
    risk_profile: Optional[str] = "moderate"

#Endpoints
@app.get("/")
def root():
    return {"message": "Invest Advisor Agent running. Use /advise or /sip endpoints."}

@app.get("/advise/{asset_type}/{symbol}")
def advise(
    asset_type: str,
    symbol: str,
    risk_profile: str = Query("moderate", description="conservative|moderate|aggressive"),
    include_llm: bool = Query(True, description="ask Gemini for a natural-language rationale")
):
    """
    asset_type: one of stock | crypto | gold | mutual_fund
    symbol: ticker or symbol (e.g., AAPL, TSLA, BTC, RELIANCE, ICICIPrudential.MF)
    """
    atype = asset_type.lower()
    if atype not in ("stock", "crypto", "gold", "mutual_fund"):
        return {"error": "asset_type must be one of stock|crypto|gold|mutual_fund"}

    try:
        ticker = normalize_ticker(symbol, atype)
    except Exception as e:
        return {"error": f"ticker normalize error: {e}"}

    tools_out = {}
    try:
        if atype == "mutual_fund":
            mf = fetch_mutual_fund_nav(ticker)
            tools_out["mutual_fund_nav"] = mf
            # simple recommendation: SIP or Lump based on risk_profile
            rec = "SIP recommended" if risk_profile != "aggressive" else "Lump or SIP depending on conviction"
        else:
            # stock/crypto/gold
            # try indicators and price
            indicators = calculate_basic_indicators(ticker)
            price = indicators["close"]
            tools_out["indicators"] = indicators
            tools_out["latest_price"] = price
            # simple rule-based signal
            signal = "HOLD"
            # example heuristic: RSI below 30 -> BUY, above 70 -> SELL, else check SMA20 vs SMA50
            if indicators["rsi"] is not None:
                if indicators["rsi"] < 30:
                    signal = "BUY"
                elif indicators["rsi"] > 70:
                    signal = "SELL"
                else:
                    # trend check
                    if indicators["sma20"] and indicators["sma50"]:
                        if indicators["sma20"] > indicators["sma50"]:
                            signal = "BUY"
                        else:
                            signal = "HOLD"
            tools_out["signal"] = signal
            rec = signal

    except Exception as e:
        return {"error": f"tools error: {e}", "tools_partial": tools_out}

    # portfolio recommendation
    try:
        portfolio = portfolio_recommendation(risk_profile, 100000)  # example base; not user amount
        tools_out["sample_portfolio"] = portfolio
    except Exception:
        tools_out["sample_portfolio"] = {}

    # Build LLM prompt
    llm_explanation = None
    if include_llm:
        prompt_lines = [
            f"You are an investment advisor (educational only).",
            f"Asset type: {atype}",
            f"Symbol: {ticker}",
            f"Risk profile: {risk_profile}",
            "Tool outputs:",
            str(tools_out),
            "Based on the above, give:",
            "1) Short recommendation (BUY/HOLD/SELL or SIP/LUMP).",
            "2) Short rationale (2-3 sentences).",
            "3) One risk-management bullet (stop-loss, horizon, allocation).",
            "Do NOT provide legal or definitive investment advice."
        ]
        prompt = "\n".join(prompt_lines)
        try:
            llm_explanation = call_gemini(prompt)
        except Exception as e:
            llm_explanation = f"LLM call failed: {e}"

    return {
        "asset_type": atype,
        "symbol": ticker,
        "tools": tools_out,
        "recommendation": rec,
        "llm_explanation": llm_explanation
    }

@app.post("/sip")
def sip_plan(req: SIPRequest):
    """
    Simulate SIP growth and return portfolio allocation suggestion based on risk profile.
    """
    sip = sip_simulation(req.monthly_amount, req.annual_return_pct, req.years)
    allocation = portfolio_recommendation(req.risk_profile, req.monthly_amount * req.years + 0)  # approximate investable
    # LLM summary
    prompt = (
        f"SIP simulation:\nmonthly={req.monthly_amount}, annual_return_pct={req.annual_return_pct}, years={req.years}\n"
        f"Results: {sip}\nRisk profile: {req.risk_profile}\nProvide a short user-facing summary (2-3 sentences) and one risk note."
    )
    llm = call_gemini(prompt) if API_KEY else None
    return {"sip": sip, "allocation": allocation, "llm_summary": llm}
