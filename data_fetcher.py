"""
Data fetching and caching layer.
Pulls ticker universe, fundamentals, price history, and analyst data from yfinance.
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

from config import (
    CACHE_DIR,
    CACHE_EXPIRY_HOURS,
    FUNDAMENTALS_CACHE_FILE,
    TICKER_LIST_CACHE_FILE,
    WATCHLIST_FILE,
    SP500_WIKI_URL,
)


def _ensure_cache_dir():
    Path(CACHE_DIR).mkdir(exist_ok=True)


def _cache_path(filename: str) -> str:
    return os.path.join(CACHE_DIR, filename)


def _is_cache_fresh(filepath: str, max_age_hours: float = CACHE_EXPIRY_HOURS) -> bool:
    if not os.path.exists(filepath):
        return False
    mtime = os.path.getmtime(filepath)
    age_hours = (time.time() - mtime) / 3600
    return age_hours < max_age_hours


# ── Ticker Universe ────────────────────────────────────────────────


@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 constituents from Wikipedia."""
    try:
        tables = pd.read_html(SP500_WIKI_URL)
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        return sorted(tickers)
    except Exception:
        return []


@st.cache_data(ttl=86400, show_spinner=False)
def get_broad_universe(min_market_cap_b: float = 10.0) -> list[str]:
    """
    Get a broad US equity universe filtered by market cap.
    Uses S&P 500 as base, then extends with common large/mid-cap tickers.
    """
    # Start with S&P 500 as the core
    sp500 = get_sp500_tickers()

    # Supplement with common large-cap tickers not always in S&P 500
    # (recent IPOs, newly eligible, etc.)
    supplemental = _get_supplemental_tickers()
    all_tickers = sorted(set(sp500 + supplemental))

    return all_tickers


def _get_supplemental_tickers() -> list[str]:
    """Additional large/mid-cap tickers that may not be in S&P 500."""
    return [
        # Large tech / recent additions
        "PLTR", "COIN", "HOOD", "RBLX", "SNOW", "NET", "CRWD", "DDOG",
        "ZS", "MDB", "BILL", "TTD", "PINS", "SNAP", "U", "PATH",
        "RIVN", "LCID", "JOBY", "GRAB", "SE", "MELI", "NU",
        # Large international ADRs
        "TSM", "BABA", "JD", "PDD", "BIDU", "NIO", "LI", "XPEV",
        "SHOP", "TD", "RY", "CNQ", "SU", "BN", "BAM",
        # Other large caps
        "BX", "KKR", "APO", "ARES", "OWL",
        "SPOT", "SQ", "MSTR", "CELH", "DUOL", "CAVA",
    ]


# ── Fundamental Data Fetching ──────────────────────────────────────


def fetch_ticker_data(ticker: str) -> dict | None:
    """Fetch all fundamental and price data for a single ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        if not info.get("marketCap"):
            return None

        # Price history for momentum calculations
        hist = t.history(period="1y")
        if hist.empty or len(hist) < 20:
            return None

        current_price = hist["Close"].iloc[-1]

        # Momentum calculations
        momentum = _calc_momentum(hist, current_price)

        # Analyst / EPS revision proxies
        analyst_data = _calc_analyst_metrics(t, info)

        # Combine everything
        data = {
            "ticker": ticker,
            "shortName": info.get("shortName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "marketCap": info.get("marketCap", 0),
            "currentPrice": round(current_price, 2),
            "currency": info.get("currency", "USD"),
            # Valuation
            "forwardPE": info.get("forwardPE"),
            "trailingPE": info.get("trailingPE"),
            "pegRatio": info.get("pegRatio"),
            "priceToBook": info.get("priceToBook"),
            "priceToSalesTrailing12Months": info.get("priceToSalesTrailing12Months"),
            "enterpriseToEbitda": info.get("enterpriseToEbitda"),
            "enterpriseToRevenue": info.get("enterpriseToRevenue"),
            # Growth
            "revenueGrowth": info.get("revenueGrowth"),
            "earningsGrowth": info.get("earningsGrowth"),
            "revenueQuarterlyGrowth": info.get("revenueQuarterlyGrowth"),
            "earningsQuarterlyGrowth": info.get("earningsQuarterlyGrowth"),
            # Profitability
            "grossMargins": info.get("grossMargins"),
            "operatingMargins": info.get("operatingMargins"),
            "profitMargins": info.get("profitMargins"),
            "returnOnEquity": info.get("returnOnEquity"),
            "returnOnAssets": info.get("returnOnAssets"),
            # Momentum (calculated)
            **momentum,
            # Analyst / EPS Revisions (calculated)
            **analyst_data,
            # Meta
            "lastUpdated": datetime.now().isoformat(),
        }

        return data

    except Exception as e:
        return None


def _calc_momentum(hist: pd.DataFrame, current_price: float) -> dict:
    """Calculate momentum metrics from price history."""
    close = hist["Close"]

    def _pct_return(days: int) -> float | None:
        if len(close) >= days:
            past = close.iloc[-(days + 1)]
            if past and past > 0:
                return round((current_price - past) / past, 4)
        return None

    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

    return {
        "momentum_1m": _pct_return(21),
        "momentum_3m": _pct_return(63),
        "momentum_6m": _pct_return(126),
        "momentum_12m": _pct_return(252) if len(close) >= 252 else _pct_return(len(close) - 1),
        "momentum_vs_sma50": round((current_price - sma50) / sma50, 4) if sma50 and sma50 > 0 else None,
        "momentum_vs_sma200": round((current_price - sma200) / sma200, 4) if sma200 and sma200 > 0 else None,
    }


def _calc_analyst_metrics(t: yf.Ticker, info: dict) -> dict:
    """Calculate EPS revision proxy metrics."""
    target_price = info.get("targetMeanPrice")
    current = info.get("currentPrice") or info.get("previousClose")
    upside = None
    if target_price and current and current > 0:
        upside = round((target_price - current) / current, 4)

    # Recommendation score: 1 = Strong Buy, 5 = Strong Sell
    rec_score = info.get("recommendationMean")

    # Earnings surprise from most recent quarter
    surprise_pct = None
    try:
        calendar = t.earnings_dates
        if calendar is not None and not calendar.empty and "Surprise(%)" in calendar.columns:
            recent = calendar.dropna(subset=["Surprise(%)"])
            if not recent.empty:
                surprise_pct = recent["Surprise(%)"].iloc[0]
                if not np.isfinite(surprise_pct):
                    surprise_pct = None
    except Exception:
        pass

    analyst_count = info.get("numberOfAnalystOpinions", 0)

    return {
        "analyst_mean_target_upside": upside,
        "analyst_recommendation_score": rec_score,
        "earnings_surprise_pct": surprise_pct,
        "analyst_count": analyst_count,
    }


# ── Batch Fetching with Progress ───────────────────────────────────


def fetch_universe_data(
    tickers: list[str],
    min_market_cap_b: float = 10.0,
    progress_callback=None,
) -> dict[str, dict]:
    """
    Fetch data for all tickers, filter by market cap, cache results.
    Returns dict of {ticker: data_dict}.
    """
    _ensure_cache_dir()
    cache_file = _cache_path(FUNDAMENTALS_CACHE_FILE)

    # Load existing cache
    cached = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cached = json.load(f)
        except Exception:
            cached = {}

    # Check if cache is fresh enough
    if _is_cache_fresh(cache_file) and len(cached) > 50:
        # Filter by market cap and return
        return _filter_by_market_cap(cached, min_market_cap_b)

    # Fetch fresh data
    results = {}
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(i / total, f"Fetching {ticker} ({i + 1}/{total})")

        # Use cached data if less than CACHE_EXPIRY_HOURS old
        if ticker in cached:
            last_updated = cached[ticker].get("lastUpdated", "")
            try:
                updated_dt = datetime.fromisoformat(last_updated)
                if datetime.now() - updated_dt < timedelta(hours=CACHE_EXPIRY_HOURS):
                    results[ticker] = cached[ticker]
                    continue
            except Exception:
                pass

        data = fetch_ticker_data(ticker)
        if data:
            results[ticker] = data

        # Brief pause to avoid rate limiting
        if i % 10 == 0 and i > 0:
            time.sleep(0.5)

    # Save to cache
    try:
        with open(cache_file, "w") as f:
            json.dump(results, f, default=str)
    except Exception:
        pass

    if progress_callback:
        progress_callback(1.0, "Done!")

    return _filter_by_market_cap(results, min_market_cap_b)


def _filter_by_market_cap(data: dict, min_cap_b: float) -> dict:
    """Filter results by minimum market cap in billions."""
    min_cap = min_cap_b * 1e9
    return {
        k: v for k, v in data.items()
        if v.get("marketCap", 0) >= min_cap
    }


# ── Watchlist Management ───────────────────────────────────────────


def load_watchlist() -> list[dict]:
    """Load watchlist from file. Returns list of {ticker, date_added}."""
    _ensure_cache_dir()
    path = _cache_path(WATCHLIST_FILE)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_watchlist(watchlist: list[dict]):
    """Save watchlist to file."""
    _ensure_cache_dir()
    path = _cache_path(WATCHLIST_FILE)
    with open(path, "w") as f:
        json.dump(watchlist, f, indent=2)


def add_to_watchlist(ticker: str) -> list[dict]:
    """Add ticker to watchlist if not already present."""
    wl = load_watchlist()
    existing = [w["ticker"] for w in wl]
    if ticker not in existing:
        wl.append({
            "ticker": ticker,
            "date_added": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        save_watchlist(wl)
    return wl


def remove_from_watchlist(ticker: str) -> list[dict]:
    """Remove ticker from watchlist."""
    wl = load_watchlist()
    wl = [w for w in wl if w["ticker"] != ticker]
    save_watchlist(wl)
    return wl


# ── Single Ticker Quick Fetch ──────────────────────────────────────


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_single_ticker(ticker: str) -> dict | None:
    """Fetch data for a single ticker (used for watchlist additions)."""
    return fetch_ticker_data(ticker)
