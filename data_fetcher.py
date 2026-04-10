"""
Data fetching and caching layer.
Primary data source: fundamentals_cache.json bundled in the repo (built locally).
Fallback: live yfinance fetch for individual tickers (watchlist additions etc).
"""

import json
import os
import time
import random
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
    WATCHLIST_FILE,
)

# Path to the pre-built cache bundled in the repo
BUNDLED_CACHE_FILE = "fundamentals_cache.json"


def _ensure_cache_dir():
    Path(CACHE_DIR).mkdir(exist_ok=True)


def _cache_path(filename: str) -> str:
    return os.path.join(CACHE_DIR, filename)


def _save_cache(results: dict):
    _ensure_cache_dir()
    cache_file = _cache_path(FUNDAMENTALS_CACHE_FILE)
    try:
        with open(cache_file, "w") as f:
            json.dump(results, f, default=str)
    except Exception:
        pass


def _load_cache() -> dict:
    """Load cache, preferring the bundled repo file over the runtime cache."""
    # First try the bundled cache in the repo root
    if os.path.exists(BUNDLED_CACHE_FILE):
        try:
            with open(BUNDLED_CACHE_FILE, "r") as f:
                data = json.load(f)
                if len(data) > 100:
                    return data
        except Exception:
            pass

    # Fall back to runtime cache
    _ensure_cache_dir()
    cache_file = _cache_path(FUNDAMENTALS_CACHE_FILE)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            pass

    return {}


# ── Ticker Universe ────────────────────────────────────────────────


@st.cache_data(ttl=86400, show_spinner=False)
def get_broad_universe(min_market_cap_b: float = 10.0) -> list[str]:
    """Return the ticker universe. Derived from the bundled cache keys."""
    data = _load_cache()
    if data:
        return sorted(data.keys())
    return []


# ── Main Data Loader ──────────────────────────────────────────────


def fetch_universe_data(
    tickers: list[str],
    min_market_cap_b: float = 10.0,
    progress_callback=None,
) -> dict[str, dict]:
    """
    Load the pre-built cache. No live fetching needed for the main universe.
    The cache was built locally using build_cache.py and uploaded to the repo.
    """
    if progress_callback:
        progress_callback(0.5, "Loading pre-built data...")

    results = _load_cache()

    if progress_callback:
        progress_callback(1.0, f"Loaded {len(results)} tickers.")

    return _filter_by_market_cap(results, min_market_cap_b)


def _filter_by_market_cap(data: dict, min_cap_b: float) -> dict:
    min_cap = min_cap_b * 1e9
    return {
        k: v for k, v in data.items()
        if v.get("marketCap", 0) >= min_cap
    }


# ── Single Ticker Live Fetch (for watchlist additions) ─────────────


def _fetch_single_live(ticker: str) -> dict | None:
    """Fetch one ticker live from yfinance. Used for watchlist additions."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        if not info.get("marketCap"):
            return None

        hist = t.history(period="1y")
        if hist.empty or len(hist) < 20:
            return None

        close = hist["Close"]
        price = float(close.iloc[-1])

        def pct_ret(days):
            if len(close) >= days + 1:
                past = float(close.iloc[-(days + 1)])
                if past > 0:
                    return round((price - past) / past, 4)
            return None

        sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        surprise = None
        try:
            cal = t.earnings_dates
            if cal is not None and not cal.empty and "Surprise(%)" in cal.columns:
                recent = cal.dropna(subset=["Surprise(%)"])
                if not recent.empty:
                    surprise = float(recent["Surprise(%)"].iloc[0])
                    if not np.isfinite(surprise):
                        surprise = None
        except Exception:
            pass

        target = info.get("targetMeanPrice")
        cur = info.get("currentPrice") or info.get("previousClose")
        upside = round((target - cur) / cur, 4) if target and cur and cur > 0 else None

        return {
            "ticker": ticker,
            "shortName": info.get("shortName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "marketCap": info.get("marketCap", 0),
            "currentPrice": round(price, 2),
            "currency": info.get("currency", "USD"),
            "forwardPE": info.get("forwardPE"),
            "trailingPE": info.get("trailingPE"),
            "pegRatio": info.get("pegRatio"),
            "priceToBook": info.get("priceToBook"),
            "priceToSalesTrailing12Months": info.get("priceToSalesTrailing12Months"),
            "enterpriseToEbitda": info.get("enterpriseToEbitda"),
            "enterpriseToRevenue": info.get("enterpriseToRevenue"),
            "revenueGrowth": info.get("revenueGrowth"),
            "earningsGrowth": info.get("earningsGrowth"),
            "revenueQuarterlyGrowth": info.get("revenueQuarterlyGrowth"),
            "earningsQuarterlyGrowth": info.get("earningsQuarterlyGrowth"),
            "grossMargins": info.get("grossMargins"),
            "operatingMargins": info.get("operatingMargins"),
            "profitMargins": info.get("profitMargins"),
            "returnOnEquity": info.get("returnOnEquity"),
            "returnOnAssets": info.get("returnOnAssets"),
            "momentum_1m": pct_ret(21),
            "momentum_3m": pct_ret(63),
            "momentum_6m": pct_ret(126),
            "momentum_12m": pct_ret(252) if len(close) >= 253 else pct_ret(max(len(close) - 2, 1)),
            "momentum_vs_sma50": round((price - sma50) / sma50, 4) if sma50 and sma50 > 0 else None,
            "momentum_vs_sma200": round((price - sma200) / sma200, 4) if sma200 and sma200 > 0 else None,
            "analyst_mean_target_upside": upside,
            "analyst_recommendation_score": info.get("recommendationMean"),
            "earnings_surprise_pct": surprise,
            "analyst_count": info.get("numberOfAnalystOpinions", 0),
            "lastUpdated": datetime.now().isoformat(),
        }
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_single_ticker(ticker: str) -> dict | None:
    """Fetch a single ticker. Checks cache first, then live."""
    data = _load_cache()
    if ticker in data:
        return data[ticker]
    return _fetch_single_live(ticker)


# ── Watchlist Management ───────────────────────────────────────────


def load_watchlist() -> list[dict]:
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
    _ensure_cache_dir()
    path = _cache_path(WATCHLIST_FILE)
    with open(path, "w") as f:
        json.dump(watchlist, f, indent=2)


def add_to_watchlist(ticker: str) -> list[dict]:
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
    wl = load_watchlist()
    wl = [w for w in wl if w["ticker"] != ticker]
    save_watchlist(wl)
    return wl
