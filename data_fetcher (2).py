"""
Data fetching and caching layer.
Key principle: NEVER lose cached data. Only overwrite a cached ticker
when a fresh fetch succeeds. Failed fetches keep the old data.
"""

import json
import os
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _save_cache(results: dict):
    """Save results to disk cache."""
    _ensure_cache_dir()
    cache_file = _cache_path(FUNDAMENTALS_CACHE_FILE)
    try:
        with open(cache_file, "w") as f:
            json.dump(results, f, default=str)
    except Exception:
        pass


def _load_cache() -> dict:
    """Load cached results from disk."""
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
def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 constituents from Wikipedia, with hardcoded fallback."""
    try:
        tables = pd.read_html(SP500_WIKI_URL)
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(tickers) > 400:
            return sorted(tickers)
    except Exception:
        pass

    return sorted([
        "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE",
        "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALK",
        "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN",
        "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", "APTV", "ARE", "ATO",
        "AVB", "AVGO", "AVY", "AWK", "AXP", "AZO", "BA", "BAC", "BAX", "BBWI",
        "BBY", "BDX", "BEN", "BF-B", "BG", "BIIB", "BIO", "BK", "BKNG", "BKR",
        "BLK", "BMY", "BR", "BRK-B", "BRO", "BSX", "BWA", "BXP", "C", "CAG",
        "CAH", "CARR", "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL", "CDAY", "CDNS",
        "CDW", "CE", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR", "CI", "CINF",
        "CL", "CLX", "CMA", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP",
        "COF", "COO", "COP", "COST", "CPB", "CPRT", "CPT", "CRL", "CRM", "CSCO",
        "CSGP", "CSX", "CTAS", "CTLT", "CTRA", "CTSH", "CTVA", "CVS", "CVX", "CZR",
        "D", "DAL", "DD", "DE", "DFS", "DG", "DGX", "DHI", "DHR", "DIS",
        "DLR", "DLTR", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN",
        "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EIX", "EL", "EMN", "EMR",
        "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ES", "ESS", "ETN", "ETR",
        "ETSY", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR", "F", "FANG", "FAST",
        "FBHS", "FCX", "FDS", "FDX", "FE", "FFIV", "FIS", "FISV", "FITB", "FLT",
        "FMC", "FOX", "FOXA", "FRT", "FTNT", "FTV", "GD", "GE", "GEHC", "GEN",
        "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN",
        "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN", "HCA", "HSIC", "HST", "HSY",
        "HUM", "HWM", "IBM", "ICE", "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTC",
        "INTU", "INVH", "IP", "IPG", "IQV", "IR", "IRM", "ISRG", "IT", "ITW",
        "IVZ", "J", "JBHT", "JCI", "JKHY", "JNJ", "JNPR", "JPM", "K", "KDP",
        "KEY", "KEYS", "KHC", "KIM", "KLAC", "KMB", "KMI", "KMX", "KO", "KR",
        "L", "LDOS", "LEN", "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNC",
        "LNT", "LOW", "LRCX", "LULU", "LUV", "LVS", "LW", "LYB", "LYV", "MA",
        "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET",
        "META", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST", "MO",
        "MOH", "MOS", "MPC", "MPWR", "MRK", "MRNA", "MRO", "MS", "MSCI", "MSFT",
        "MSI", "MTB", "MTCH", "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM",
        "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE",
        "NVDA", "NVR", "NWL", "NWS", "NWSA", "NXPI", "O", "ODFL", "OGN", "OKE",
        "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY", "PARA", "PAYC", "PAYX", "PCAR",
        "PCG", "PEAK", "PEG", "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM",
        "PKG", "PKI", "PLD", "PM", "PNC", "PNR", "PNW", "POOL", "PPG", "PPL",
        "PRU", "PSA", "PSX", "PTC", "PVH", "PWR", "PYPL", "QCOM", "QRVO", "RCL",
        "RE", "REG", "REGN", "RF", "RHI", "RJF", "RL", "RMD", "ROK", "ROL",
        "ROP", "ROST", "RSG", "RTX", "RVTY", "SBAC", "SBUX", "SCHW", "SEE", "SHW",
        "SJM", "SLB", "SNA", "SNPS", "SO", "SPG", "SPGI", "SRE", "STE", "STLD",
        "STT", "STX", "STZ", "SWK", "SWKS", "SYF", "SYK", "SYY", "T", "TAP",
        "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TFX", "TGT", "TMO", "TMUS",
        "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO",
        "TXN", "TXT", "TYL", "UAL", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS",
        "URI", "USB", "V", "VFC", "VICI", "VLO", "VMC", "VRSK", "VRSN", "VRTX",
        "VTR", "VTRS", "VZ", "WAB", "WAT", "WBA", "WBD", "WDC", "WEC", "WELL",
        "WFC", "WHR", "WM", "WMB", "WMT", "WRB", "WRK", "WST", "WTW", "WY",
        "WYNN", "XEL", "XOM", "XRAY", "XYL", "YUM", "ZBH", "ZBRA", "ZION", "ZTS",
        "COIN", "PLTR", "CRWD", "DASH", "DDOG", "SNOW", "NET", "ZS", "MDB",
        "TTD", "PANW", "ABNB", "RIVN", "HOOD", "RBLX", "ARM", "SMCI", "VST",
        "DECK", "AXON", "FICO", "GDDY", "HUBB", "TW", "GEV", "VLTO", "KVUE",
        "SOLV", "SW",
    ])


@st.cache_data(ttl=86400, show_spinner=False)
def get_broad_universe(min_market_cap_b: float = 10.0) -> list[str]:
    sp500 = get_sp500_tickers()
    supplemental = _get_supplemental_tickers()
    return sorted(set(sp500 + supplemental))


def _get_supplemental_tickers() -> list[str]:
    return [
        "PLTR", "COIN", "HOOD", "RBLX", "SNOW", "NET", "CRWD", "DDOG",
        "ZS", "MDB", "BILL", "TTD", "PINS", "SNAP", "U", "PATH",
        "RIVN", "LCID", "JOBY", "GRAB", "SE", "MELI", "NU",
        "TSM", "BABA", "JD", "PDD", "BIDU", "NIO", "LI", "XPEV",
        "SHOP", "TD", "RY", "CNQ", "SU", "BN", "BAM",
        "BX", "KKR", "APO", "ARES", "OWL",
        "SPOT", "SQ", "MSTR", "CELH", "DUOL", "CAVA",
    ]


# ── Batch Price Download ───────────────────────────────────────────


def _batch_download_prices(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Download 1y price history in batch. Chunks of 50 with pauses."""
    all_prices = {}
    chunk_size = 50

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            data = yf.download(
                tickers=chunk,
                period="1y",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )

            if len(chunk) == 1:
                ticker = chunk[0]
                if not data.empty and len(data) >= 20:
                    all_prices[ticker] = data
            else:
                for ticker in chunk:
                    try:
                        ticker_data = data[ticker].dropna(how="all")
                        if not ticker_data.empty and len(ticker_data) >= 20:
                            all_prices[ticker] = ticker_data
                    except (KeyError, TypeError):
                        continue
        except Exception:
            pass

        if i + chunk_size < len(tickers):
            time.sleep(2)

    return all_prices


# ── Single Ticker Info Fetch with Retry ────────────────────────────


def _fetch_info_with_retry(ticker: str, max_retries: int = 3) -> dict | None:
    """Fetch .info for a single ticker with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            if not info.get("marketCap"):
                return None

            surprise_pct = None
            try:
                calendar = t.earnings_dates
                if calendar is not None and not calendar.empty and "Surprise(%)" in calendar.columns:
                    recent = calendar.dropna(subset=["Surprise(%)"])
                    if not recent.empty:
                        surprise_pct = float(recent["Surprise(%)"].iloc[0])
                        if not np.isfinite(surprise_pct):
                            surprise_pct = None
            except Exception:
                pass

            info["_earnings_surprise_pct"] = surprise_pct
            return info

        except Exception as e:
            err_str = str(e).lower()
            if "rate limit" in err_str or "too many" in err_str or "401" in err_str:
                wait = (5 * (3 ** attempt)) + random.uniform(0, 3)
                time.sleep(wait)
            else:
                return None

    return None


# ── Build Ticker Data ──────────────────────────────────────────────


def _build_ticker_data(ticker: str, info: dict, price_hist: pd.DataFrame) -> dict | None:
    try:
        if price_hist is None or price_hist.empty or len(price_hist) < 20:
            return None

        close = price_hist["Close"]
        current_price = float(close.iloc[-1])
        momentum = _calc_momentum(close, current_price)
        analyst_data = _calc_analyst_metrics_from_info(info)

        return {
            "ticker": ticker,
            "shortName": info.get("shortName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "marketCap": info.get("marketCap", 0),
            "currentPrice": round(current_price, 2),
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
            **momentum,
            **analyst_data,
            "lastUpdated": datetime.now().isoformat(),
        }
    except Exception:
        return None


def _calc_momentum(close: pd.Series, current_price: float) -> dict:
    def _pct_return(days: int) -> float | None:
        if len(close) >= days + 1:
            past = float(close.iloc[-(days + 1)])
            if past and past > 0:
                return round((current_price - past) / past, 4)
        return None

    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

    return {
        "momentum_1m": _pct_return(21),
        "momentum_3m": _pct_return(63),
        "momentum_6m": _pct_return(126),
        "momentum_12m": _pct_return(252) if len(close) >= 253 else _pct_return(max(len(close) - 2, 1)),
        "momentum_vs_sma50": round((current_price - sma50) / sma50, 4) if sma50 and sma50 > 0 else None,
        "momentum_vs_sma200": round((current_price - sma200) / sma200, 4) if sma200 and sma200 > 0 else None,
    }


def _calc_analyst_metrics_from_info(info: dict) -> dict:
    target_price = info.get("targetMeanPrice")
    current = info.get("currentPrice") or info.get("previousClose")
    upside = None
    if target_price and current and current > 0:
        upside = round((target_price - current) / current, 4)

    return {
        "analyst_mean_target_upside": upside,
        "analyst_recommendation_score": info.get("recommendationMean"),
        "earnings_surprise_pct": info.get("_earnings_surprise_pct"),
        "analyst_count": info.get("numberOfAnalystOpinions", 0),
    }


# ── Main Fetch Orchestrator ───────────────────────────────────────


def fetch_universe_data(
    tickers: list[str],
    min_market_cap_b: float = 10.0,
    progress_callback=None,
) -> dict[str, dict]:
    """
    Fetch data for all tickers.

    KEY PRINCIPLE: Start with ALL existing cached data. Only overwrite
    a ticker when a new fetch succeeds. Failed fetches KEEP the old data.
    This means the universe can only grow or stay the same, never shrink.
    """
    # ── Step 1: Load ALL existing cached data as our starting point ─
    results = _load_cache()

    # If cache is fresh and large enough, just return it
    if _is_cache_fresh(_cache_path(FUNDAMENTALS_CACHE_FILE)) and len(results) > 200:
        return _filter_by_market_cap(results, min_market_cap_b)

    # ── Step 2: Figure out which tickers need fetching ─────────────
    # A ticker needs fetching if it's not in cache OR its data is stale
    tickers_to_fetch = []
    for ticker in tickers:
        if ticker in results:
            last_updated = results[ticker].get("lastUpdated", "")
            try:
                updated_dt = datetime.fromisoformat(last_updated)
                if datetime.now() - updated_dt < timedelta(hours=CACHE_EXPIRY_HOURS):
                    continue  # Fresh enough, skip
            except Exception:
                pass
        tickers_to_fetch.append(ticker)

    if not tickers_to_fetch:
        return _filter_by_market_cap(results, min_market_cap_b)

    total = len(tickers_to_fetch)

    # ── Step 3: Batch price download ───────────────────────────────
    if progress_callback:
        progress_callback(0.05, f"Downloading prices for {total} tickers...")

    all_prices = _batch_download_prices(tickers_to_fetch)

    if progress_callback:
        progress_callback(0.25, f"Prices for {len(all_prices)} tickers. Fetching fundamentals...")

    # ── Step 4: Sequential .info fetch in small batches ────────────
    completed = 0
    new_successes = 0
    batch_size = 3

    for batch_start in range(0, total, batch_size):
        batch = tickers_to_fetch[batch_start:batch_start + batch_size]

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_ticker = {
                executor.submit(_fetch_info_with_retry, t): t
                for t in batch
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                completed += 1
                try:
                    info = future.result()
                    if info and ticker in all_prices:
                        data = _build_ticker_data(ticker, info, all_prices[ticker])
                        if data:
                            # SUCCESS: overwrite the old cached entry
                            results[ticker] = data
                            new_successes += 1
                    # If fetch failed, we do NOT delete results[ticker]
                    # The old cached data stays intact
                except Exception:
                    pass  # Old cached data stays intact

        if progress_callback:
            pct = 0.25 + (completed / total) * 0.70
            progress_callback(
                min(pct, 0.95),
                f"Fundamentals: {completed}/{total} done ({new_successes} new, {len(results)} total)..."
            )

        # Save partial results every 25 tickers
        if completed % 25 == 0:
            _save_cache(results)

        # Pause between mini-batches
        time.sleep(1.5 + random.uniform(0, 1))

    # ── Step 5: Final save ─────────────────────────────────────────
    _save_cache(results)

    if progress_callback:
        progress_callback(1.0, f"Done! {len(results)} total tickers ({new_successes} new this run).")

    return _filter_by_market_cap(results, min_market_cap_b)


def _filter_by_market_cap(data: dict, min_cap_b: float) -> dict:
    min_cap = min_cap_b * 1e9
    return {
        k: v for k, v in data.items()
        if v.get("marketCap", 0) >= min_cap
    }


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


# ── Single Ticker Quick Fetch ──────────────────────────────────────


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_single_ticker(ticker: str) -> dict | None:
    try:
        info = _fetch_info_with_retry(ticker, max_retries=2)
        if not info:
            return None
        prices = _batch_download_prices([ticker])
        if ticker not in prices:
            return None
        return _build_ticker_data(ticker, info, prices[ticker])
    except Exception:
        return None
