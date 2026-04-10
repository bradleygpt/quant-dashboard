"""
Data fetching and caching layer.
Universe: S&P 500 + S&P MidCap 400 + Nasdaq 100 + supplemental.
Fetch strategy: one ticker at a time with retry. Slow but reliable.
NEVER loses cached data.
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
    TICKER_LIST_CACHE_FILE,
    WATCHLIST_FILE,
    SP500_WIKI_URL,
)

SP400_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
NASDAQ100_WIKI_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"


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
    _ensure_cache_dir()
    cache_file = _cache_path(FUNDAMENTALS_CACHE_FILE)
    try:
        with open(cache_file, "w") as f:
            json.dump(results, f, default=str)
    except Exception:
        pass


def _load_cache() -> dict:
    _ensure_cache_dir()
    cache_file = _cache_path(FUNDAMENTALS_CACHE_FILE)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ── Wikipedia Scraping ─────────────────────────────────────────────


def _scrape_wiki_tickers(url: str, symbol_col: str = "Symbol") -> list[str]:
    try:
        tables = pd.read_html(url)
        for table in tables:
            if symbol_col in table.columns:
                tickers = table[symbol_col].str.replace(".", "-", regex=False).tolist()
                if len(tickers) > 50:
                    return sorted(tickers)
            if "Ticker" in table.columns:
                tickers = table["Ticker"].str.replace(".", "-", regex=False).tolist()
                if len(tickers) > 50:
                    return sorted(tickers)
    except Exception:
        pass
    return []


# ── Ticker Universe ────────────────────────────────────────────────


@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_tickers() -> list[str]:
    result = _scrape_wiki_tickers(SP500_WIKI_URL, "Symbol")
    if len(result) > 400:
        return result
    return _sp500_fallback()


@st.cache_data(ttl=86400, show_spinner=False)
def get_sp400_tickers() -> list[str]:
    result = _scrape_wiki_tickers(SP400_WIKI_URL, "Symbol")
    if len(result) > 200:
        return result
    return _sp400_fallback()


@st.cache_data(ttl=86400, show_spinner=False)
def get_nasdaq100_tickers() -> list[str]:
    result = _scrape_wiki_tickers(NASDAQ100_WIKI_URL, "Ticker")
    if len(result) > 80:
        return result
    return _nasdaq100_fallback()


@st.cache_data(ttl=86400, show_spinner=False)
def get_broad_universe(min_market_cap_b: float = 10.0) -> list[str]:
    sp500 = get_sp500_tickers()
    sp400 = get_sp400_tickers()
    ndx100 = get_nasdaq100_tickers()
    supplemental = _get_supplemental_tickers()
    return sorted(set(sp500 + sp400 + ndx100 + supplemental))


def _get_supplemental_tickers() -> list[str]:
    return [
        "TSM", "BABA", "JD", "PDD", "BIDU", "NIO", "LI", "XPEV",
        "SHOP", "TD", "RY", "CNQ", "SU", "BN", "BAM", "SE", "MELI", "NU", "GRAB",
        "BX", "KKR", "APO", "ARES", "OWL",
        "SPOT", "SQ", "MSTR", "CELH", "DUOL", "CAVA",
        "RIVN", "LCID", "JOBY", "BILL", "PATH", "SNAP", "U", "PINS",
    ]


# ── Single Ticker Complete Fetch ───────────────────────────────────


def _fetch_single_complete(ticker: str, max_retries: int = 2) -> dict | None:
    """
    Fetch EVERYTHING for one ticker in one call: price history + fundamentals.
    This avoids the batch price download that gets rate-limited in bulk.
    """
    for attempt in range(max_retries):
        try:
            t = yf.Ticker(ticker)

            # Get info (fundamentals)
            info = t.info or {}
            if not info.get("marketCap"):
                return None

            # Get price history
            hist = t.history(period="1y")
            if hist.empty or len(hist) < 20:
                return None

            close = hist["Close"]
            current_price = float(close.iloc[-1])

            # Momentum
            momentum = _calc_momentum(close, current_price)

            # Earnings surprise
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

            # Analyst metrics
            target_price = info.get("targetMeanPrice")
            current = info.get("currentPrice") or info.get("previousClose")
            upside = None
            if target_price and current and current > 0:
                upside = round((target_price - current) / current, 4)

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
                "analyst_mean_target_upside": upside,
                "analyst_recommendation_score": info.get("recommendationMean"),
                "earnings_surprise_pct": surprise_pct,
                "analyst_count": info.get("numberOfAnalystOpinions", 0),
                "lastUpdated": datetime.now().isoformat(),
            }

        except Exception as e:
            err_str = str(e).lower()
            if "rate limit" in err_str or "too many" in err_str or "401" in err_str:
                wait = (3 * (2 ** attempt)) + random.uniform(0, 2)
                time.sleep(wait)
            else:
                return None

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


# ── Main Fetch Orchestrator ───────────────────────────────────────


def fetch_universe_data(
    tickers: list[str],
    min_market_cap_b: float = 10.0,
    progress_callback=None,
) -> dict[str, dict]:
    """
    Fetch one ticker at a time. Slow but reliable.
    NEVER loses cached data. Only overwrites on successful fetch.
    """
    results = _load_cache()

    # Only skip if cache is fresh AND we have 80%+ of universe
    target_size = len(tickers)
    have_ratio = len(results) / target_size if target_size > 0 else 0
    if _is_cache_fresh(_cache_path(FUNDAMENTALS_CACHE_FILE)) and have_ratio > 0.80:
        return _filter_by_market_cap(results, min_market_cap_b)

    # Build list of tickers that need fetching
    tickers_to_fetch = []
    for ticker in tickers:
        if ticker in results:
            last_updated = results[ticker].get("lastUpdated", "")
            try:
                updated_dt = datetime.fromisoformat(last_updated)
                if datetime.now() - updated_dt < timedelta(hours=CACHE_EXPIRY_HOURS):
                    continue
            except Exception:
                pass
        tickers_to_fetch.append(ticker)

    if not tickers_to_fetch:
        _save_cache(results)
        return _filter_by_market_cap(results, min_market_cap_b)

    total = len(tickers_to_fetch)
    new_successes = 0
    consecutive_failures = 0

    if progress_callback:
        progress_callback(0.05, f"Fetching {total} tickers ({len(results)} cached)...")

    for i, ticker in enumerate(tickers_to_fetch):
        data = _fetch_single_complete(ticker)

        if data:
            results[ticker] = data
            new_successes += 1
            consecutive_failures = 0
        else:
            consecutive_failures += 1

        # Progress update
        if progress_callback and (i + 1) % 5 == 0:
            pct = 0.05 + ((i + 1) / total) * 0.90
            progress_callback(
                min(pct, 0.95),
                f"{i + 1}/{total} checked ({new_successes} new, {len(results)} total)..."
            )

        # Save every 20 tickers so progress is never lost
        if (i + 1) % 20 == 0:
            _save_cache(results)

        # If we hit 10+ consecutive failures, we're rate-limited hard.
        # Back off for 30 seconds then continue.
        if consecutive_failures >= 10:
            if progress_callback:
                progress_callback(
                    min(0.05 + ((i + 1) / total) * 0.90, 0.95),
                    f"Rate limited. Pausing 30s... ({len(results)} total so far)"
                )
            time.sleep(30)
            consecutive_failures = 0
        else:
            # Normal delay between tickers
            time.sleep(0.8 + random.uniform(0, 0.5))

    # Final save
    _save_cache(results)

    if progress_callback:
        progress_callback(1.0, f"Done! {len(results)} total ({new_successes} new this run).")

    return _filter_by_market_cap(results, min_market_cap_b)


def _filter_by_market_cap(data: dict, min_cap_b: float) -> dict:
    min_cap = min_cap_b * 1e9
    return {
        k: v for k, v in data.items()
        if v.get("marketCap", 0) >= min_cap
    }


# ── Hardcoded Fallbacks ───────────────────────────────────────────


def _sp500_fallback() -> list[str]:
    return sorted([
        "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE",
        "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALK",
        "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN",
        "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", "APTV", "ARE", "ATO",
        "AVB", "AVGO", "AVY", "AWK", "AXP", "AZO", "BA", "BAC", "BAX", "BBWI",
        "BBY", "BDX", "BEN", "BF-B", "BG", "BIIB", "BIO", "BK", "BKNG", "BKR",
        "BLK", "BMY", "BR", "BRK-B", "BRO", "BSX", "BWA", "BXP", "C", "CAG",
        "CAH", "CARR", "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL", "CDNS",
        "CDW", "CE", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR", "CI", "CINF",
        "CL", "CLX", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP",
        "COF", "COO", "COP", "COST", "CPB", "CPRT", "CPT", "CRL", "CRM", "CSCO",
        "CSGP", "CSX", "CTAS", "CTRA", "CTSH", "CTVA", "CVS", "CVX", "CZR",
        "D", "DAL", "DD", "DE", "DG", "DGX", "DHI", "DHR", "DIS",
        "DLR", "DLTR", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN",
        "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EIX", "EL", "EMN", "EMR",
        "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ES", "ESS", "ETN", "ETR",
        "ETSY", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR", "F", "FANG", "FAST",
        "FCX", "FDS", "FDX", "FE", "FFIV", "FIS", "FISV", "FITB",
        "FMC", "FOX", "FOXA", "FRT", "FTNT", "FTV", "GD", "GE", "GEHC", "GEN",
        "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN",
        "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN", "HCA", "HSIC", "HST", "HSY",
        "HUM", "HWM", "IBM", "ICE", "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTC",
        "INTU", "INVH", "IP", "IQV", "IR", "IRM", "ISRG", "IT", "ITW",
        "IVZ", "J", "JBHT", "JCI", "JKHY", "JNJ", "JPM", "KDP",
        "KEY", "KEYS", "KHC", "KIM", "KLAC", "KMB", "KMI", "KMX", "KO", "KR",
        "L", "LDOS", "LEN", "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNC",
        "LNT", "LOW", "LRCX", "LULU", "LUV", "LVS", "LW", "LYB", "LYV", "MA",
        "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET",
        "META", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST", "MO",
        "MOH", "MOS", "MPC", "MPWR", "MRK", "MRNA", "MS", "MSCI", "MSFT",
        "MSI", "MTB", "MTCH", "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM",
        "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE",
        "NVDA", "NVR", "NWL", "NWS", "NWSA", "NXPI", "O", "ODFL", "OGN", "OKE",
        "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY", "PAYC", "PAYX", "PCAR",
        "PCG", "PEG", "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM",
        "PKG", "PLD", "PM", "PNC", "PNR", "PNW", "POOL", "PPG", "PPL",
        "PRU", "PSA", "PSX", "PTC", "PVH", "PWR", "PYPL", "QCOM", "QRVO", "RCL",
        "REG", "REGN", "RF", "RHI", "RJF", "RL", "RMD", "ROK", "ROL",
        "ROP", "ROST", "RSG", "RTX", "RVTY", "SBAC", "SBUX", "SCHW", "SEE", "SHW",
        "SJM", "SLB", "SNA", "SNPS", "SO", "SPG", "SPGI", "SRE", "STE", "STLD",
        "STT", "STX", "STZ", "SWK", "SWKS", "SYF", "SYK", "SYY", "T", "TAP",
        "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TFX", "TGT", "TMO", "TMUS",
        "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO",
        "TXN", "TXT", "TYL", "UAL", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS",
        "URI", "USB", "V", "VFC", "VICI", "VLO", "VMC", "VRSK", "VRSN", "VRTX",
        "VTR", "VTRS", "VZ", "WAB", "WAT", "WBD", "WDC", "WEC", "WELL",
        "WFC", "WHR", "WM", "WMB", "WMT", "WRB", "WST", "WTW", "WY",
        "WYNN", "XEL", "XOM", "XRAY", "XYL", "YUM", "ZBH", "ZBRA", "ZION", "ZTS",
        "COIN", "PLTR", "CRWD", "DASH", "DDOG", "SNOW", "NET", "ZS", "MDB",
        "TTD", "PANW", "ABNB", "HOOD", "RBLX", "ARM", "SMCI", "VST",
        "DECK", "AXON", "FICO", "GDDY", "HUBB", "TW", "GEV", "VLTO", "KVUE",
        "SOLV", "SW",
    ])


def _nasdaq100_fallback() -> list[str]:
    return sorted([
        "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
        "AMZN", "ANSS", "APP", "ARM", "ASML", "AVGO", "AZN", "BIIB", "BKNG", "BKR",
        "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COIN", "COST", "CPRT", "CRWD",
        "CSCO", "CSGP", "CTAS", "CTSH", "DASH", "DDOG", "DLTR", "DXCM", "EA", "EXC",
        "FANG", "FAST", "FTNT", "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX",
        "ILMN", "INTC", "INTU", "ISRG", "KDP", "KHC", "KLAC", "LIN", "LRCX", "LULU",
        "MAR", "MCHP", "MDB", "MDLZ", "MELI", "META", "MNST", "MRVL", "MSFT", "MU",
        "NFLX", "NVDA", "NXPI", "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD",
        "PEP", "PLTR", "PYPL", "QCOM", "REGN", "ROP", "ROST", "SBUX", "SMCI", "SNPS",
        "TEAM", "TMUS", "TSLA", "TTD", "TTWO", "TXN", "VRSK", "VRTX", "WBD", "WDAY",
        "ZS",
    ])


def _sp400_fallback() -> list[str]:
    return sorted([
        "ACGL", "ACM", "ACI", "AFRM", "AFG", "AGCO", "ALNY", "ALLY",
        "ARMK", "ARES", "ARW", "ATR", "AVTR",
        "BFAM", "BJ", "BLD", "BMRN", "BROS", "BURL",
        "CART", "CAVA", "CBSH", "CCK", "CHDN", "CHE", "CG", "CGNX",
        "CLH", "COHR", "COKE", "COOP", "CRUS",
        "CVNA", "CW", "CYBR",
        "DAR", "DINO", "DKS", "DOCU", "DOX", "DT", "DUOL",
        "ENTG", "ENSG", "EPRT", "ESI", "ESTC", "EVR", "EWBC",
        "EXP", "EXAS",
        "FIVE", "FIX", "FNB", "FND", "FNF", "FSLR", "FSS", "FTI",
        "GATX", "GBCI", "GLPI", "GMS", "GPK",
        "H", "HALO", "HGV", "HLI", "HOLX", "HP", "HQY", "HRB",
        "IAC", "IBKR", "ICL", "ICLR", "IDCC", "IDA",
        "INSP", "IOT", "IRTC", "ITT",
        "JEF", "JLL",
        "KNSL", "KNX",
        "LAMR", "LBRDA", "LBRDK", "LEA", "LFUS", "LNTH", "LSCC",
        "MANH", "MASI", "MEDP", "MIDD", "MKSI",
        "MORN", "MTDR", "MTN", "MTZ",
        "NBIX", "NEU", "NNN", "NOV", "NVT",
        "OLED", "OLN", "ORI", "OVV", "PCOR", "PCTY", "PEN", "PII",
        "PLNT", "PNFP", "PPC", "PSTG", "PRI",
        "RBA", "RBC", "RGA", "RGLD", "RNR",
        "SAM", "SAIA", "SBRA", "SCI", "SFM", "SKX", "SMAR", "SNV",
        "SSD", "ST", "SWN",
        "TNET", "TOST", "TPG", "TPL", "TREX", "TWLO",
        "UHAL", "UTHR",
        "VEEV", "VOYA", "VVV",
        "WAL", "WBS", "WCC", "WEN", "WEX", "WH", "WMS",
        "WSC", "WSM", "WPC",
        "YETI",
        "ZWS",
    ])


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
    return _fetch_single_complete(ticker, max_retries=2)
