# 📊 Quantitative Strategy Dashboard

A free, open-source stock scoring dashboard modeled on the Seeking Alpha Quant framework. Scores ~800-1,000 US equities across five fundamental pillars with adjustable weights, interactive drill-down, watchlist tracking, and side-by-side comparison.

## Five Scoring Pillars

| Pillar | What It Measures | Key Metrics |
|--------|-----------------|-------------|
| **Valuation** | Is the stock cheap or expensive? | Forward P/E, Trailing P/E, PEG, P/B, P/S, EV/EBITDA, EV/Revenue |
| **Growth** | Is the company growing? | Revenue Growth (QoQ & YoY), Earnings Growth (QoQ & YoY) |
| **Profitability** | Is the company profitable? | Gross Margin, Operating Margin, Net Margin, ROE, ROA |
| **Momentum** | Is the stock trending up? | 1/3/6/12-month returns, Price vs 50-day SMA, Price vs 200-day SMA |
| **EPS Revisions** | Are analysts upgrading estimates? | Mean Target Upside, Analyst Rec Score, Earnings Surprise, Analyst Count |

## Features

- **Screener**: Rank the full universe by composite score with sector and rating filters
- **Watchlist**: Pin tickers ordered by date added (oldest first), with expandable detail views
- **Stock Detail**: Full metric breakdown across all five pillars with radar chart visualization
- **Compare**: Side-by-side comparison of up to 5 stocks with radar overlay and bar charts
- **Adjustable Weights**: Slider controls to emphasize the pillars that matter to your strategy
- **Market Cap Filter**: Adjustable floor ($1B - $50B) to control universe size

## Tech Stack

- **Frontend**: Streamlit
- **Data**: yfinance (free, no API key required)
- **Scoring**: Percentile-based ranking within universe → letter grades (A+ through F)
- **Caching**: Local JSON cache with 12-hour expiry
- **Hosting**: Streamlit Community Cloud (free)

---

## Quick Start (Local Development)

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/quant-dashboard.git
cd quant-dashboard
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

> **First run note**: The initial data fetch pulls fundamentals for ~500+ tickers and may take 15-30 minutes. Subsequent loads use the cache and are near-instant.

---

## Deploy to Streamlit Community Cloud (Free)

### 1. Push to GitHub

Create a new GitHub repository and push this code:

```bash
git init
git add .
git commit -m "Initial commit - Quant Strategy Dashboard v1"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/quant-dashboard.git
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository: `YOUR_USERNAME/quant-dashboard`
5. Set the main file path to: `app.py`
6. Click **"Deploy"**

Your dashboard will be live at: `https://YOUR_USERNAME-quant-dashboard.streamlit.app`

### 3. Share with others

Send the Streamlit Cloud URL to anyone — they can view the dashboard in their browser with no account or setup required.

---

## Configuration

### Pillar Weights

Adjust weights in the sidebar to match your strategy. For example:
- **Value-focused**: Valuation 40%, Profitability 25%, Growth 15%, Momentum 10%, EPS Rev 10%
- **Growth-focused**: Growth 35%, Momentum 25%, EPS Rev 20%, Profitability 15%, Valuation 5%
- **Balanced (default)**: Equal 20% across all five pillars

### Market Cap Filter

Default floor is $10B (captures ~800-1,000 US stocks). Adjust via the sidebar slider:
- **$22.7B+**: S&P 500 eligible large caps only (~400-500)
- **$10B+**: Large + upper mid-cap (~800-1,000) ← **default**
- **$5B+**: Includes more mid-caps (~1,200-1,500)
- **$1B+**: Full small-cap coverage (~3,000+, slower to load)

### Data Refresh

Data is cached for 12 hours. Click "Refresh Data" in the sidebar to force a fresh pull. On Streamlit Cloud, the cache resets whenever the app restarts.

---

## Project Structure

```
quant-dashboard/
├── .streamlit/
│   └── config.toml          # Streamlit theme (dark mode)
├── app.py                    # Main dashboard application
├── config.py                 # Scoring config, weights, grade thresholds
├── data_fetcher.py           # yfinance data pulling and caching
├── scoring.py                # Five-pillar scoring engine
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Scoring Methodology

1. **Fetch fundamentals** for each ticker via yfinance
2. **Percentile rank** each metric within the universe (e.g., if a stock's Forward P/E is better than 85% of peers, it scores in the 85th percentile)
3. **Assign letter grades** based on percentile buckets (A+ = 95-100th, A = 85-95th, ..., F = 0-5th)
4. **Convert grades to numeric scores** (A+ = 12, A = 11, ..., F = 1)
5. **Average metric scores** within each pillar to get pillar scores
6. **Weighted average** of pillar scores → composite score
7. **Map composite score** to overall rating (Strong Buy / Buy / Hold / Sell / Strong Sell)

For valuation metrics (where lower is better), the percentile ranking is inverted so that cheaper stocks score higher.

---

## Limitations & Future Improvements

- **EPS Revisions**: yfinance has limited analyst revision data. A future version could integrate Financial Modeling Prep or Alpha Vantage for richer revision history.
- **Sector-relative scoring**: Current scoring is universe-wide. A future version could score within-sector for fairer comparison across industries.
- **Backtesting**: Not yet implemented. A future version could simulate historical returns of top-ranked stocks.
- **Fidelity CSV import**: Planned for v2 — upload your Fidelity positions export and overlay your portfolio on the dashboard.
- **Alerts**: Email or push notifications when a watchlist stock's rating changes.

---

## Disclaimer

This dashboard is for educational and informational purposes only. It is not financial advice. The scoring methodology is inspired by publicly described quantitative frameworks but is an independent implementation. Past performance does not guarantee future results. Always do your own research and consult a qualified financial advisor before making investment decisions.
