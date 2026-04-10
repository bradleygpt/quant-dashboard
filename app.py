"""
Quantitative Strategy Dashboard
Sector-relative scoring across five pillars.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from config import (
    DEFAULT_PILLAR_WEIGHTS,
    PILLAR_METRICS,
    GRADE_COLORS,
    RATING_COLORS,
    GRADE_SCORES,
    DEFAULT_MARKET_CAP_FLOOR_B,
    MIN_MARKET_CAP_FLOOR_B,
    MAX_MARKET_CAP_FLOOR_B,
)
from data_fetcher import (
    get_broad_universe,
    fetch_universe_data,
    fetch_single_ticker,
    load_watchlist,
    add_to_watchlist,
    remove_from_watchlist,
)
from scoring import (
    score_universe,
    get_pillar_detail,
    get_top_stocks,
    get_sector_stats,
)

# ── Page Config ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Quant Strategy Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────

st.markdown("""
<style>
    .grade-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.85em;
        text-align: center;
        min-width: 36px;
        color: #111;
    }
    .main-header {
        font-size: 1.8em;
        font-weight: 800;
        background: linear-gradient(90deg, #00D4AA, #00A3FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header { color: #888; font-size: 0.95em; margin-top: -8px; }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ─────────────────────────────────────────────

if "scored_df" not in st.session_state:
    st.session_state.scored_df = None
if "raw_data" not in st.session_state:
    st.session_state.raw_data = None
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None
if "compare_tickers" not in st.session_state:
    st.session_state.compare_tickers = []
if "weights" not in st.session_state:
    st.session_state.weights = DEFAULT_PILLAR_WEIGHTS.copy()
if "sector_relative" not in st.session_state:
    st.session_state.sector_relative = True


# ── Helpers ────────────────────────────────────────────────────────

def format_market_cap(cap_b: float) -> str:
    if cap_b >= 1000:
        return f"${cap_b / 1000:.1f}T"
    return f"${cap_b:.1f}B"


def make_radar_chart(pillar_scores: dict, ticker: str) -> go.Figure:
    categories = list(pillar_scores.keys())
    values = [pillar_scores[c] for c in categories]
    values.append(values[0])
    categories.append(categories[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill="toself",
        fillcolor="rgba(0, 212, 170, 0.2)",
        line=dict(color="#00D4AA", width=2), name=ticker,
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 12],
                tickvals=[3, 6, 9, 12], ticktext=["D", "C", "B", "A+"],
                gridcolor="#2a2f3e"),
            angularaxis=dict(gridcolor="#2a2f3e"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False, margin=dict(l=60, r=60, t=30, b=30),
        height=320, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0e0e0"),
    )
    return fig


def make_comparison_radar(tickers_data: dict) -> go.Figure:
    colors = ["#00D4AA", "#FF6B6B", "#4ECDC4", "#FFE66D", "#A8E6CF"]
    fig = go.Figure()
    pillars = list(PILLAR_METRICS.keys())
    for i, (ticker, scores) in enumerate(tickers_data.items()):
        values = [scores.get(p, 0) for p in pillars]
        values.append(values[0])
        cats = pillars + [pillars[0]]
        fig.add_trace(go.Scatterpolar(
            r=values, theta=cats, fill="toself",
            fillcolor=f"rgba({','.join(str(int(colors[i % len(colors)][j:j+2], 16)) for j in (1,3,5))}, 0.1)",
            line=dict(color=colors[i % len(colors)], width=2), name=ticker,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 12],
                tickvals=[3, 6, 9, 12], ticktext=["D", "C", "B", "A+"],
                gridcolor="#2a2f3e"),
            angularaxis=dict(gridcolor="#2a2f3e"),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=60, r=60, t=40, b=40), height=400,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Settings")

    st.markdown("---")
    st.markdown("### Market Cap Filter")
    market_cap_floor = st.slider(
        "Minimum Market Cap ($B)",
        min_value=MIN_MARKET_CAP_FLOOR_B, max_value=MAX_MARKET_CAP_FLOOR_B,
        value=DEFAULT_MARKET_CAP_FLOOR_B, step=1,
    )

    st.markdown("---")
    st.markdown("### Scoring Mode")
    sector_relative = st.toggle("Sector-Relative Scoring", value=True,
        help="When ON, stocks are ranked within their sector. When OFF, ranked against the entire universe.")
    st.session_state.sector_relative = sector_relative

    st.markdown("---")
    st.markdown("### Pillar Weights")
    st.caption("Adjust emphasis. Weights normalize automatically.")

    w_val = st.slider("Valuation", 0.0, 1.0, st.session_state.weights["Valuation"], 0.05, key="w_val")
    w_gro = st.slider("Growth", 0.0, 1.0, st.session_state.weights["Growth"], 0.05, key="w_gro")
    w_pro = st.slider("Profitability", 0.0, 1.0, st.session_state.weights["Profitability"], 0.05, key="w_pro")
    w_mom = st.slider("Momentum", 0.0, 1.0, st.session_state.weights["Momentum"], 0.05, key="w_mom")
    w_eps = st.slider("EPS Revisions", 0.0, 1.0, st.session_state.weights["EPS Revisions"], 0.05, key="w_eps")

    total_w = w_val + w_gro + w_pro + w_mom + w_eps
    if total_w > 0:
        st.session_state.weights = {
            "Valuation": w_val / total_w, "Growth": w_gro / total_w,
            "Profitability": w_pro / total_w, "Momentum": w_mom / total_w,
            "EPS Revisions": w_eps / total_w,
        }

    st.caption("Normalized:")
    for p, w in st.session_state.weights.items():
        st.caption(f"  {p}: {w:.0%}")

    st.markdown("---")
    st.markdown("### Add to Watchlist")
    new_ticker = st.text_input("Ticker Symbol", placeholder="e.g. AAPL").upper().strip()
    if st.button("Add to Watchlist", key="sidebar_add_wl") and new_ticker:
        add_to_watchlist(new_ticker)
        st.success(f"Added {new_ticker}")
        st.rerun()

    st.markdown("---")
    st.markdown("### Data")
    if st.button("Refresh Data", key="sidebar_refresh"):
        st.cache_data.clear()
        st.session_state.scored_df = None
        st.session_state.raw_data = None
        import os
        try:
            cache_file = os.path.join("data_cache", "fundamentals_cache.json")
            if os.path.exists(cache_file):
                os.utime(cache_file, (0, 0))
        except Exception:
            pass
        st.rerun()

    st.caption(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ── Main Content ───────────────────────────────────────────────────

st.markdown('<p class="main-header">Quantitative Strategy Dashboard</p>', unsafe_allow_html=True)
scoring_mode = "Sector-Relative" if st.session_state.sector_relative else "Universe-Wide"
st.markdown(f'<p class="sub-header">Five-pillar scoring ({scoring_mode}): Valuation . Growth . Profitability . Momentum . EPS Revisions</p>', unsafe_allow_html=True)

tab_screener, tab_watchlist, tab_detail, tab_compare = st.tabs([
    "Screener", "Watchlist", "Stock Detail", "Compare"
])

# ── Data Loading ───────────────────────────────────────────────────

@st.cache_data(ttl=43200, show_spinner=False)
def load_and_score(market_cap_b: float, weights_tuple: tuple, sector_rel: bool):
    weights = dict(zip(DEFAULT_PILLAR_WEIGHTS.keys(), weights_tuple))
    tickers = get_broad_universe(market_cap_b)
    if not tickers:
        tickers = []

    progress = st.progress(0, text="Loading data...")
    def update_progress(pct, msg):
        progress.progress(pct, text=msg)

    raw_data = fetch_universe_data(tickers, market_cap_b, update_progress)
    progress.empty()

    scored = score_universe(raw_data, weights, sector_relative=sector_rel)
    sector_stats = get_sector_stats(scored) if not scored.empty else {}
    return raw_data, scored, sector_stats

weights_tuple = tuple(st.session_state.weights.values())

try:
    raw_data, scored_df, sector_stats = load_and_score(
        market_cap_floor, weights_tuple, st.session_state.sector_relative
    )
    st.session_state.raw_data = raw_data
    st.session_state.scored_df = scored_df
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

if scored_df is None or scored_df.empty:
    st.warning("No data loaded yet. Click 'Refresh Data' in the sidebar.")
    st.stop()


# ── TAB: Screener ──────────────────────────────────────────────────

with tab_screener:
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sectors = ["All"] + sorted(scored_df["sector"].dropna().unique().tolist())
        selected_sector = st.selectbox("Sector", sectors)
    with col_f2:
        ratings = ["All", "Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
        selected_rating = st.selectbox("Rating Filter", ratings)
    with col_f3:
        top_n = st.selectbox("Show Top", [50, 100, 250, 500], index=3)

    filtered = get_top_stocks(scored_df, top_n, selected_sector, selected_rating)

    if filtered.empty:
        st.info("No stocks match the current filters.")
    else:
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Universe Size", f"{len(scored_df):,}")
        with col_s2:
            strong_buys = len(scored_df[scored_df["overall_rating"] == "Strong Buy"])
            st.metric("Strong Buys", strong_buys)
        with col_s3:
            st.metric("Avg Composite", f"{scored_df['composite_score'].mean():.1f}")
        with col_s4:
            st.metric("Showing", f"{len(filtered):,}")

        st.markdown("---")

        display_cols = ["shortName", "sector", "marketCapB", "currentPrice"]
        for pillar in PILLAR_METRICS:
            display_cols.append(f"{pillar}_grade")
        display_cols += ["composite_score", "overall_rating"]

        display_df = filtered[display_cols].copy()
        display_df.columns = [
            "Company", "Sector", "Mkt Cap ($B)", "Price",
            "Valuation", "Growth", "Profit", "Momentum", "EPS Rev",
            "Score", "Rating",
        ]

        st.dataframe(display_df, use_container_width=True, height=700,
            column_config={
                "Mkt Cap ($B)": st.column_config.NumberColumn(format="%.1f"),
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "Score": st.column_config.NumberColumn(format="%.1f"),
            },
        )

        st.markdown("---")
        col_det1, col_det2 = st.columns([3, 1])
        with col_det1:
            detail_ticker = st.selectbox(
                "Select ticker for quick preview",
                options=filtered.index.tolist(),
                format_func=lambda x: f"{x} -- {filtered.loc[x, 'shortName']}" if x in filtered.index else x,
                key="screener_detail_select",
            )
        with col_det2:
            if st.button("View Full Detail", key="screener_detail_btn"):
                st.session_state.selected_ticker = detail_ticker
            if st.button("Add to Watchlist", key="screener_wl_btn"):
                add_to_watchlist(detail_ticker)
                st.success(f"Added {detail_ticker}")

        if detail_ticker:
            detail = get_pillar_detail(detail_ticker, scored_df, sector_stats)
            if detail:
                st.markdown(f"### Quick Preview: {detail_ticker}")
                ticker_sector = scored_df.loc[detail_ticker, "sector"] if detail_ticker in scored_df.index else "Unknown"
                st.caption(f"Sector: {ticker_sector} | Grades are relative to sector peers")

                pillar_scores = {p: d["pillar_score"] for p, d in detail.items()}
                st.plotly_chart(make_radar_chart(pillar_scores, detail_ticker), use_container_width=True, key="screener_radar")

                for pillar_name, pillar_data in detail.items():
                    grade = pillar_data["pillar_grade"]
                    score = pillar_data["pillar_score"]

                    with st.expander(f"{pillar_name}  |  Grade: {grade}  |  Score: {score:.1f}/12"):
                        # Header row
                        hc1, hc2, hc3, hc4, hc5, hc6 = st.columns([2.5, 1.2, 0.8, 0.8, 1.2, 1.2])
                        with hc1:
                            st.markdown("**Metric**")
                        with hc2:
                            st.markdown("**Value**")
                        with hc3:
                            st.markdown("**Grade**")
                        with hc4:
                            st.markdown("**%ile**")
                        with hc5:
                            st.markdown("**Sector Avg**")
                        with hc6:
                            st.markdown("**A Threshold**")

                        for m in pillar_data["metrics"]:
                            mc1, mc2, mc3, mc4, mc5, mc6 = st.columns([2.5, 1.2, 0.8, 0.8, 1.2, 1.2])
                            with mc1:
                                direction = "higher better" if m["higher_is_better"] else "lower better"
                                st.markdown(f"{m['metric']}")
                                st.caption(direction)
                            with mc2:
                                st.markdown(f"**{m['value']}**")
                            with mc3:
                                g = m["grade"]
                                gc = GRADE_COLORS.get(g, "#666")
                                st.markdown(f'<span style="background:{gc};padding:2px 8px;border-radius:4px;font-weight:700;color:#111;">{g}</span>', unsafe_allow_html=True)
                            with mc4:
                                st.markdown(m["percentile"])
                            with mc5:
                                st.markdown(m["sector_avg"])
                            with mc6:
                                st.markdown(f'**{m["a_threshold"]}**')


# ── TAB: Watchlist ─────────────────────────────────────────────────

with tab_watchlist:
    watchlist = load_watchlist()

    if not watchlist:
        st.info("Your watchlist is empty. Add tickers from the sidebar or screener.")
    else:
        st.markdown(f"### Watchlist ({len(watchlist)} stocks)")
        st.caption("Ordered by date added (oldest first)")

        for entry in watchlist:
            ticker = entry["ticker"]
            date_added = entry["date_added"]

            if ticker in scored_df.index:
                row = scored_df.loc[ticker]
                rating = row.get("overall_rating", "N/A")
                r_color = RATING_COLORS.get(rating, "#666")
                composite = row.get("composite_score", 0)

                col1, col2, col3, col4, col5, col6 = st.columns([1.5, 2.5, 1.5, 1, 1.5, 0.8])
                with col1:
                    st.markdown(f"**{ticker}**")
                with col2:
                    st.markdown(row.get("shortName", ""))
                with col3:
                    st.markdown(f'<span style="background:{r_color};padding:2px 10px;border-radius:4px;font-weight:700;color:#111;">{rating}</span>', unsafe_allow_html=True)
                with col4:
                    st.markdown(f"**{composite:.1f}**")
                with col5:
                    st.caption(f"Added {date_added}")
                with col6:
                    if st.button("X", key=f"wl_rm_{ticker}"):
                        remove_from_watchlist(ticker)
                        st.rerun()

                with st.expander(f"View {ticker} Detail"):
                    detail = get_pillar_detail(ticker, scored_df, sector_stats)
                    if detail:
                        pillar_scores = {p: d["pillar_score"] for p, d in detail.items()}
                        st.plotly_chart(make_radar_chart(pillar_scores, ticker), use_container_width=True, key=f"wl_radar_{ticker}")
            else:
                col1, col2, col3 = st.columns([1.5, 4, 0.8])
                with col1:
                    st.markdown(f"**{ticker}**")
                with col2:
                    st.caption(f"Not in universe | Added {date_added}")
                with col3:
                    if st.button("X", key=f"wl_rm_{ticker}_oos"):
                        remove_from_watchlist(ticker)
                        st.rerun()

            st.markdown("---")


# ── TAB: Stock Detail ──────────────────────────────────────────────

with tab_detail:
    all_tickers = sorted(scored_df.index.tolist())
    default_idx = 0
    if st.session_state.selected_ticker and st.session_state.selected_ticker in all_tickers:
        default_idx = all_tickers.index(st.session_state.selected_ticker)

    selected = st.selectbox(
        "Select Ticker", all_tickers, index=default_idx,
        format_func=lambda x: f"{x} -- {scored_df.loc[x, 'shortName']}" if x in scored_df.index else x,
        key="detail_ticker_select",
    )

    if selected and selected in scored_df.index:
        row = scored_df.loc[selected]
        detail = get_pillar_detail(selected, scored_df, sector_stats)

        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
        with col_h1:
            st.markdown(f"## {selected}")
            st.caption(row.get("shortName", ""))
        with col_h2:
            st.metric("Price", f"${row.get('currentPrice', 0):.2f}")
        with col_h3:
            st.metric("Market Cap", format_market_cap(row.get("marketCapB", 0)))
        with col_h4:
            rating = row.get("overall_rating", "Hold")
            composite = row.get("composite_score", 0)
            st.metric("Composite Score", f"{composite:.1f} / 12")
            st.markdown(f'<span style="background:{RATING_COLORS.get(rating, "#666")};padding:4px 14px;border-radius:6px;font-weight:700;color:#111;font-size:1.1em;">{rating}</span>', unsafe_allow_html=True)

        st.markdown(f"**Sector:** {row.get('sector', 'N/A')}  |  **Industry:** {row.get('industry', 'N/A')}")
        st.caption("All grades are relative to sector peers" if st.session_state.sector_relative else "All grades are relative to the full universe")

        st.markdown("---")

        col_chart, col_grades = st.columns([1, 1])
        with col_chart:
            st.markdown("#### Pillar Overview")
            if detail:
                pillar_scores = {p: d["pillar_score"] for p, d in detail.items()}
                st.plotly_chart(make_radar_chart(pillar_scores, selected), use_container_width=True, key="detail_radar")

        with col_grades:
            st.markdown("#### Pillar Grades")
            for pillar_name in PILLAR_METRICS:
                grade = row.get(f"{pillar_name}_grade", "N/A")
                score_val = row.get(f"{pillar_name}_score", 0)
                gc = GRADE_COLORS.get(grade, "#666")
                bar_width = (score_val / 12) * 100
                st.markdown(f"""
                <div style="margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                        <span style="color:#ccc;">{pillar_name}</span>
                        <span style="background:{gc};padding:2px 10px;border-radius:4px;font-weight:700;color:#111;">{grade}</span>
                    </div>
                    <div style="background:#2a2f3e;border-radius:4px;height:8px;width:100%;">
                        <div style="background:{gc};border-radius:4px;height:8px;width:{bar_width}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Full Metric Breakdown")
        st.caption("Click any pillar to see all inputs with sector context.")

        if detail:
            for pillar_name, pillar_data in detail.items():
                grade = pillar_data["pillar_grade"]
                score_val = pillar_data["pillar_score"]

                with st.expander(f"{pillar_name}  |  Grade: {grade}  |  Score: {score_val:.1f} / 12", expanded=False):
                    hc1, hc2, hc3, hc4, hc5, hc6 = st.columns([2.5, 1.2, 0.8, 0.8, 1.2, 1.2])
                    with hc1:
                        st.markdown("**Metric**")
                    with hc2:
                        st.markdown("**Value**")
                    with hc3:
                        st.markdown("**Grade**")
                    with hc4:
                        st.markdown("**%ile**")
                    with hc5:
                        st.markdown("**Sector Avg**")
                    with hc6:
                        st.markdown("**A Threshold**")

                    st.markdown("---")

                    for m in pillar_data["metrics"]:
                        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns([2.5, 1.2, 0.8, 0.8, 1.2, 1.2])
                        with mc1:
                            st.markdown(m["metric"])
                            st.caption("higher better" if m["higher_is_better"] else "lower better")
                        with mc2:
                            st.markdown(f"**{m['value']}**")
                        with mc3:
                            g = m["grade"]
                            gc = GRADE_COLORS.get(g, "#666")
                            st.markdown(f'<span style="background:{gc};padding:2px 8px;border-radius:4px;font-weight:700;color:#111;">{g}</span>', unsafe_allow_html=True)
                        with mc4:
                            st.markdown(m["percentile"])
                        with mc5:
                            st.markdown(m["sector_avg"])
                        with mc6:
                            st.markdown(f'**{m["a_threshold"]}**')

        st.markdown("---")
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button(f"Add {selected} to Watchlist", key="detail_add_wl"):
                add_to_watchlist(selected)
                st.success(f"Added {selected}")
        with col_act2:
            if st.button(f"Add {selected} to Compare", key="detail_add_cmp"):
                if selected not in st.session_state.compare_tickers:
                    st.session_state.compare_tickers.append(selected)
                    st.success(f"Added {selected} to comparison")


# ── TAB: Compare ───────────────────────────────────────────────────

with tab_compare:
    st.markdown("### Side-by-Side Comparison")

    compare_options = sorted(scored_df.index.tolist())
    selected_compare = st.multiselect(
        "Select tickers (2-5)", compare_options,
        default=st.session_state.compare_tickers[:5], max_selections=5,
        format_func=lambda x: f"{x} -- {scored_df.loc[x, 'shortName']}" if x in scored_df.index else x,
        key="compare_multiselect",
    )
    st.session_state.compare_tickers = selected_compare

    if len(selected_compare) < 2:
        st.info("Select at least 2 tickers to compare.")
    else:
        tickers_radar_data = {}
        for t in selected_compare:
            detail = get_pillar_detail(t, scored_df, sector_stats)
            if detail:
                tickers_radar_data[t] = {p: d["pillar_score"] for p, d in detail.items()}

        if tickers_radar_data:
            st.plotly_chart(make_comparison_radar(tickers_radar_data), use_container_width=True, key="compare_radar")

        st.markdown("#### Scores & Grades")
        comp_rows = []
        for t in selected_compare:
            if t in scored_df.index:
                r = scored_df.loc[t]
                row_data = {"Ticker": t, "Company": r.get("shortName", "")}
                for p in PILLAR_METRICS:
                    row_data[p] = f"{r.get(f'{p}_grade', 'N/A')} ({r.get(f'{p}_score', 0):.1f})"
                row_data["Composite"] = f"{r.get('composite_score', 0):.1f}"
                row_data["Rating"] = r.get("overall_rating", "N/A")
                comp_rows.append(row_data)

        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        st.markdown("#### Pillar Score Comparison")
        bar_data = []
        for t in selected_compare:
            if t in scored_df.index:
                r = scored_df.loc[t]
                for p in PILLAR_METRICS:
                    bar_data.append({"Ticker": t, "Pillar": p, "Score": r.get(f"{p}_score", 0)})

        if bar_data:
            bar_df = pd.DataFrame(bar_data)
            fig = px.bar(bar_df, x="Pillar", y="Score", color="Ticker", barmode="group",
                color_discrete_sequence=["#00D4AA", "#FF6B6B", "#4ECDC4", "#FFE66D", "#A8E6CF"])
            fig.update_layout(
                yaxis=dict(range=[0, 12], title="Score (0-12)"), xaxis=dict(title=""),
                height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0e0"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig, use_container_width=True, key="compare_bar")


# ── Footer ─────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Quantitative Strategy Dashboard v2.0 | Data via yfinance | "
    "Sector-relative scoring inspired by Seeking Alpha Quant framework | "
    "Not financial advice"
)
