"""
Scoring engine for the five-pillar quant grading system.
Ranks stocks by percentile within the universe, assigns letter grades,
and computes weighted composite scores.
"""

import numpy as np
import pandas as pd

from config import (
    PILLAR_METRICS,
    GRADE_PERCENTILE_MAP,
    GRADE_SCORES,
    OVERALL_RATING_MAP,
    DEFAULT_PILLAR_WEIGHTS,
)


def score_universe(
    data: dict[str, dict],
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Score all tickers across the five pillars.

    Args:
        data: Dict of {ticker: fundamentals_dict}
        weights: Dict of {pillar_name: weight}, must sum to 1.0

    Returns:
        DataFrame with columns for each metric, grade, pillar score, and composite.
    """
    if not data:
        return pd.DataFrame()

    weights = weights or DEFAULT_PILLAR_WEIGHTS

    # Build DataFrame from raw data
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "ticker"

    # Score each pillar
    pillar_scores = {}
    pillar_grades = {}
    metric_grades = {}

    for pillar_name, metrics in PILLAR_METRICS.items():
        pillar_metric_scores = []

        for yf_key, display_name, higher_is_better in metrics:
            if yf_key not in df.columns:
                continue

            col = pd.to_numeric(df[yf_key], errors="coerce")

            # Compute percentile rank
            if higher_is_better:
                pct = col.rank(pct=True, na_option="bottom") * 100
            else:
                # Lower is better: invert the rank
                pct = (1 - col.rank(pct=True, na_option="bottom")) * 100

            # Assign letter grade based on percentile
            grades = pct.apply(_percentile_to_grade)
            grade_nums = grades.map(GRADE_SCORES).fillna(1)

            metric_grades[f"{pillar_name}|{display_name}"] = grades
            pillar_metric_scores.append(grade_nums)

        if pillar_metric_scores:
            # Pillar score = average of its metric scores
            pillar_avg = pd.concat(pillar_metric_scores, axis=1).mean(axis=1)
            pillar_scores[pillar_name] = pillar_avg
            pillar_grades[pillar_name] = pillar_avg.apply(_score_to_grade)
        else:
            pillar_scores[pillar_name] = pd.Series(1, index=df.index)
            pillar_grades[pillar_name] = pd.Series("F", index=df.index)

    # Compute weighted composite score
    composite = pd.Series(0.0, index=df.index)
    for pillar_name, w in weights.items():
        if pillar_name in pillar_scores:
            composite += pillar_scores[pillar_name] * w

    # Normalize composite to sum of weights (in case some pillars are missing)
    total_weight = sum(w for p, w in weights.items() if p in pillar_scores)
    if total_weight > 0:
        composite = composite / total_weight * (sum(weights.values()))

    # Overall rating
    overall_rating = composite.apply(_score_to_rating)

    # Build result DataFrame
    result = df[["shortName", "sector", "industry", "marketCap", "currentPrice"]].copy()
    result["marketCapB"] = (result["marketCap"] / 1e9).round(1)

    # Add pillar scores and grades
    for pillar_name in PILLAR_METRICS:
        result[f"{pillar_name}_score"] = pillar_scores.get(pillar_name, 1).round(2)
        result[f"{pillar_name}_grade"] = pillar_grades.get(pillar_name, "F")

    # Add individual metric grades
    for key, grades in metric_grades.items():
        result[f"metric|{key}"] = grades

    # Add composite
    result["composite_score"] = composite.round(2)
    result["overall_rating"] = overall_rating

    # Sort by composite score descending
    result = result.sort_values("composite_score", ascending=False)

    return result


def _percentile_to_grade(pct: float) -> str:
    """Convert percentile to letter grade."""
    if pd.isna(pct):
        return "F"
    for grade, (low, high) in GRADE_PERCENTILE_MAP.items():
        if low <= pct < high:
            return grade
    # Edge case: exactly 100
    if pct >= 100:
        return "A+"
    return "F"


def _score_to_grade(score: float) -> str:
    """Convert numeric score (1-12) to letter grade."""
    if pd.isna(score):
        return "F"
    # Find closest grade
    best_grade = "F"
    best_diff = float("inf")
    for grade, num in GRADE_SCORES.items():
        diff = abs(score - num)
        if diff < best_diff:
            best_diff = diff
            best_grade = grade
    return best_grade


def _score_to_rating(score: float) -> str:
    """Convert composite score to overall rating."""
    if pd.isna(score):
        return "Hold"
    for rating, (low, high) in OVERALL_RATING_MAP.items():
        if low <= score <= high:
            return rating
    return "Hold"


def get_pillar_detail(ticker: str, scored_df: pd.DataFrame) -> dict:
    """
    Get detailed metric breakdown for a single ticker across all pillars.
    Returns dict of {pillar: [{metric_name, raw_value, grade}, ...]}.
    """
    if ticker not in scored_df.index:
        return {}

    row = scored_df.loc[ticker]
    detail = {}

    for pillar_name, metrics in PILLAR_METRICS.items():
        pillar_detail = []
        for yf_key, display_name, higher_is_better in metrics:
            raw_val = row.get(yf_key)
            grade_key = f"metric|{pillar_name}|{display_name}"
            grade = row.get(grade_key, "N/A")

            # Format the raw value
            if raw_val is not None and not (isinstance(raw_val, float) and np.isnan(raw_val)):
                if "margin" in display_name.lower() or "return" in display_name.lower() or "growth" in display_name.lower() or "upside" in display_name.lower() or "surprise" in display_name.lower():
                    formatted = f"{raw_val * 100:.1f}%" if isinstance(raw_val, (int, float)) and abs(raw_val) < 100 else f"{raw_val:.1f}%"
                elif "p/e" in display_name.lower() or "peg" in display_name.lower() or "ev" in display_name.lower() or "price" in display_name.lower():
                    formatted = f"{raw_val:.1f}x" if isinstance(raw_val, (int, float)) else str(raw_val)
                elif display_name.startswith("#"):
                    formatted = f"{int(raw_val)}" if isinstance(raw_val, (int, float)) else str(raw_val)
                else:
                    formatted = f"{raw_val:.2f}" if isinstance(raw_val, float) else str(raw_val)
            else:
                formatted = "N/A"

            pillar_detail.append({
                "metric": display_name,
                "value": formatted,
                "grade": grade if grade != "N/A" else "—",
                "higher_is_better": higher_is_better,
            })

        detail[pillar_name] = {
            "metrics": pillar_detail,
            "pillar_grade": row.get(f"{pillar_name}_grade", "N/A"),
            "pillar_score": row.get(f"{pillar_name}_score", 0),
        }

    return detail


def get_top_stocks(
    scored_df: pd.DataFrame,
    n: int = 25,
    sector: str | None = None,
    rating_filter: str | None = None,
) -> pd.DataFrame:
    """Get top N stocks by composite score with optional filters."""
    df = scored_df.copy()

    if sector and sector != "All":
        df = df[df["sector"] == sector]

    if rating_filter and rating_filter != "All":
        df = df[df["overall_rating"] == rating_filter]

    return df.head(n)
