"""Translate the day-ahead price forecast into a directional prompt-curve view.

A trader wants a decision, not just numbers: buy (LONG) expecting prices to
rise, sell (SHORT) expecting them to fall, or stand aside (NO VIEW) when the
signal is too weak. This module compares the model's average forecast against a
recent trailing average of realised prices (a transparent proxy for the live
front-week prompt curve) and emits a view with reasoning and an invalidation
condition.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
PREDICTIONS_PATH = ROOT / "predictions.csv"

# Decision threshold: deviation must exceed +/- this to justify a position.
THRESHOLD_PCT = 5.0
# Trailing window used as the reference / proxy for the front-week curve.
TRAILING_DAYS = 14

DATA_LIMITATION = (
    "The reference level is a 14-day trailing average of realised day-ahead "
    "prices, used as a transparent, documented proxy for the live quoted "
    "front-week prompt curve. In a real trading seat the forecast average would "
    "be compared against the broker-screen front-week price; that feed is not "
    "available in this environment, so the trailing average stands in for it."
)


def compute_curve_view() -> Dict:
    """Compute the prompt-curve view, save it to outputs/curve_view.json, and
    return it as a dict."""
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # --- 1. Forecast average over the whole prediction window ---
    preds = pd.read_csv(PREDICTIONS_PATH)
    preds["datetime"] = pd.to_datetime(preds["datetime"], utc=True)
    forecast_avg = float(preds["y_pred"].mean())
    test_start = preds["datetime"].min()

    # --- 2. Reference level: 14-day trailing avg of actual price, strictly
    #        before the test window starts ---
    merged = pd.read_csv(DATA_DIR / "merged.csv")
    merged["startTime"] = pd.to_datetime(merged["startTime"], utc=True)
    window_start = test_start - pd.Timedelta(days=TRAILING_DAYS)
    trailing = merged[
        (merged["startTime"] >= window_start)
        & (merged["startTime"] < test_start)
    ]
    reference_level = float(trailing["price_gbp_mwh"].dropna().mean())

    # --- 3. Deviation ---
    deviation_pct = (forecast_avg - reference_level) / reference_level * 100.0

    # --- 4. Decision rule ---
    if deviation_pct > THRESHOLD_PCT:
        view = "LONG"
    elif deviation_pct < -THRESHOLD_PCT:
        view = "SHORT"
    else:
        view = "NO VIEW"

    # --- 5. Plain-English explanation ---
    f, r, d = forecast_avg, reference_level, deviation_pct
    if view == "LONG":
        reasoning = (
            f"The model's average forecast of GBP {f:.2f}/MWh is {d:.1f}% above "
            f"the recent {TRAILING_DAYS}-day trailing average of GBP {r:.2f}"
            f"/MWh, so the front-week prompt contract looks cheap relative to "
            f"where the model expects prices to sit; a trader would buy prompt "
            f"now expecting to sell higher later."
        )
        invalidation = (
            "A sharp upward revision to wind forecasts or a significant fall in "
            "demand forecasts would cut net demand and push prices back down; "
            "close or reverse the long if either materialises."
        )
    elif view == "SHORT":
        reasoning = (
            f"The model's average forecast of GBP {f:.2f}/MWh is {abs(d):.1f}% "
            f"below the recent {TRAILING_DAYS}-day trailing average of GBP "
            f"{r:.2f}/MWh, so the front-week prompt contract looks expensive "
            f"relative to where the model expects prices to sit; a trader would "
            f"sell prompt now expecting to buy back lower later."
        )
        invalidation = (
            "A sharp downward revision to wind forecasts or a significant rise "
            "in demand forecasts would raise net demand and push prices back "
            "up; close or reverse the short if either materialises."
        )
    else:
        reasoning = (
            f"The model's average forecast of GBP {f:.2f}/MWh is only {d:.1f}% "
            f"from the recent {TRAILING_DAYS}-day trailing average of GBP "
            f"{r:.2f}/MWh, inside the +/-{THRESHOLD_PCT:.0f}% threshold, so the "
            f"signal is not strong enough to justify a position given model "
            f"uncertainty."
        )
        invalidation = (
            f"Conditions would need to shift materially -- net demand moving the "
            f"forecast more than {THRESHOLD_PCT:.0f}% away from recent levels -- "
            f"before a long or short position is justified."
        )

    result = {
        "view": view,
        "forecast_avg_gbp_mwh": round(forecast_avg, 2),
        "reference_level_gbp_mwh": round(reference_level, 2),
        "deviation_pct": round(deviation_pct, 2),
        "reasoning": reasoning,
        "invalidation": invalidation,
        "data_limitation": DATA_LIMITATION,
    }

    (OUTPUTS_DIR / "curve_view.json").write_text(
        json.dumps(result, indent=2) + "\n")
    return result


def print_curve_view(result: Dict) -> None:
    """Print the curve view clearly to the terminal."""
    print("\n" + "=" * 64)
    print("PROMPT-CURVE VIEW (front-week)")
    print("=" * 64)
    print(f"View:            {result['view']}")
    print(f"Forecast avg:    GBP {result['forecast_avg_gbp_mwh']:.2f}/MWh")
    print(f"Reference level: GBP {result['reference_level_gbp_mwh']:.2f}/MWh "
          f"(14-day trailing avg)")
    print(f"Deviation:       {result['deviation_pct']:.2f}%")
    print(f"Reasoning:       {result['reasoning']}")
    print(f"Invalidation:    {result['invalidation']}")
    print(f"Data limitation: {result['data_limitation']}")
    print("=" * 64)


if __name__ == "__main__":
    print_curve_view(compute_curve_view())
