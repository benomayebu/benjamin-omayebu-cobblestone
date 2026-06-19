"""Train/test split, baseline + improved models, validation and outputs.

The split is strictly time-based (never shuffled): the final 7 days are held out
as the test set so the model can never peek at future prices during training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "figures"
OUTPUTS_DIR = ROOT / "outputs"
PREDICTIONS_PATH = ROOT / "predictions.csv"

TARGET = "price_gbp_mwh"
FEATURES = [
    "hour_of_day", "day_of_week", "is_weekend",
    "wind_mw", "demand_mw", "net_demand_mw",
    "price_lag_24h", "price_lag_168h",
]
TEST_DAYS = 7


def time_split(df: pd.DataFrame, test_days: int = TEST_DAYS
               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the final ``test_days`` chronologically as the test set.
    Order is preserved -- no shuffling."""
    df = df.sort_values("startTime").reset_index(drop=True)
    cutoff = df["startTime"].max() - pd.Timedelta(days=test_days)
    train = df[df["startTime"] <= cutoff].reset_index(drop=True)
    test = df[df["startTime"] > cutoff].reset_index(drop=True)
    return train, test


def _metrics(y_true, y_pred) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"MAE": mae, "RMSE": rmse}


def run_models(df: pd.DataFrame) -> Dict:
    """Fit the baseline and improved models, evaluate on the held-out test set,
    write all outputs, and return a results dictionary."""
    FIGURES_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    train, test = time_split(df)

    # --- Baseline: seasonal naive (price one week ago at the same hour) ---
    baseline_pred = test["price_lag_168h"].to_numpy()
    baseline_metrics = _metrics(test[TARGET], baseline_pred)

    # --- Improved: gradient-boosted trees ---
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(train[FEATURES], train[TARGET])
    improved_pred = model.predict(test[FEATURES])
    improved_metrics = _metrics(test[TARGET], improved_pred)

    beat_baseline = improved_metrics["MAE"] < baseline_metrics["MAE"]

    # --- Validation plot ---
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test["startTime"], test[TARGET], label="Actual",
            color="black", linewidth=1.6)
    ax.plot(test["startTime"], baseline_pred, label="Baseline (seasonal naive)",
            color="tab:orange", linewidth=1.2, alpha=0.85)
    ax.plot(test["startTime"], improved_pred,
            label="Improved (HistGradientBoosting)",
            color="tab:blue", linewidth=1.2, alpha=0.85)
    ax.set_title("GB day-ahead price - test set validation (final 7 days)")
    ax.set_xlabel("Delivery hour (UTC)")
    ax.set_ylabel("Price (GBP/MWh)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "validation.png", dpi=120)
    plt.close(fig)

    # --- Metrics JSON ---
    metrics_payload = {
        "baseline_seasonal_naive": baseline_metrics,
        "improved_hist_gradient_boosting": improved_metrics,
        "improved_beats_baseline": bool(beat_baseline),
    }
    (OUTPUTS_DIR / "validation_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2) + "\n")

    # --- Out-of-sample predictions (improved model) ---
    predictions = pd.DataFrame({
        "datetime": test["startTime"].map(lambda t: t.isoformat()),
        "y_pred": improved_pred,
    })
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    return {
        "n_train": len(train),
        "n_test": len(test),
        "test_start": test["startTime"].min(),
        "test_end": test["startTime"].max(),
        "baseline": baseline_metrics,
        "improved": improved_metrics,
        "improved_beats_baseline": beat_baseline,
    }
