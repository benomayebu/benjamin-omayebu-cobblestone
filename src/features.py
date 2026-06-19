"""Cleaning and feature engineering for the GB price forecasting pipeline.

Takes the merged hourly dataset produced by ``data_ingestion`` and turns it into
a model-ready feature table.
"""

from __future__ import annotations

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the merged dataset.

    - Drop rows with a missing ``price_gbp_mwh`` (price is the target; a row
      without it cannot be used for training or testing).
    - Fill missing ``demand_mw`` by time-based linear interpolation (a small gap
      is better filled than dropped).
    """
    df = df.copy()
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    df = df.sort_values("startTime").reset_index(drop=True)

    # Price is the target -> drop rows where it is missing.
    df = df.dropna(subset=["price_gbp_mwh"]).reset_index(drop=True)

    # Fill the small number of missing demand values via linear interpolation
    # across actual timestamps (robust to any gaps left by dropped rows).
    df = df.set_index("startTime")
    df["demand_mw"] = df["demand_mw"].interpolate(method="time",
                                                  limit_direction="both")
    df = df.reset_index()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, net-demand and price-lag features, then drop the leading
    rows that have no 24h/168h price history to look back to."""
    df = df.copy()
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    df = df.sort_values("startTime").reset_index(drop=True)

    # Calendar features
    df["hour_of_day"] = df["startTime"].dt.hour
    df["day_of_week"] = df["startTime"].dt.dayofweek          # Mon=0 .. Sun=6
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Residual demand after wind is used first (key price driver)
    df["net_demand_mw"] = df["demand_mw"] - df["wind_mw"]

    # Price lags by EXACT clock time (not positional shift), so gaps left by
    # dropped rows don't corrupt the lookup. Missing match -> NaN.
    price_by_time = df.set_index("startTime")["price_gbp_mwh"]
    df["price_lag_24h"] = (
        df["startTime"] - pd.Timedelta(hours=24)).map(price_by_time)
    df["price_lag_168h"] = (
        df["startTime"] - pd.Timedelta(hours=168)).map(price_by_time)

    # The earliest rows have no week-ago / day-ago price -> drop them.
    df = df.dropna(subset=["price_lag_24h", "price_lag_168h"]).reset_index(
        drop=True)
    return df
