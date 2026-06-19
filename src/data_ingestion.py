"""Data ingestion for GB day-ahead electricity price forecasting.

Pulls 90 days (ending yesterday) of real public data from three Elexon BMRS
API v1 endpoints, cleans each source, merges them into one hourly table, runs
quality checks, and writes the result to data/merged.csv.

All three endpoints are public and require no API key.
Base URL: https://data.elexon.co.uk/bmrs/api/v1
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"

# Project paths (resolved relative to this file so the script runs from anywhere)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

# HTTP behaviour
REQUEST_TIMEOUT = 60  # seconds
MAX_RETRIES = 4
RETRY_BACKOFF = 2.0  # seconds, multiplied by attempt number


# --------------------------------------------------------------------------- #
# Date range helpers
# --------------------------------------------------------------------------- #
def get_date_range():
    """Return (start_date, end_date) as UTC dates for the 90 days ending
    yesterday. The range is inclusive on both ends (90 calendar days)."""
    today = datetime.now(timezone.utc).date()
    end_date = today - timedelta(days=1)          # yesterday
    start_date = end_date - timedelta(days=89)    # 90 days inclusive
    return start_date, end_date


def _iso(dt: datetime) -> str:
    """Format a UTC datetime as an ISO 8601 string the Elexon API accepts."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Low-level request helper
# --------------------------------------------------------------------------- #
def _get(endpoint: str, params: dict) -> List[dict]:
    """GET an Elexon endpoint and return the list under the 'data' key.

    Retries transient failures with linear backoff. Returns an empty list if
    the request ultimately fails or the response has no data array.
    """
    url = f"{BASE_URL}{endpoint}"
    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            # Elexon responses wrap rows in {"data": [...]}; some return a bare list
            if isinstance(payload, dict):
                return payload.get("data", []) or []
            if isinstance(payload, list):
                return payload
            return []
        except (requests.RequestException, ValueError) as err:
            last_err = err
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    print(f"  ! request failed after {MAX_RETRIES} attempts "
          f"({endpoint}, params={params}): {last_err}")
    return []


# --------------------------------------------------------------------------- #
# Source 1: Day-ahead price (market index)
# --------------------------------------------------------------------------- #
def fetch_price(start_date, end_date) -> pd.DataFrame:
    """Pull the GB market index price (APXMIDP) in 7-day chunks and aggregate
    to hourly. Returns columns: startTime, price_gbp_mwh."""
    endpoint = "/balancing/pricing/market-index"
    rows: List[dict] = []

    chunk_start = start_date
    while chunk_start <= end_date:
        # inclusive 7-day window; 'to' is the day after the last day in the chunk
        chunk_end = min(chunk_start + timedelta(days=6), end_date)
        params = {
            "from": _iso(datetime(chunk_start.year, chunk_start.month,
                                  chunk_start.day, 0, 0, 0, tzinfo=timezone.utc)),
            "to": _iso(datetime(chunk_end.year, chunk_end.month, chunk_end.day,
                                0, 0, 0, tzinfo=timezone.utc) + timedelta(days=1)),
            # APXMIDP carries populated GB market-index prices; the originally
            # specified N2EXMIDP provider returns all-zero prices here.
            "dataProviders": "APXMIDP",
            "format": "json",
        }
        print(f"  price chunk {chunk_start} -> {chunk_end}")
        rows.extend(_get(endpoint, params))
        chunk_start = chunk_end + timedelta(days=1)

    if not rows:
        return pd.DataFrame(columns=["startTime", "price_gbp_mwh"])

    df = pd.DataFrame(rows)
    df = df[["startTime", "price"]].rename(columns={"price": "price_gbp_mwh"})
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    df["price_gbp_mwh"] = pd.to_numeric(df["price_gbp_mwh"], errors="coerce")

    # Half-hourly -> hourly by averaging the two rows within each clock hour
    df["startTime"] = df["startTime"].dt.floor("h")
    df = (df.groupby("startTime", as_index=False)["price_gbp_mwh"]
            .mean())
    return df


# --------------------------------------------------------------------------- #
# Source 2: Wind generation forecast (WINDFOR)
# --------------------------------------------------------------------------- #
def _day_window(day):
    """Build a single-UTC-day publishDateTime window (the Elexon dataset
    endpoints cap the publish window at 1 day)."""
    start = datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=timezone.utc)
    return _iso(start), _iso(end)


def fetch_wind(start_date, end_date) -> pd.DataFrame:
    """Pull the NGESO wind forecast one publish-day at a time, then keep the
    most recent forecast per startTime. Returns columns: startTime, wind_mw.

    NOTE: the live WINDFOR endpoint ignores a bare ``publishTime`` parameter and
    only ever returns the latest snapshot, so we filter by a 1-day
    ``publishDateTimeFrom``/``publishDateTimeTo`` window instead. This still
    honours the per-day loop and the most-recent-publishTime dedup rule.
    """
    endpoint = "/datasets/WINDFOR"
    rows: List[dict] = []

    day = start_date
    while day <= end_date:
        pub_from, pub_to = _day_window(day)
        params = {"publishDateTimeFrom": pub_from, "publishDateTimeTo": pub_to,
                  "format": "json"}
        rows.extend(_get(endpoint, params))
        day += timedelta(days=1)
    print(f"  wind: {len(rows)} raw rows")

    if not rows:
        return pd.DataFrame(columns=["startTime", "wind_mw"])

    df = pd.DataFrame(rows)
    df = df[["startTime", "generation", "publishTime"]].rename(
        columns={"generation": "wind_mw"})
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    df["publishTime"] = pd.to_datetime(df["publishTime"], utc=True)
    df["wind_mw"] = pd.to_numeric(df["wind_mw"], errors="coerce")

    # Already hourly. For duplicate startTimes keep the most recent publishTime.
    df = (df.sort_values("publishTime")
            .drop_duplicates(subset="startTime", keep="last"))
    df = df[["startTime", "wind_mw"]].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Source 3: National demand forecast (NDF)
# --------------------------------------------------------------------------- #
def fetch_demand(start_date, end_date) -> pd.DataFrame:
    """Pull the national demand forecast one publish-day at a time.
    Returns columns: startTime, demand_mw."""
    endpoint = "/datasets/NDF"
    rows: List[dict] = []

    day = start_date
    while day <= end_date:
        pub_from, pub_to = _day_window(day)
        params = {"publishDateTimeFrom": pub_from, "publishDateTimeTo": pub_to,
                  "format": "json"}
        rows.extend(_get(endpoint, params))
        day += timedelta(days=1)
    print(f"  demand: {len(rows)} raw rows")

    if not rows:
        return pd.DataFrame(columns=["startTime", "demand_mw"])

    df = pd.DataFrame(rows)
    # Keep only the GB-wide boundary rows
    if "boundary" in df.columns:
        df = df[df["boundary"] == "N"]
    df = df[["startTime", "demand", "publishTime"]].rename(
        columns={"demand": "demand_mw"})
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    df["publishTime"] = pd.to_datetime(df["publishTime"], utc=True)
    df["demand_mw"] = pd.to_numeric(df["demand_mw"], errors="coerce")

    # A full publish-day window returns several re-forecasts per settlement
    # period; keep the most recent publishTime for each half-hourly startTime.
    df = (df.sort_values("publishTime")
            .drop_duplicates(subset="startTime", keep="last"))

    # Half-hourly -> hourly by averaging the two rows within each clock hour
    df["startTime"] = df["startTime"].dt.floor("h")
    df = (df.groupby("startTime", as_index=False)["demand_mw"]
            .mean())
    return df


# --------------------------------------------------------------------------- #
# Merge + QA
# --------------------------------------------------------------------------- #
def merge_sources(price: pd.DataFrame, wind: pd.DataFrame,
                  demand: pd.DataFrame) -> pd.DataFrame:
    """Merge the three sources on startTime into one hourly table."""
    merged = (price.merge(wind, on="startTime", how="outer")
                   .merge(demand, on="startTime", how="outer"))
    merged = merged.sort_values("startTime").reset_index(drop=True)
    return merged


def run_qa(df: pd.DataFrame) -> str:
    """Build the QA report text for the merged dataset."""
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("QA LOG - merged GB day-ahead price dataset")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("=" * 60)

    lines.append(f"Total rows: {len(df)}")

    if len(df):
        lines.append(f"Date range: {df['startTime'].min()} "
                     f"to {df['startTime'].max()}")
    else:
        lines.append("Date range: (empty dataset)")

    lines.append("")
    lines.append("Missing values per column:")
    for col in df.columns:
        lines.append(f"  {col}: {int(df[col].isna().sum())}")

    dup = int(df["startTime"].duplicated().sum()) if "startTime" in df else 0
    lines.append("")
    lines.append(f"Duplicate startTime values: {dup}")

    neg_wind = int((df["wind_mw"] < 0).sum()) if "wind_mw" in df else 0
    neg_demand = int((df["demand_mw"] < 0).sum()) if "demand_mw" in df else 0
    lines.append(f"Rows with negative wind_mw: {neg_wind}")
    lines.append(f"Rows with negative demand_mw: {neg_demand}")

    if "price_gbp_mwh" in df:
        out_of_range = int(((df["price_gbp_mwh"] < -500) |
                            (df["price_gbp_mwh"] > 3000)).sum())
    else:
        out_of_range = 0
    lines.append(f"Rows with price_gbp_mwh < -500 or > 3000: {out_of_range}")
    lines.append("=" * 60)

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def ingest() -> pd.DataFrame:
    """Run the full ingestion pipeline and persist outputs. Returns the merged
    DataFrame."""
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    start_date, end_date = get_date_range()
    print(f"Pulling data for {start_date} -> {end_date} (UTC)")

    print("Source 1/3: day-ahead price (market-index)")
    price = fetch_price(start_date, end_date)
    print(f"  -> {len(price)} hourly price rows")

    print("Source 2/3: wind forecast (WINDFOR)")
    wind = fetch_wind(start_date, end_date)
    print(f"  -> {len(wind)} hourly wind rows")

    print("Source 3/3: demand forecast (NDF)")
    demand = fetch_demand(start_date, end_date)
    print(f"  -> {len(demand)} hourly demand rows")

    print("Merging sources...")
    merged = merge_sources(price, wind, demand)

    qa_text = run_qa(merged)
    qa_path = OUTPUTS_DIR / "qa_log.txt"
    qa_path.write_text(qa_text + "\n")
    print(f"QA log written to {qa_path}")

    merged_path = DATA_DIR / "merged.csv"
    merged.to_csv(merged_path, index=False)
    print(f"Merged dataset written to {merged_path}")

    return merged


if __name__ == "__main__":
    ingest()
