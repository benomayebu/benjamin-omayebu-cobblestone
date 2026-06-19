# GB Power Day-Ahead Price Forecast — Benjamin Omayebu

Python 3.9 or above is required.

## Market and Option

- **Market:** GB (Great Britain)
- **Option:** A — next-day hourly Day-Ahead prices
- **Out-of-sample window:** the last 7 days of the pulled data range — 168 hourly predictions saved in `predictions.csv`.

## How to Run

1. Clone the repository:

   ```
   git clone https://github.com/benomayebu/benjamin-omayebu-cobblestone.git
   ```

2. Navigate into the project folder:

   ```
   cd benjamin-omayebu-cobblestone/benjamin_omayebu
   ```

3. Install dependencies: `pip install -r requirements.txt`
4. The pipeline fetches all data live from the Elexon public API, which requires **no API key**.
5. For the AI commentary step only, create a `.env` file in the project root containing `ANTHROPIC_API_KEY=your-key`. If this file is not present the pipeline still runs and skips only the commentary step.
6. Run `python main.py`. That single command runs the full pipeline end to end and produces all outputs.

## Data Sources

All three endpoints are public and require **no API key**. Base URL: `https://data.elexon.co.uk/bmrs/api/v1`

- **Day-ahead price** (Market Index, `APXMIDP` provider): `https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/market-index`
- **Wind generation forecast** (WINDFOR): `https://data.elexon.co.uk/bmrs/api/v1/datasets/WINDFOR`
- **National demand forecast** (NDF): `https://data.elexon.co.uk/bmrs/api/v1/datasets/NDF`

## What the Pipeline Produces

- `data/merged.csv` — the merged hourly dataset: one row per clock hour with day-ahead price, wind forecast, and demand forecast.
- `outputs/qa_log.txt` — data quality checks (row count, date range, missing values, duplicates, negatives, out-of-range prices).
- `predictions.csv` — the 168 out-of-sample hourly predictions from the improved model, with columns `datetime` (ISO 8601) and `y_pred` (GBP/MWh).
- `figures/validation.png` — the test-set chart showing actual price, baseline prediction, and improved-model prediction.
- `outputs/validation_metrics.json` — MAE and RMSE for both the baseline and the improved model.
- `outputs/curve_view.json` — the directional prompt-curve trading view (LONG / SHORT / NO VIEW) with reasoning, invalidation condition, and stated data limitation.
- `outputs/commentary.txt` — the LLM-generated written trading commentary (only when an API key is present).
- `ai_logs/commentary_log.json` — the complete, self-contained AI interaction log: full prompt, full response, hallucination check, and status.

## Runtime

Approximately **2–3 minutes** on a typical laptop. The time is dominated by ~190 sequential calls to the Elexon API (one per 7-day price chunk, plus one per day for the 90-day wind and demand pulls). Cleaning, feature engineering, model training, validation, and the curve view complete in a few seconds; the LLM commentary call adds a few more seconds when a key is configured.

## Repository

https://github.com/benomayebu/benjamin-omayebu-cobblestone
