# GB Power Day-Ahead Price Forecast

**Benjamin Omayebu** — benjaminomayebu@gmail.com

---

## 1. Market and Data

The market is **GB (Great Britain)**, forecasting next-day hourly day-ahead electricity prices.

Three public data sources are used, all from the Elexon BMRS API and **none requiring an API key**:

- **Day-ahead price** — the Elexon Market Index endpoint (`/balancing/pricing/market-index`) using the **APXMIDP** provider.
- **Wind generation forecast** — the **WINDFOR** dataset endpoint (`/datasets/WINDFOR`).
- **National demand forecast** — the **NDF** dataset endpoint (`/datasets/NDF`).

The pipeline pulls the 90 days ending yesterday. For this run the merged hourly dataset spans **2026-03-20 21:00 UTC to 2026-06-20 20:00 UTC** (2,208 hourly rows). One real finding from the QA log: there were **no duplicate timestamps, no negative wind or demand values, and no out-of-range prices**, but the price feed had **47 missing hours** (dropped, since price is the target) and demand had **20 missing hours** (filled by time-based linear interpolation rather than dropped).

> Note on the price source: the originally intended N2EXMIDP provider returned all-zero prices in this environment, which would have made the target degenerate. The APXMIDP provider on the same endpoint carries real GB market-index prices in GBP/MWh and was used instead.

## 2. Feature Engineering

Three types of features are built from the cleaned hourly data:

- **Calendar features** — hour of day (0–23), day of week (Monday=0 to Sunday=6), and a weekend flag (1 for Saturday/Sunday, else 0). These capture the strong daily and weekly cycles in electricity demand.
- **Raw fundamental drivers** — the wind generation forecast and the demand forecast, the two physical quantities that set the supply/demand balance.
- **Key engineered feature — net demand**, calculated as demand minus wind.

Net demand matters because wind is dispatched first: it has near-zero marginal cost, so net demand represents the load that must still be met by gas plants. When net demand is high, gas sets the marginal price and electricity prices rise; when wind covers most of demand, net demand falls and prices soften.

Two price-history features are also added — the price 24 hours earlier and the price 168 hours (one week) earlier — to capture the daily and weekly repetition of prices. The earliest rows with no week-ago history are dropped, leaving **1,993 model-ready rows**.

## 3. Forecasting Models and Validation

**Baseline — seasonal naive.** The baseline predicts that the price this hour equals the price exactly one week ago at the same hour (the 168-hour lag). It was chosen because weekly patterns explain a large share of price variation, making it the standard, hard-to-beat benchmark in electricity price forecasting; any model worth using must beat it.

**Improved model — HistGradientBoostingRegressor.** A gradient-boosted tree model was chosen over a linear model because the relationship between price and its drivers is non-linear (for example, prices rise steeply only once net demand is high enough that expensive gas plants are needed) and involves interactions between features that a linear model cannot capture. It also handles the mixed feature scales and any residual missing values robustly, with `random_state=42` for reproducibility.

Out-of-sample validation results on the held-out final 7 days (168 hours):

| Model | MAE (GBP/MWh) | RMSE (GBP/MWh) |
|---|---|---|
| Baseline (seasonal naive) | 23.17 | 34.13 |
| Improved (HistGradientBoosting) | 18.84 | 26.65 |

The improved model **beat the baseline**, cutting MAE by **18.7%** (from 23.17 to 18.84) and RMSE by about 22%. A strict **time-based split** was used — the final 7 days are held out and the data is never shuffled. A random split would be wrong because it would let the model train on future hours to predict past ones, leaking information that would never be available in a real trading setting and producing over-optimistic results.

## 4. Prompt Curve Translation

The day-ahead forecast is translated into a directional view on the front-week prompt contract:

- **Forecast average:** 85.15 GBP/MWh
- **Reference level (14-day trailing average of actual price):** 94.18 GBP/MWh
- **Deviation:** −9.58%
- **Trading view:** **SHORT**

**Reasoning:** the model's average forecast sits 9.58% below where prices have recently been, so the front-week prompt contract looks expensive relative to where the model expects prices to land; the trade idea is to sell prompt now and buy back lower later.

**Invalidation:** a sharp downward revision to wind forecasts or a significant rise in demand forecasts would raise net demand and push prices back up; the short should be closed or reversed if either materialises.

**Data limitation:** the reference level is a 14-day trailing average of realised prices used as a transparent proxy — in a real trading seat it would be compared against the live quoted front-week curve price from a broker screen, which is not available in this environment.

## 5. AI and LLM Integration

The LLM component automatically generates a written **morning trading commentary** from the structured pipeline outputs (the validation metrics and the curve view), producing the three-paragraph note in Section 4's spirit: model performance, the trading view and its reasoning, and the invalidation condition.

An LLM is the right tool for this specific step for two reasons. First, producing a readable morning note from already-computed numbers is a manual writing task that requires no new analysis — it is natural language generation from structured input, exactly what a language model does well. Second, the model is constrained to use only the figures provided in the prompt, and a post-hoc regex check extracts every number in the response and verifies it appears in the prompt, which guards against the model inventing figures.

The full prompt and the full response are logged to `ai_logs/commentary_log.json` so the entire interaction can be audited without re-running the code or holding an API key. On the final run the **hallucination check confirmed zero unexplained numbers** in the response (all seven figures traced back to the prompt), and the step recorded `status: success`.
