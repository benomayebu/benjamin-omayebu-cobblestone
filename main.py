"""Entry point for the GB day-ahead electricity price forecasting project.

Runs the full pipeline end to end:
  1. Data ingestion (Elexon public API) -> data/merged.csv
  2. Cleaning + feature engineering
  3. Train/test split + baseline and improved models
  4. Validation outputs: figures/validation.png, outputs/validation_metrics.json,
     predictions.csv
"""

import json

import pandas as pd

from src.curve_view import compute_curve_view, print_curve_view
from src.data_ingestion import DATA_DIR, ingest
from src.features import add_features, clean_data
from src.llm_commentary import LOG_PATH, generate_commentary
from src.models import OUTPUTS_DIR, run_models


def main() -> None:
    # --- 1. Ingestion ---
    ingest()
    merged = pd.read_csv(DATA_DIR / "merged.csv")
    n_merged = len(merged)

    # --- 2. Cleaning ---
    cleaned = clean_data(merged)
    n_cleaned = len(cleaned)
    print(f"\nAfter cleaning (dropped missing price, interpolated demand): "
          f"{n_cleaned} rows")

    # --- 3. Feature engineering ---
    featured = add_features(cleaned)
    n_featured = len(featured)
    print(f"After feature engineering (dropped rows lacking 24h/168h lags): "
          f"{n_featured} rows")

    # --- 4. Models + validation ---
    res = run_models(featured)

    # --- 5. Summary ---
    base, imp = res["baseline"], res["improved"]
    mae_impr = (base["MAE"] - imp["MAE"]) / base["MAE"] * 100 if base["MAE"] else 0.0
    verdict = ("YES - improved model beats the baseline"
               if res["improved_beats_baseline"]
               else "NO - improved model does not beat the baseline")

    print("\n" + "=" * 64)
    print("PIPELINE SUMMARY")
    print("=" * 64)
    print(f"Merged rows:              {n_merged}")
    print(f"After cleaning:           {n_cleaned}")
    print(f"After feature engineering:{n_featured:>4}")
    print(f"Train rows:               {res['n_train']}")
    print(f"Test rows (final 7 days): {res['n_test']}  "
          f"({res['test_start']} -> {res['test_end']})")
    print("-" * 64)
    print(f"{'Model':<34}{'MAE':>12}{'RMSE':>12}")
    print(f"{'Baseline (seasonal naive)':<34}"
          f"{base['MAE']:>12.3f}{base['RMSE']:>12.3f}")
    print(f"{'Improved (HistGradientBoosting)':<34}"
          f"{imp['MAE']:>12.3f}{imp['RMSE']:>12.3f}")
    print("-" * 64)
    print(f"MAE improvement vs baseline: {mae_impr:.1f}%")
    print(f"Verdict: {verdict}")
    print("=" * 64)
    print("Outputs written: figures/validation.png, "
          "outputs/validation_metrics.json, predictions.csv")

    # --- 6. Prompt-curve view (runs after model validation) ---
    curve = compute_curve_view()
    print_curve_view(curve)
    print("Output written: outputs/curve_view.json")

    # --- 7. LLM trading commentary (final step) ---
    metrics = json.loads((OUTPUTS_DIR / "validation_metrics.json").read_text())
    curve_view = json.loads((OUTPUTS_DIR / "curve_view.json").read_text())
    commentary = generate_commentary(metrics, curve_view)

    print("\n" + "=" * 64)
    print("AI TRADING COMMENTARY")
    print("=" * 64)
    if commentary is not None:
        print(commentary)
        (OUTPUTS_DIR / "commentary.txt").write_text(commentary + "\n")
        print("\nOutput written: outputs/commentary.txt")
    else:
        log = json.loads(LOG_PATH.read_text())
        print("Commentary step SKIPPED -- no commentary was produced.")
        print(f"Reason: {log['status']}")
        print(f"Full AI interaction log: {LOG_PATH}")
    print("=" * 64)


if __name__ == "__main__":
    main()
