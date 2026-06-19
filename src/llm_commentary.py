from dotenv import load_dotenv
load_dotenv()

"""LLM-generated trading commentary from structured pipeline outputs.

This is the project's programmatic AI component. The pipeline already produces
all the numbers (validation metrics + a curve view); turning them into a short
written commentary that a trader can read in 30 seconds is a manual,
type-it-by-hand-each-morning task. It needs no new analysis -- only natural
language generation from structured input -- which is exactly what an LLM is
good at. The model is used purely as a writing tool: every figure must come from
the pipeline outputs, never from the model's own knowledge, and a post-hoc check
flags any number in the response that was not in the prompt.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
AI_LOGS_DIR = ROOT / "ai_logs"
LOG_PATH = AI_LOGS_DIR / "commentary_log.json"

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 600

# Matches integers and decimals, with an optional leading minus sign.
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _build_prompt(metrics: Dict, curve_view: Dict) -> str:
    """Assemble the prompt from the structured outputs -- and nothing else."""
    base = metrics["baseline_seasonal_naive"]
    impr = metrics["improved_hist_gradient_boosting"]
    beat = "Yes" if metrics["improved_beats_baseline"] else "No"

    return (
        "You are writing a short morning trading commentary for a GB power "
        "trader. Use ONLY the numbers and facts provided below. Do NOT "
        "introduce any new figures, do NOT compute new percentages, and do NOT "
        "add facts that are not listed here.\n\n"
        "MODEL VALIDATION (held-out test set):\n"
        f"- Baseline (seasonal naive): MAE = {base['MAE']:.2f} GBP/MWh, "
        f"RMSE = {base['RMSE']:.2f} GBP/MWh\n"
        f"- Improved (gradient boosting): MAE = {impr['MAE']:.2f} GBP/MWh, "
        f"RMSE = {impr['RMSE']:.2f} GBP/MWh\n"
        f"- Did the improved model beat the baseline? {beat}\n\n"
        "TRADING VIEW (front-week prompt curve):\n"
        f"- Forecast average price: {curve_view['forecast_avg_gbp_mwh']:.2f} "
        "GBP/MWh\n"
        f"- Reference level (recent prices): "
        f"{curve_view['reference_level_gbp_mwh']:.2f} GBP/MWh\n"
        f"- Deviation: {curve_view['deviation_pct']:.2f}%\n"
        f"- View: {curve_view['view']}\n"
        f"- Reasoning: {curve_view['reasoning']}\n"
        f"- Invalidation: {curve_view['invalidation']}\n\n"
        "Write EXACTLY three short paragraphs in plain, simple English:\n"
        "Paragraph 1: How well the model performed in validation compared to "
        "the baseline.\n"
        "Paragraph 2: State the trading view and explain the reasoning behind "
        "it clearly.\n"
        "Paragraph 3: State what would invalidate the view and what the trader "
        "should watch for.\n\n"
        "Use only the numbers provided above. Do not invent or compute any new "
        "figures."
    )


def _validate_numbers(response_text: str, prompt_text: str) -> Dict:
    """Extract every number from the response and check each appears in the
    prompt. Any that does not is flagged as a potential hallucination."""
    numbers_in_response: List[str] = NUMBER_RE.findall(response_text)
    unexplained: List[str] = [
        n for n in numbers_in_response if n not in prompt_text
    ]
    return {
        "numbers_in_response": numbers_in_response,
        "unexplained_numbers": unexplained,
        "any_unexplained": len(unexplained) > 0,
    }


def _write_log(entry: Dict) -> None:
    AI_LOGS_DIR.mkdir(exist_ok=True)
    LOG_PATH.write_text(json.dumps(entry, indent=2) + "\n")


def generate_commentary(metrics: Dict, curve_view: Dict) -> Optional[str]:
    """Generate written trading commentary via the Anthropic API.

    Returns the commentary string on success, or None if the import or API call
    fails. A complete, self-contained log is always written to
    ai_logs/commentary_log.json so the interaction can be understood without
    re-running the code (or holding an API key).
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    prompt_text = _build_prompt(metrics, curve_view)

    try:
        import anthropic

        client = anthropic.Anthropic()
        message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt_text}],
        )
        commentary = message.content[0].text
    except Exception as err:  # import error, missing key, API failure, etc.
        error_msg = f"{type(err).__name__}: {err}"
        warning = None
        if "ANTHROPIC_API_KEY" in error_msg or "api_key" in error_msg:
            warning = ("ANTHROPIC_API_KEY is not set in the environment, so the "
                       "live API call could not be made.")
        _write_log({
            "timestamp": timestamp,
            "model_used": MODEL,
            "full_prompt": prompt_text,
            "full_response": None,
            "validation_check": {
                "numbers_in_response": [],
                "unexplained_numbers": [],
                "any_unexplained": False,
                "note": "No response received; validation not performed.",
            },
            "status": error_msg if warning is None else f"{error_msg} | {warning}",
        })
        return None

    validation_check = _validate_numbers(commentary, prompt_text)
    if validation_check["any_unexplained"]:
        validation_check["warning"] = (
            "Potential hallucination: the response contains number(s) "
            f"{validation_check['unexplained_numbers']} not present in the "
            "prompt."
        )

    _write_log({
        "timestamp": timestamp,
        "model_used": MODEL,
        "full_prompt": prompt_text,
        "full_response": commentary,
        "validation_check": validation_check,
        "status": "success",
    })
    return commentary
