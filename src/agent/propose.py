"""
propose.py
Feature proposal module.
Calls Claude to propose a candidate feature including a Python implementation.
"""

import os
import re
import json

import anthropic
from pydantic import ValidationError

from src.agent.context import format_context_for_prompt, format_principles_for_prompt
from src.agent.registry import format_registry_for_prompt
from src.agent.active_features import (
    get_research_stage,
    get_level_counts,
    format_active_features_for_prompt,
)
from src.agent.schemas import FeatureProposal
from src.agent.tool import ToolResult, retry_api_call


CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"


_STAGE_DESCRIPTION = {
    "primitive": (
        "PRIMITIVE feature exploration — propose a direct measurement from the price series. "
        "Examples: log_return, abs_return, high_low_range, realised_vol, rolling_std, ATR. "
        "No transforms or combinations. One quantity, one window."
    ),
    "transform": (
        "TRANSFORM feature exploration — apply a simple statistical transformation to an "
        "existing primitive. Examples: rolling z-score, percentile rank, persistence, "
        "volatility ratio, decay-weighted statistic, lagged value. "
        "The input must be a primitive already in the active set."
    ),
    "composite": (
        "COMPOSITE feature exploration — combine multiple primitives or transforms. "
        "Examples: regime flag, feature ratio, interaction term, persistence ratio. "
        "Complexity must be justified by what simpler features cannot explain."
    ),
}


_LEVEL_RULES = {
    "primitive": (
        "- Propose a Level 1 (Primitive) feature.\n"
        "- Direct measurement only — no ratios, no z-scores, no combinations.\n"
        "- A single rolling operation on a price-derived quantity is the target complexity."
    ),
    "transform": (
        "- Propose a Level 2 (Transform) feature.\n"
        "- Must be a statistical transformation of a Level 1 primitive already in the active set.\n"
        "- Do NOT propose another primitive — there are already primitives to build on."
    ),
    "composite": (
        "- Propose a Level 3 (Composite) feature.\n"
        "- Must combine or interact multiple primitives or transforms already in the active set.\n"
        "- Justify the added complexity explicitly in the rationale."
    ),
}


def propose_feature(
    context: dict,
    principles: list,
    registry: list,
    user_hint: str = None,
    active_features: list = None
) -> ToolResult:
    """
    Ask Claude to propose a candidate feature including a Python implementation.
    Returns a ToolResult containing a FeatureProposal on success.
    """

    if active_features is None:
        active_features = []

    context_str = format_context_for_prompt(context)
    principles_str = format_principles_for_prompt(principles)
    registry_str = format_registry_for_prompt(registry)
    active_str = format_active_features_for_prompt(active_features)

    freq_str = context["data"]["frequency"]
    m = re.match(r'(\d+)s', freq_str)
    freq_seconds = int(m.group(1)) if m else 10
    horizon_bars = context["data"]["horizon_seconds"] // freq_seconds

    stage = get_research_stage(active_features)
    counts = get_level_counts(active_features)
    stage_description = _STAGE_DESCRIPTION[stage]
    level_rules = _LEVEL_RULES[stage]

    hint_block = (
        f"\nThe researcher has provided the following direction hint:\n{user_hint}\n"
        if user_hint else ""
    )

    prompt = f"""
You are a systematic quantitative research agent specialising in FX volatility forecasting.

Your task is to propose ONE candidate feature for testing, including a working Python implementation.

The feature must be:
- Economically motivated with a clear and testable signal hypothesis
- Constructable from the available data described below
- Not already tested (check the registry history)
- Consistent with the research principles
- Implementable as a pure function of a pandas DataFrame with OHLC columns
- Focused on a single hypothesis (avoid multi-component designs unless justified)

Research philosophy:
- Before proposing, consider what has already been established and what the simplest meaningful next step would be
- Prefer foundational signals early in the research process — simple, interpretable constructions that test one hypothesis at a time
- Complexity must be justified by what simpler features cannot explain; do not propose multi-component features when a single-component version is untested

{hint_block}

CURRENT RESEARCH STAGE: {stage.upper()} EXPLORATION
{stage_description}

ACTIVE FEATURE SET:
{active_str}

HIERARCHY RULES FOR THIS CYCLE:
{level_rules}
- Primitives promoted: {counts['primitive']}
- Transforms promoted: {counts['transform']}
- Composites promoted: {counts['composite']}

DATA CONTEXT:
{context_str}

RESEARCH PRINCIPLES:
{principles_str}

RECENT REGISTRY HISTORY (avoid repeating these):
{registry_str}

The code field must contain a single self-contained Python function with this exact signature:
    def compute_feature(df: pandas.DataFrame) -> pandas.Series

Requirements for the code:
- df has columns: open, high, low, close (float64), indexed by timestamp
- IMPORTANT: there is NO volume column. Do not reference df["volume"] or any volume-derived quantity.
- Data is at {freq_str} resolution ({freq_seconds} seconds per bar)
- The forecast horizon is {horizon_bars} bars — this is NOT a window size, it is the prediction target
- ALL rolling window sizes in your code must be in BARS (integer candle counts), not seconds
- The function must return a pandas Series of the same length as df
- All computations must be strictly backward-looking (no look-ahead)
- Use only: numpy, pandas — no other imports
- Handle NaNs gracefully — return NaN where insufficient history exists
- Include numpy and pandas imports inside the function body
- Use min_periods on ALL rolling operations
- Guard against zero denominators
- Cap rolling windows at 360 bars maximum unless justified
- Your feature MUST NOT produce NaN for more than the first 10% of rows

Respond ONLY with a valid JSON object in exactly this structure, no preamble, no markdown:
{{
  "name": "short_descriptive_feature_name",
  "level": "primitive",
  "motivation": "one-line research motivation",
  "hypothesis": "plain language explanation of the signal hypothesis",
  "construction": "precise description of how to compute this feature",
  "code": "def compute_feature(df):\\n    import numpy as np\\n    import pandas as pd\\n    # implementation here\\n    return result",
  "rejection_criteria": "specific pre-stated conditions under which this feature should be rejected",
  "rationale": "why this feature is worth testing given current context and registry history"
}}
""".strip()

    # --- LLM call with retry ---
    def call_llm():
        return CLIENT.messages.create(
            model=MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

    try:
        response = retry_api_call(call_llm)
        raw = response.content[0].text.strip()
    except (IndexError, AttributeError) as e:
        return ToolResult(
            ok=False,
            error=f"[propose] Invalid API response structure: {str(e)}"
        )
    except Exception as e:
        return ToolResult(
            ok=False,
            error=f"[propose] API call failed: {type(e).__name__}: {str(e)}"
        )

    # --- Strip markdown fences safely ---
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.strip().startswith("json"):
            raw = raw.strip()[4:]
    raw = raw.strip()

    # --- Parse into schema ---
    try:
        proposal = FeatureProposal.model_validate_json(raw)
        return ToolResult(ok=True, value=proposal)
    except (json.JSONDecodeError, ValidationError) as e:
        return ToolResult(
            ok=False,
            error=f"[propose] {type(e).__name__}: {str(e)} | raw={raw[:200]}"
        )
