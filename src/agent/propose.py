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
from src.agent.state import AgentState


CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"


_ALL_ACTIONS = [
    "explore_new_class",
    "refine_existing",
    "combine_features",
    "increase_robustness",
]


def _format_action_stats(stats: dict, label: str) -> str:
    """Render action stats as a compact, readable table for the prompt."""
    if not stats:
        return f"{label}: no data yet."

    lines = [f"{label}:"]
    seen = set()
    for action in _ALL_ACTIONS:
        if action not in stats:
            continue
        seen.add(action)
        s = stats[action]
        count = s["count"]
        conf = " (LOW CONFIDENCE)" if s["low_confidence"] else ""
        sr = f"{s['success_rate']:.0%}"
        imp = f"{s['avg_improvement']:+.3f}%" if s["avg_improvement"] is not None else "n/a"
        stab = f"{s['stability_score']:.2f}" if s["stability_score"] is not None else "n/a"
        lines.append(
            f"  {action:<24}: {count} cycles{conf} | success {sr} | avg improvement {imp} | stability {stab}"
        )

    missing = [a for a in _ALL_ACTIONS if a not in seen]
    if missing:
        lines.append(f"  [no data yet for: {', '.join(missing)}]")

    return "\n".join(lines)


def _format_top_features(top_features: list) -> str:
    """Render top-K promoted features for the decision prompt."""
    if not top_features:
        return "No promoted features yet."
    lines = []
    for f in top_features:
        score_str = f"{f['feature_score']:.4f}" if f.get("feature_score") is not None else "unscored"
        lines.append(
            f"  {f['name']:<30} level={f['level']:<12} score={score_str}"
            + (f"  [{f['motivation'][:60]}]" if f.get("motivation") else "")
        )
    return "\n".join(lines)


def _format_coverage(coverage: dict) -> str:
    """Render feature space coverage as a compact table."""
    if not coverage:
        return "No coverage data yet (no features tested)."
    lines = []
    for key, info in coverage.items():
        lines.append(f"  {key:<30}: {info['coverage']:<8} ({info['count']} tested)")
    return "\n".join(lines)


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
    state: AgentState,
    user_hint: str = None,
) -> ToolResult:
    """
    Ask Claude to decide a research action and propose a candidate feature.

    Claude first diagnoses the research state (recent verdicts, rejection patterns,
    active feature coverage) and selects one of four actions before designing
    the feature. The action and its rationale are returned alongside the feature.

    Returns a ToolResult containing a FeatureProposal on success.
    """
    context = state.context
    context_str = format_context_for_prompt(context)
    principles_str = format_principles_for_prompt(state.principles)
    registry_str = format_registry_for_prompt(state.registry)
    active_str = format_active_features_for_prompt(state.active_features)

    freq_str = context["data"]["frequency"]
    m = re.match(r'(\d+)s', freq_str)
    freq_seconds = int(m.group(1)) if m else 10
    horizon_bars = context["data"]["horizon_seconds"] // freq_seconds

    stage = state.research_stage
    counts = state.level_counts
    stage_description = _STAGE_DESCRIPTION[stage]
    level_rules = _LEVEL_RULES[stage]

    # Build a compact diagnostic block from structured state
    recent_verdict_summary = (
        ", ".join(state.recent_verdicts) if state.recent_verdicts else "none yet"
    )
    rejection_pattern = (
        ", ".join(sorted(set(state.recent_rejection_reasons)))
        if state.recent_rejection_reasons else "none"
    )

    hint_block = (
        f"\nThe researcher has provided the following direction hint:\n{user_hint}\n"
        if user_hint else ""
    )

    overall_stats_str = _format_action_stats(state.action_stats, "Overall (last 20 cycles)")
    stage_stats_str = _format_action_stats(
        state.action_stats_by_stage.get(stage, {}),
        f"Current stage — {stage.upper()} features only"
    )
    top_features_str = _format_top_features(state.top_features)
    coverage_str = _format_coverage(state.feature_space_coverage)

    prompt = f"""
You are a systematic quantitative research agent specialising in FX volatility forecasting.

Your task has TWO parts:
  1. Decide which research action to take next, based on current state
  2. Propose ONE candidate feature that executes that action

---

PART 1 — DECIDE YOUR ACTION

Choose exactly one of the following actions. Base your choice on the research state below.

Available actions:
  - explore_new_class     → No features in this class yet, or repeated failures in same family → try something structurally different
  - refine_existing       → A feature showed weak but consistent signal → tighten parameterisation or window choice
  - combine_features      → Two complementary weak signals exist → build interaction or regime flag combining them
  - increase_robustness   → A feature showed strong but unstable importance → add guards, wider windows, or clipped extremes

Decision policy (apply in order):
  1. If registry is empty or research stage is primitive with no active features → explore_new_class
  2. If the same principle (e.g. P01, P03) appears repeatedly in recent rejections → increase_robustness or explore_new_class
  3. If a "modified" verdict appeared recently → refine_existing
  4. If two weak promoted features cover the same economic phenomenon → combine_features
  5. Default → explore_new_class

CURRENT RESEARCH STATE:
  Stage              : {stage.upper()} (primitives: {counts['primitive']}, transforms: {counts['transform']}, composites: {counts['composite']})
  Total cycles run   : {state.cycle_count}
  Features promoted  : {state.promoted_count}
  Recent verdicts    : {recent_verdict_summary}
  Rejection triggers : {rejection_pattern}

ACTION PERFORMANCE HISTORY:
{overall_stats_str}

{stage_stats_str}

How to use these stats (soft guidance only):
- Prioritise the current-stage stats when count ≥ 3; fall back to overall if stage data is sparse
- Favour actions with high success_rate AND positive avg_improvement
- Avoid actions with consistently negative avg_improvement across ≥ 3 cycles
- LOW CONFIDENCE (< 3 cycles): treat as a weak signal — do not over-index on it
- Do NOT mechanically follow the stats; context, stage, and recent rejections matter more
- If all stats are empty (early cycles), rely on the decision policy above

{hint_block}

---

PART 2 — PROPOSE THE FEATURE

The feature must be:
- Consistent with your chosen action
- Economically motivated with a clear and testable signal hypothesis
- Constructable from the available data described below
- Not already tested (check the registry history)
- Consistent with the research principles
- Implementable as a pure function of a pandas DataFrame with OHLC columns
- Focused on a single hypothesis (avoid multi-component designs unless justified)

Research philosophy:
- Prefer foundational signals early — simple, interpretable constructions that test one hypothesis at a time
- Complexity must be justified by what simpler features cannot explain

CURRENT RESEARCH STAGE: {stage.upper()} EXPLORATION
{stage_description}

TOP-RANKED PROMOTED FEATURES (composite score: perf×0.5 + stability×0.3 + novelty×0.2, then penalised for duplicates/instability):
{top_features_str}

Guidance (soft):
- If any feature has score > 0.6: strongly consider refine_existing or combine_features targeting it
- If no feature exceeds 0.5: the active set is weak — prefer explore_new_class in a low-coverage region
- Do NOT refine a feature purely because it is top-ranked; require a clear hypothesis for improvement

FEATURE SPACE COVERAGE (base_type + transform_type across all tested features — HIGH means overrepresented):
{coverage_str}

Guidance (soft):
- Strongly prefer LOW coverage regions when no strong candidates exist (score < 0.5)
- Avoid HIGH coverage regions unless the proposed construction is clearly distinct from existing ones
- MEDIUM coverage regions are fair game if the hypothesis is novel within that family

ACTIVE FEATURE SET:
{active_str}

HIERARCHY RULES FOR THIS CYCLE:
{level_rules}

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
  "action": "explore_new_class",
  "action_rationale": "why you chose this action given the current research state",
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
