"""
reason.py
Reasoning and verdict module.
Calls Claude to interpret test results against the principles register
and produce a structured plain-language verdict.
"""

import os
import json

import anthropic
from pydantic import ValidationError

from src.agent.context import format_principles_for_prompt
from src.agent.active_features import get_level_counts
from src.agent.schemas import Verdict, TestResults
from src.agent.tool import ToolResult, retry_api_call


CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"


def format_test_results_for_prompt(test_results) -> str:
    """Converts test results into a readable prompt block. Accepts TestResults model or dict."""
    if isinstance(test_results, TestResults):
        agg = dict(test_results.aggregate)
        per_pair = dict(test_results.per_pair)
    else:
        agg = test_results.get("aggregate", {})
        per_pair = test_results.get("per_pair", {})

    lines = ["AGGREGATE RESULTS:"]

    if agg.get("skipped"):
        lines.append(f"  TEST SKIPPED: {agg.get('skip_reason', 'unknown reason')}")
    else:
        lines.append(f"  Improvement vs LightGBM baseline: {agg.get('mean_improvement_vs_lgbm_pct')}%")
        lines.append(f"  Improvement vs naive rolling vol baseline: {agg.get('mean_improvement_vs_naive_pct')}%")
        lines.append(f"  Mean candidate importance: {agg.get('mean_candidate_importance_pct')}%")
        lines.append(f"  Mean importance drift across folds: {agg.get('mean_importance_drift_pct')}%")
        lines.append(f"  Any monotonic decay detected: {agg.get('any_monotonic_decay')}")
        if agg.get("errors"):
            lines.append(f"  Errors: {'; '.join(agg['errors'])}")

    # Novelty signals — always shown if present
    novelty_class = agg.get("novelty_class")
    if novelty_class:
        lines.append(f"\nNOVELTY ASSESSMENT:")
        lines.append(f"  Class: {novelty_class} (score {agg.get('novelty_score', 'n/a')})")
        if agg.get("most_similar_feature"):
            lines.append(f"  Most similar existing feature: {agg['most_similar_feature']}")
        lines.append(f"  Detail: {agg.get('novelty_explanation', '')}")

    lines.append("\nPER-PAIR RESULTS:")
    for pair, results in per_pair.items():
        lines.append(f"\n  {pair}:")
        if "error" in results:
            lines.append(f"    Error: {results['error']}")
            continue

        lines.append(f"    Overall improvement vs LightGBM baseline: {results.get('overall_improvement_vs_lgbm_pct')}%")
        lines.append(f"    Overall improvement vs naive baseline: {results.get('overall_improvement_vs_naive_pct')}%")
        lines.append(f"    Mean candidate importance: {results.get('mean_candidate_importance_pct')}%")
        lines.append(f"    Importance drift: {results.get('importance_drift_pct')}%")
        lines.append(f"    Monotonic decay: {results.get('monotonic_decay')}")
        lines.append(f"    Folds completed: {results.get('n_folds_completed')}")

        for fold in results.get("folds", []):
            lines.append(
                f"      Fold {fold.get('fold')}: improvement vs LightGBM {fold['improvement_vs_lgbm_baseline_pct']}%, "
                f"vs naive {fold['improvement_vs_naive_baseline_pct']}%, "
                f"importance {fold['candidate_importance_pct']}%"
            )

    return "\n".join(lines)


def reason_and_verdict(
    feature: dict,
    test_results: dict,
    principles: list,
    context: dict,
    active_features: list = None
) -> ToolResult:
    """
    Ask Claude to reason about test results and issue a structured verdict.

    Returns:
        ToolResult:
            ok=True → value is a Verdict
            ok=False → error contains failure details
    """

    if active_features is None:
        active_features = []

    principles_str = format_principles_for_prompt(principles)
    results_str = format_test_results_for_prompt(test_results)
    counts = get_level_counts(active_features)

    if isinstance(test_results, TestResults):
        agg = test_results.aggregate
    else:
        agg = test_results.get("aggregate", {}) if isinstance(test_results, dict) else {}

    novelty_class = agg.get("novelty_class", "novel")
    novelty_block = ""
    if novelty_class == "near_duplicate":
        novelty_block = (
            f"\nNOVELTY GUARDRAIL: This feature is classified as near_duplicate "
            f"(most similar: {agg.get('most_similar_feature', 'unknown')}). "
            f"It should be rejected unless it demonstrates clearly superior performance "
            f"AND a distinct mechanism not captured by the existing feature.\n"
        )
    elif novelty_class == "similar_family":
        novelty_block = (
            f"\nNOVELTY NOTE: This feature is from a similar family to "
            f"'{agg.get('most_similar_feature', 'an existing feature')}'. "
            f"Accept only if it shows material improvement. "
            f"Reject if the improvement is marginal relative to the similarity.\n"
        )

    candidate_level = feature.get("level", "primitive") if isinstance(feature, dict) else getattr(feature, "level", "primitive")

    hierarchy_violation = None
    if candidate_level == "transform" and counts["primitive"] == 0:
        hierarchy_violation = "Hierarchy violation: Level 2 feature without active Level 1 primitives — must reject."
    elif candidate_level == "composite" and counts["transform"] == 0:
        hierarchy_violation = "Level 3 feature without Level 2 transforms — reject."

    hierarchy_block = (
        f"\nHIERARCHY GUARDRAIL: {hierarchy_violation}\n"
        if hierarchy_violation else ""
    )

    prompt = f"""
You are a systematic quantitative research agent reviewing the test results for a candidate FX volatility feature.

Your role is to reason carefully against the research principles and issue a clear, defensible verdict.
You must be sceptical. Marginal improvements do not justify promotion. Instability is grounds for rejection.
{hierarchy_block}{novelty_block}
FEATURE PROPOSAL:
Name: {feature.get('name') if isinstance(feature, dict) else feature.name}
Level: {candidate_level}
Motivation: {feature.get('motivation') if isinstance(feature, dict) else feature.motivation}
Hypothesis: {feature.get('hypothesis') if isinstance(feature, dict) else feature.hypothesis}
Construction: {feature.get('construction') if isinstance(feature, dict) else feature.construction}
Pre-stated rejection criteria: {feature.get('rejection_criteria') if isinstance(feature, dict) else feature.rejection_criteria}

TEST RESULTS:
{results_str}

RESEARCH PRINCIPLES:
{principles_str}

Instructions:
1. Evaluate the results rigorously against the research principles and any pre-stated rejection criteria.

2. Assess the following explicitly:
   - Magnitude: is the improvement over baseline meaningful, not marginal?
   - Consistency: is the improvement stable across pairs and folds, or driven by a small subset?
   - Stability: is there evidence of importance drift or monotonic decay across folds?
   - Interpretability: does the observed behaviour match the stated hypothesis?

3. Issue one of three verdicts:
   - promoted: clear, consistent, and stable improvement aligned with the hypothesis
   - rejected: weak, inconsistent, unstable, or contradicting the hypothesis or rejection criteria
   - modified: hypothesis appears valid but the construction or parameterisation is likely suboptimal

If results are mixed, marginal, or ambiguous, default to "rejected" rather than "promoted".

4. Write a summary of exactly 2–3 sentences:
   - sentence 1: state the verdict and the primary reason
   - sentence 2: highlight the most important supporting evidence from the results
   - sentence 3 (only if promoted or modified): what risk or behaviour should be monitored next

5. State the single most useful next action:
   - be specific and actionable
   - focus on the next experiment that would most reduce uncertainty
   
Respond ONLY with a valid JSON object in exactly this structure:
{{
  "verdict": "promoted" | "rejected" | "modified",
  "summary": "plain language verdict summary",
  "triggered_principles": ["P01", "P03"],
  "next_action": "single concrete recommendation"
}}
""".strip()

    # --- LLM call with retry ---
    def call_llm():
        return CLIENT.messages.create(
            model=MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

    try:
        response = retry_api_call(call_llm)
        raw = response.content[0].text.strip()
    except (IndexError, AttributeError) as e:
        return ToolResult(
            ok=False,
            error=f"[reason] Invalid API response structure: {str(e)}"
        )
    except Exception as e:
        return ToolResult(
            ok=False,
            error=f"[reason] API call failed: {type(e).__name__}: {str(e)}"
        )

    # --- Strip markdown fences ---
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.strip().startswith("json"):
            raw = raw.strip()[4:]
    raw = raw.strip()

    # --- Parse into schema ---
    try:
        verdict = Verdict.model_validate_json(raw)
        return ToolResult(ok=True, value=verdict)
    except (json.JSONDecodeError, ValidationError) as e:
        return ToolResult(
            ok=False,
            error=f"[reason] {type(e).__name__}: {str(e)} | raw={raw[:200]}"
        )