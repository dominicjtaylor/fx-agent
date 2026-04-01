"""
loop.py
Orchestrates a single full agent cycle:
propose -> test -> reason -> log

Follows a structured research pipeline:
  Step 1  Load data
  Step 2  Report dataset state
  Step 3  Describe current feature set
  Step 4  Select research stage
  Step 5  Propose candidate feature
  Step 6  Explain reasoning for proposal
  Step 7  Train model with walk-forward validation
  Step 8  Evaluate candidate feature
  Step 9  Apply promotion / rejection rules
  Step 10 Log decision and update feature registry
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from src.agent.context import load_context, load_principles
from src.agent.registry import load_registry, save_entry, make_entry
from src.agent.propose import propose_feature
from src.agent.active_features import (
    load_active_features,
    add_active_feature,
    get_research_stage,
    get_level_counts,
    format_active_features_for_prompt,
)
from src.agent.test import run_feature_test
from src.agent.reason import reason_and_verdict

LOG_DIR = Path(__file__).resolve().parents[2] / "outputs" / "logs"

_STAGE_LABEL = {
    "primitive": "Primitive feature exploration",
    "transform": "Transform feature exploration",
    "composite": "Composite feature exploration",
}

_LEVEL_LABEL = {
    "primitive": "Level 1: Primitive",
    "transform": "Level 2: Transform",
    "composite": "Level 3: Composite",
}


def save_log(entry: dict) -> None:
    """Save a markdown reasoning log for this cycle."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = LOG_DIR / f"{timestamp}_{entry['name']}.md"

    with open(filename, "w") as f:
        f.write(f"# {entry['name']}\n\n")
        f.write(f"**Timestamp:** {entry['timestamp']}\n")
        f.write(f"**Level:** {entry.get('level', 'unknown')}\n")
        f.write(f"**Verdict:** {entry['verdict'].upper()}\n\n")
        f.write(f"## Motivation\n{entry.get('motivation', '')}\n\n")
        f.write(f"## Hypothesis\n{entry['hypothesis']}\n\n")
        f.write(f"## Construction\n{entry['construction']}\n\n")
        f.write(f"## Rejection Criteria\n{entry['rejection_criteria']}\n\n")
        f.write(f"## Test Results\n```json\n{json.dumps(entry['test_results'], indent=2)}\n```\n\n")
        f.write(f"## Triggered Principles\n{', '.join(entry['triggered_principles'])}\n\n")
        f.write(f"## Summary\n{entry['summary']}\n\n")
        f.write(f"## Next Action\n{entry['next_action']}\n")


def run_cycle(
    data: Dict[str, pd.DataFrame],
    user_hint: Optional[str] = None
) -> dict:

    # ── Step 1-2: Dataset state ───────────────────────────────────────────────
    print("Dataset state...")
    total_rows = sum(len(df) for df in data.values())
    pair_summary = ", ".join(f"{pair}: {len(df):,} rows" for pair, df in data.items())
    print(f"  State: {total_rows:,} total observations — {pair_summary}")

    # ── Step 3-4: Feature set and research stage ──────────────────────────────
    context = load_context()
    principles = load_principles()
    registry = load_registry()
    active_features = load_active_features()
    counts = get_level_counts(active_features)
    stage = get_research_stage(active_features)

    print("Current feature set...")
    print(
        f"  State: {len(active_features)} active features "
        f"({counts['primitive']} primitive, {counts['transform']} transform, {counts['composite']} composite)"
    )

    print("Research stage...")
    print(f"  Stage: {_STAGE_LABEL[stage]}")

    # ── Step 5-6: Proposal ────────────────────────────────────────────────────
    print("Proposing feature...")
    propose_result = propose_feature(
        context, principles, registry,
        user_hint=user_hint,
        active_features=active_features
    )
    if not propose_result.ok:
        raise RuntimeError(propose_result.error)
    feature = propose_result.value
    level_label = _LEVEL_LABEL.get(feature.level, feature.level)
    print(f"  Motivation: {feature.motivation}")
    print(f"  Candidate: {feature.name} ({level_label})")

    # ── Step 7: Walk-forward validation ──────────────────────────────────────
    ctx = context["data"]
    freq_str = ctx["frequency"]
    horizon = ctx["horizon_seconds"]
    import re
    m = re.match(r'(\d+)s', freq_str)
    freq_seconds = int(m.group(1)) if m else 10
    horizon_bars = horizon // freq_seconds
    n_folds = 5

    print("Walk-forward validation...")
    print(
        f"  Method: chronological folds with expanding training window | "
        f"{n_folds} folds | min train: 50% of data | "
        f"horizon: {horizon_bars} bars ({horizon}s)"
    )

    # ── Step 8: Test ──────────────────────────────────────────────────────────
    print("Running feature tests...")
    test_result = run_feature_test(feature, context, data, active_features=active_features)
    if not test_result.ok:
        raise RuntimeError(test_result.error)
    test_results = test_result.value
    agg = test_results.aggregate
    lgbm_imp = agg.get("mean_improvement_vs_lgbm_pct")
    naive_imp = agg.get("mean_improvement_vs_naive_pct")
    drift = agg.get("mean_importance_drift_pct")
    imp = agg.get("mean_candidate_importance_pct")
    errors = agg.get("errors", [])

    if lgbm_imp is not None:
        print(f"  Result: {lgbm_imp:+.3f}% vs LightGBM baseline | {naive_imp:+.3f}% vs naive rolling vol")
        print(f"  Importance: {imp:.2f}% mean | {drift:.1f}% drift across folds")
    if errors:
        for err in errors:
            print(f"  Error: {err}")

    # ── Step 9: Verdict ───────────────────────────────────────────────────────
    print("Reasoning and issuing verdict...")
    reason_result = reason_and_verdict(feature, test_results, principles, context, active_features=active_features)
    if not reason_result.ok:
        raise RuntimeError(reason_result.error)
    reasoning = reason_result.value
    verdict = reasoning.verdict
    print(f"  Verdict: {verdict.upper()}")

    # ── Step 10: Log ──────────────────────────────────────────────────────────
    print("Registry update...")
    entry = make_entry(feature, test_results, reasoning)
    save_entry(entry)
    save_log(entry)

    if verdict == "promoted":
        add_active_feature(feature)
        new_counts = get_level_counts(load_active_features())
        print(
            f"  Feature added to {feature.get('level', 'primitive')} registry "
            f"({new_counts['primitive']} primitive, {new_counts['transform']} transform, {new_counts['composite']} composite)"
        )
    else:
        print(f"  Entry saved: {entry['id']}")

    # ── Cycle summary ─────────────────────────────────────────────────────────
    print()
    print("Cycle summary")
    print("─" * 45)
    print(f"  Stage      : {_STAGE_LABEL[stage]}")
    print(f"  Feature    : {feature.name} ({level_label})")
    print(f"  Motivation : {feature.motivation, '—')}")
    print(
        f"  Validation : walk-forward CV, {n_folds} chronological folds, expanding window"
    )
    if lgbm_imp is not None:
        print(f"  RMSE change: {lgbm_imp:+.3f}% vs LightGBM | {naive_imp:+.3f}% vs naive")
    print(f"  Verdict    : {verdict.upper()}")
    print()

    entry["summary"] = reasoning.summary
    return entry
