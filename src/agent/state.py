"""
state.py
Typed research state loaded from disk at the start of each cycle.

AgentState is the single structured object that carries all context
into the decision and proposal steps. It replaces loose dict passing.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.agent.context import load_context, load_principles
from src.agent.registry import load_registry
from src.agent.active_features import (
    load_active_features,
    get_research_stage,
    get_level_counts,
)


@dataclass
class AgentState:
    """
    Structured snapshot of all research state for one cycle.
    Loaded from disk once at cycle start — not mutated during the cycle.
    """
    context: dict
    principles: list
    registry: list
    active_features: list
    research_stage: str                   # "primitive" | "transform" | "composite"
    level_counts: dict                    # {"primitive": int, "transform": int, "composite": int}
    cycle_count: int                      # total entries in registry (all verdicts)
    promoted_count: int                   # number of promoted entries
    recent_rejection_reasons: List[str]   # triggered_principles from last 5 rejected entries
    recent_verdicts: List[str]            # verdicts from last 5 entries, newest first
    action_stats: Dict[str, dict]         # overall per-action metrics, last 20 cycles
    action_stats_by_stage: Dict[str, Dict[str, dict]]  # same, filtered by feature level
    top_features: List[dict]              # top-K promoted features by feature_score
    feature_space_coverage: Dict[str, dict]  # {base+transform: {count, coverage}}


def compute_action_stats(
    registry: list,
    window: int = 20,
    stage_filter: Optional[str] = None,
) -> Dict[str, dict]:
    """
    Compute per-action performance metrics over the last `window` registry entries.

    Args:
        registry:     Full registry list (chronological).
        window:       How many recent entries to consider.
        stage_filter: If set, restrict to entries where feature level == stage_filter.
                      Stage names match level names: "primitive", "transform", "composite".

    Returns:
        Dict keyed by action name, each value containing:
          count          — number of cycles using this action
          low_confidence — True if count < 3 (metrics not statistically meaningful)
          success_rate   — fraction of promoted outcomes (0.0–1.0)
          avg_improvement — mean RMSE improvement vs LightGBM baseline (pct), or None
          stability_score — mean of (1 - drift/100), clipped to [0,1], or None
    """
    recent = registry[-window:] if len(registry) > window else registry
    if stage_filter:
        recent = [e for e in recent if e.get("level") == stage_filter]

    buckets: Dict[str, dict] = {}
    for entry in recent:
        action = entry.get("action")
        if not action:
            continue
        if action not in buckets:
            buckets[action] = {"count": 0, "promoted": 0, "improvements": [], "drifts": []}
        b = buckets[action]
        b["count"] += 1
        if entry.get("verdict") == "promoted":
            b["promoted"] += 1
        agg = (entry.get("test_results") or {}).get("aggregate", {})
        imp = agg.get("mean_improvement_vs_lgbm_pct")
        if imp is not None:
            b["improvements"].append(float(imp))
        drift = agg.get("mean_importance_drift_pct")
        if drift is not None:
            # Clip to [0, 100] so stability_score stays in [0, 1]
            b["drifts"].append(min(max(float(drift), 0.0), 100.0))

    result: Dict[str, dict] = {}
    for action, b in buckets.items():
        count = b["count"]
        imps = b["improvements"]
        drifts = b["drifts"]
        result[action] = {
            "count": count,
            "low_confidence": count < 3,
            "success_rate": round(b["promoted"] / count, 2) if count > 0 else 0.0,
            "avg_improvement": round(sum(imps) / len(imps), 3) if imps else None,
            "stability_score": (
                round(sum(1.0 - d / 100.0 for d in drifts) / len(drifts), 2)
                if drifts else None
            ),
        }
    return result


# ── Feature scoring ───────────────────────────────────────────────────────────

_W_PERF = 0.5
_W_STAB = 0.3
_W_NOV  = 0.2
_PERF_MIN, _PERF_MAX = -2.0, 5.0          # % improvement normalisation bounds
_NOVELTY_MAP = {"near_duplicate": 0.0, "similar_family": 0.4, "novel": 1.0}
_NOVELTY_PENALTY = {"near_duplicate": 0.5}  # multiplicative; similar_family=1.0 (no change)
_NOVELTY_BONUS   = {"novel": 1.1}           # conservative upside, capped at 1.0 post-clip
_STABILITY_THRESHOLD = 0.5                  # below this → stability guard fires
_STABILITY_GUARD = 0.7                      # multiplicative reduction for noisy features
TOP_K = 5


def compute_feature_score(agg: dict) -> Optional[float]:
    """
    Compute a single composite score [0, 1] for a tested feature.

    Step 1 — weighted sum:
      score = 0.5 * perf_norm + 0.3 * stability + 0.2 * novelty_val

    Step 2 — non-linear novelty adjustment (multiplicative):
      near_duplicate → score *= 0.5  (hard penalty: cannot dominate ranking)
      novel          → score *= 1.1, capped at 1.0  (small bonus)
      similar_family → unchanged

    Step 3 — stability guard (multiplicative):
      stability < 0.5 → score *= 0.7  (noisy features penalised)

    Returns None if the test was skipped or performance data is unavailable.
    """
    if agg.get("skipped"):
        return None
    perf_raw = agg.get("mean_improvement_vs_lgbm_pct")
    if perf_raw is None:
        return None

    perf_clipped = min(max(float(perf_raw), _PERF_MIN), _PERF_MAX)
    perf_norm = (perf_clipped - _PERF_MIN) / (_PERF_MAX - _PERF_MIN)  # → [0, 1]

    drift = agg.get("mean_importance_drift_pct", 50.0)
    stability = 1.0 - min(max(float(drift), 0.0), 100.0) / 100.0

    novelty_class = agg.get("novelty_class", "novel")
    novelty_val = _NOVELTY_MAP.get(novelty_class, 1.0)

    # Step 1: weighted base score
    score = _W_PERF * perf_norm + _W_STAB * stability + _W_NOV * novelty_val

    # Step 2: non-linear novelty adjustment
    if novelty_class in _NOVELTY_PENALTY:
        score *= _NOVELTY_PENALTY[novelty_class]
    elif novelty_class in _NOVELTY_BONUS:
        score = min(score * _NOVELTY_BONUS[novelty_class], 1.0)

    # Step 3: stability guard
    if stability < _STABILITY_THRESHOLD:
        score *= _STABILITY_GUARD

    return round(score, 4)


def _top_features_from_registry(registry: list, k: int = TOP_K) -> List[dict]:
    """
    Return the top-k promoted features sorted by feature_score (descending).
    Falls back to most-recent promoted if no scores are stored yet.
    """
    promoted = [e for e in registry if e.get("verdict") == "promoted"]
    scored = [e for e in promoted if e.get("feature_score") is not None]

    if scored:
        ranked = sorted(scored, key=lambda e: e["feature_score"], reverse=True)
    else:
        ranked = list(reversed(promoted))  # fallback: most recent first

    return [
        {
            "name": e.get("name"),
            "level": e.get("level"),
            "feature_score": e.get("feature_score"),
            "action": e.get("action"),
            "motivation": e.get("motivation", ""),
        }
        for e in ranked[:k]
    ]


# ── Feature space coverage ────────────────────────────────────────────────────

def compute_coverage(registry: list) -> Dict[str, dict]:
    """
    Count how many times each (base_type, transform_type) combination has been tested.

    Uses the novelty module's fingerprint to infer structural types from registry entries.
    Applies low/medium/high labels relative to the most-tested combination.
    """
    from src.agent.novelty import _fingerprint

    counts: Dict[str, int] = {}
    for entry in registry:
        fp = _fingerprint(entry)
        if not fp["name"]:
            continue
        key = f"{fp['base_type']}+{fp['transform_type']}"
        counts[key] = counts.get(key, 0) + 1

    if not counts:
        return {}

    max_count = max(counts.values())
    result: Dict[str, dict] = {}
    for key, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        ratio = count / max_count
        coverage = "low" if ratio < 0.33 else ("medium" if ratio < 0.67 else "high")
        result[key] = {"count": count, "coverage": coverage}
    return result


def load_state() -> AgentState:
    """
    Load all research state from disk and return a typed AgentState.
    This is the single entrypoint for state initialisation in the loop.
    """
    context = load_context()
    principles = load_principles()
    registry = load_registry()
    active_features = load_active_features()

    stage = get_research_stage(active_features)
    counts = get_level_counts(active_features)

    # Extract rejection signals from recent registry entries (last 5, newest first)
    recent = list(reversed(registry[-5:])) if registry else []
    recent_rejection_reasons = []
    for entry in recent:
        if entry.get("verdict") == "rejected":
            recent_rejection_reasons.extend(entry.get("triggered_principles", []))

    recent_verdicts = [e.get("verdict", "unknown") for e in recent]

    promoted_count = sum(1 for e in registry if e.get("verdict") == "promoted")

    action_stats = compute_action_stats(registry, window=20)
    action_stats_by_stage = {
        s: compute_action_stats(registry, window=20, stage_filter=s)
        for s in ("primitive", "transform", "composite")
    }
    top_features = _top_features_from_registry(registry)
    feature_space_coverage = compute_coverage(registry)

    return AgentState(
        context=context,
        principles=principles,
        registry=registry,
        active_features=active_features,
        research_stage=stage,
        level_counts=counts,
        cycle_count=len(registry),
        promoted_count=promoted_count,
        recent_rejection_reasons=recent_rejection_reasons,
        recent_verdicts=recent_verdicts,
        action_stats=action_stats,
        action_stats_by_stage=action_stats_by_stage,
        top_features=top_features,
        feature_space_coverage=feature_space_coverage,
    )
