"""
active_features.py
Manages the active feature set — the set of promoted features
that are included in every LightGBM training run.
Persists to outputs/active_features.json.
"""

import json
from pathlib import Path
from typing import List

ACTIVE_FEATURES_PATH = Path(__file__).resolve().parents[2] / "outputs" / "active_features.json"


def load_active_features() -> List[dict]:
    """Load the current active feature set. Returns empty list if none yet."""
    if not ACTIVE_FEATURES_PATH.exists():
        return []
    with open(ACTIVE_FEATURES_PATH, "r") as f:
        return json.load(f)


def save_active_features(features: List[dict]) -> None:
    """Save the full active feature set to disk."""
    ACTIVE_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_FEATURES_PATH, "w") as f:
        json.dump(features, f, indent=2)


def add_active_feature(feature: dict) -> None:
    """Add a promoted feature to the active set."""
    active = load_active_features()
    if not any(f["name"] == feature["name"] for f in active):
        active.append({
            "name": feature["name"],
            "code": feature["code"],
            "level": feature.get("level", "primitive"),
        })
        save_active_features(active)


def get_active_feature_names() -> List[str]:
    """Return just the names of active features."""
    return [f["name"] for f in load_active_features()]


def get_level_counts(active_features: List[dict]) -> dict:
    """Return count of features at each hierarchy level."""
    counts = {"primitive": 0, "transform": 0, "composite": 0}
    for f in active_features:
        level = f.get("level", "primitive")
        if level in counts:
            counts[level] += 1
    return counts


def get_research_stage(active_features: List[dict]) -> str:
    """
    Determine the current research stage based on what levels are established.

    Rules:
    - primitive  — no Level 1 features promoted yet
    - transform  — Level 1 exists, but no Level 2
    - composite  — Level 1 and Level 2 both exist
    """
    counts = get_level_counts(active_features)
    if counts["primitive"] == 0:
        return "primitive"
    elif counts["transform"] == 0:
        return "transform"
    else:
        return "composite"


def format_active_features_for_prompt(active_features: List[dict]) -> str:
    """Concise summary for inclusion in Claude prompts, grouped by level."""
    if not active_features:
        return "No features currently in the active set."
    counts = get_level_counts(active_features)
    lines = [
        f"Active features: {len(active_features)} total "
        f"({counts['primitive']} primitive, {counts['transform']} transform, {counts['composite']} composite)"
    ]
    for level in ("primitive", "transform", "composite"):
        feats = [f for f in active_features if f.get("level", "primitive") == level]
        if feats:
            lines.append(f"  {level.capitalize()}s: " + ", ".join(f["name"] for f in feats))
    return "\n".join(lines)
