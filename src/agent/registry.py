"""
registry.py
Manages the persistent hypothesis registry.
Every feature tested gets a full audit entry written to outputs/registry.json.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from src.agent.schemas import FeatureProposal, Verdict, TestResults

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "outputs" / "registry.json"


def load_registry() -> list:
    """Load the full registry from disk. Returns empty list if registry doesn't exist yet."""
    if not REGISTRY_PATH.exists():
        return []
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def save_entry(entry: dict) -> None:
    """Append a new feature entry to the registry."""
    registry = load_registry()
    registry.append(entry)
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def get_entry(feature_id: str) -> Optional[dict]:
    """Retrieve a specific entry by feature ID. Returns None if not found."""
    registry = load_registry()
    for entry in registry:
        if entry.get("id") == feature_id:
            return entry
    return None


def filter_by_verdict(verdict: str) -> list:
    """Return all entries matching a given verdict: promoted, rejected, or modified."""
    registry = load_registry()
    return [e for e in registry if e.get("verdict") == verdict]


def make_entry(feature: Any, test_results: Any, reasoning: Any) -> dict:
    """
    Construct a complete registry entry from proposal, test results, and reasoning.

    Accepts Pydantic model instances (FeatureProposal, TestResults, Verdict) or
    plain dicts — uses getattr with fallback to support both.
    """
    def _get(obj, key, default=None):
        if hasattr(obj, key):
            return getattr(obj, key)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    # Serialise test_results: Pydantic model → dict, or pass dict through
    if hasattr(test_results, "model_dump"):
        test_results_dict = test_results.model_dump()
    elif isinstance(test_results, dict):
        test_results_dict = test_results
    else:
        test_results_dict = {}

    return {
        "id": f"feat_{uuid.uuid4().hex[:6]}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        # Decision fields (new)
        "action": _get(feature, "action"),
        "action_rationale": _get(feature, "action_rationale"),
        # Feature fields
        "name": _get(feature, "name"),
        "level": _get(feature, "level"),
        "motivation": _get(feature, "motivation"),
        "hypothesis": _get(feature, "hypothesis"),
        "construction": _get(feature, "construction"),
        "code": _get(feature, "code"),
        "rejection_criteria": _get(feature, "rejection_criteria"),
        # Results and verdict
        "test_results": test_results_dict,
        "triggered_principles": _get(reasoning, "triggered_principles", []),
        "verdict": _get(reasoning, "verdict"),
        "summary": _get(reasoning, "summary"),
        "next_action": _get(reasoning, "next_action"),
    }


def format_registry_for_prompt(registry: list, max_entries: int = 5) -> str:
    """
    Returns a concise summary of recent registry entries
    suitable for inclusion in a Claude prompt.
    Limits to most recent max_entries to avoid bloating the context window.
    """
    if not registry:
        return "No features tested yet."

    recent = registry[-max_entries:]
    lines = []
    for e in recent:
        lines.append(f"- {e['name']} [{e.get('verdict', 'unknown').upper()}]: {e.get('hypothesis', '')[:100]}")
    return "\n".join(lines)