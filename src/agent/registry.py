"""
registry.py
Manages the persistent hypothesis registry.
Every feature tested gets a full audit entry written to outputs/registry.json.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

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


def make_entry(
    feature: dict,
    test_results: dict,
    reasoning: dict
) -> dict:
    """
    Construct a complete registry entry from proposal, test results, and reasoning.
    This is the canonical structure for every entry in the registry.
    """
    return {
        "id": f"feat_{uuid.uuid4().hex[:6]}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "name": feature.get("name"),
        "level": feature.get("level"),
        "motivation": feature.get("motivation"),
        "hypothesis": feature.get("hypothesis"),
        "construction": feature.get("construction"),
        "code": feature.get("code"),
        "rejection_criteria": feature.get("rejection_criteria"),
        "test_results": test_results,
        "triggered_principles": reasoning.get("triggered_principles", []),
        "verdict": reasoning.get("verdict"),
        "summary": reasoning.get("summary"),
        "next_action": reasoning.get("next_action"),
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