"""
context.py
Loads and saves data context and research principles from config files.
All agent modules read from here rather than directly from disk.
"""

import re
import yaml
from pathlib import Path

# Resolve config directory relative to repo root
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_context() -> dict:
    """Load data context from config/context.yaml."""
    path = CONFIG_DIR / "context.yaml"
    if not path.exists():
        raise FileNotFoundError(f"context.yaml not found at {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_principles() -> list:
    """Load research principles from config/principles.yaml."""
    path = CONFIG_DIR / "principles.yaml"
    if not path.exists():
        raise FileNotFoundError(f"principles.yaml not found at {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("principles", [])


def save_context(context: dict) -> None:
    """Write updated context back to config/context.yaml."""
    path = CONFIG_DIR / "context.yaml"
    with open(path, "w") as f:
        yaml.dump(context, f, default_flow_style=False, allow_unicode=True)


def format_context_for_prompt(context: dict) -> str:
    """
    Returns a concise plain-text summary of the data context
    suitable for inclusion in a Claude prompt.
    """
    pairs = ", ".join(context["data"]["pairs"])
    freq = context["data"]["frequency"]
    start = context["data"]["time_range"]["start"]
    end = context["data"]["time_range"]["end"]
    horizon = context["data"]["horizon_seconds"]

    m = re.match(r'(\d+)s', freq)
    freq_seconds = int(m.group(1)) if m else 10
    horizon_bars = horizon // freq_seconds

    regimes = "\n".join(
        f"  - {r['name']}: {r['description']}"
        for r in context.get("known_regimes", [])
    )

    breaks = "\n".join(
        f"  - {b['date']}: {b['description']}"
        for b in context.get("structural_breaks", [])
    )

    flags = "\n".join(
        f"  - {f}" for f in context.get("quality_flags", [])
    )

    return f"""
DATA CONTEXT
------------
Pairs: {pairs}
Frequency: {freq}
Time range: {start} to {end}
Forecast horizon: {horizon} seconds ({horizon_bars} bars at {freq} resolution)

Known regimes:
{regimes}

Structural breaks:
{breaks}

Data quality flags:
{flags}
""".strip()


def format_principles_for_prompt(principles: list) -> str:
    """
    Returns a concise plain-text summary of research principles
    suitable for inclusion in a Claude prompt.
    """
    lines = []
    for p in principles:
        lines.append(f"[{p['id']}] {p['name']}")
        lines.append(f"  {p['description'].strip()}")
        lines.append(f"  Rejection trigger: {p['rejection_trigger']}")
        lines.append("")
    return "\n".join(lines).strip()