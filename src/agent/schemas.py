"""
schemas.py
Pydantic v2 models for all data contracts between agent modules.
"""

from typing import Any, List, Literal
from pydantic import BaseModel, ConfigDict

# The four research actions Claude must choose between before proposing.
Action = Literal[
    "explore_new_class",    # No features in this class yet, or repeated failures → try something new
    "refine_existing",      # Weak but consistent signal → tighten parameterisation
    "combine_features",     # Complementary weak signals → interaction or regime flag
    "increase_robustness",  # Strong but unstable → add guards, widen windows, clip extremes
]


class FeatureProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Decision fields — model must choose action BEFORE designing the feature
    action: Action
    action_rationale: str  # why this action given current registry state

    # Feature fields — unchanged
    name: str
    level: Literal["primitive", "transform", "composite"]
    motivation: str
    hypothesis: str
    construction: str
    code: str
    rejection_criteria: str
    rationale: str


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["promoted", "rejected", "modified"]
    summary: str
    triggered_principles: List[str]
    next_action: str


class TestResults(BaseModel):
    """
    Structured test results from walk-forward LightGBM validation.
    aggregate and per_pair values kept as dicts to avoid schema churn
    as the validation pipeline evolves.
    """
    model_config = ConfigDict(extra="forbid")

    per_pair: dict
    aggregate: dict