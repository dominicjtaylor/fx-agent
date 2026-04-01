"""
schemas.py
Pydantic v2 models for all data contracts between agent modules.
"""

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict


class FeatureProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    triggered_principles: list[str]
    next_action: str


class TestResults(BaseModel):
    """
    Structured test results from walk-forward LightGBM validation.
    aggregate and per_pair values kept as dicts to avoid schema churn
    as the validation pipeline evolves.
    """
    model_config = ConfigDict(extra="forbid")

    per_pair: dict[str, dict[str, Any]]
    aggregate: dict[str, Any]