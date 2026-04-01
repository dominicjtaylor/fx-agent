"""
novelty.py
Lightweight structural novelty check for candidate features.

Detects when a proposed feature is too similar to existing ones using three signals:
  - name_sim:          Jaccard similarity on normalised name tokens
  - construction_sim:  SequenceMatcher ratio on construction description
  - structural_match:  1 if both base_type and transform_type agree, else 0

Combined score: 0.4 * name_sim + 0.3 * construction_sim + 0.3 * structural_match

Thresholds:
  >= 0.85 → near_duplicate   (strong flag; skip test if no meaningful parameter change)
  >= 0.60 → similar_family   (allowed, but flagged for reasoning step)
  <  0.60 → novel

Design principles:
  - False negatives preferred: allow ambiguous cases through
  - Fully transparent: every flag names the specific conflicting feature
  - Zero latency: all comparisons are string ops on short texts
"""

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple


# ── Weights and thresholds ────────────────────────────────────────────────────

NAME_W = 0.4
CONSTRUCTION_W = 0.3
STRUCTURAL_W = 0.3

NEAR_DUPLICATE_THRESHOLD = 0.85
SIMILAR_FAMILY_THRESHOLD = 0.60

# How many recent registry entries to check (active features always checked fully)
REGISTRY_LOOKBACK = 15

# A parameter ratio > this is considered a "meaningful" change (e.g. window 60 vs 100)
MEANINGFUL_PARAM_RATIO = 1.5


# ── Structural type maps (keyword → type label) ───────────────────────────────

_BASE_TYPES: List[Tuple[str, List[str]]] = [
    ("volatility", ["vol", "volatility", "variance", "std", "atr", "range", "hl", "parkinson", "garman"]),
    ("returns",    ["return", "ret", "log_return", "momentum", "mom", "pct"]),
    ("spread",     ["spread", "diff", "difference", "gap", "bid", "ask"]),
    ("ratio",      ["ratio", "relative", "fraction"]),
    ("price",      ["price", "close", "open", "high", "low", "mid"]),
]

_TRANSFORM_TYPES: List[Tuple[str, List[str]]] = [
    # More specific transforms checked first; smoothing (rolling) is a broad catch-all
    ("normalisation",  ["zscore", "z_score", "rank", "percentile", "norm", "scaled", "scale"]),
    ("lagged",         ["lag", "shift", "delay", "previous", "past"]),
    ("smoothing",      ["mean", "average", "ewm", "ewma", "rolling", "smooth", "window"]),
    ("raw",            []),  # fallback — no transform keywords found
]


def _infer_type(text: str, type_map: List[Tuple[str, List[str]]]) -> str:
    """Return the first matching type label whose keywords appear in text, else last entry."""
    lowered = text.lower()
    for label, keywords in type_map[:-1]:   # skip fallback entry
        if any(kw in lowered for kw in keywords):
            return label
    return type_map[-1][0]   # fallback


# ── Fingerprint helpers ───────────────────────────────────────────────────────

def _get(obj: Any, key: str) -> str:
    """Attribute-safe getter — works on Pydantic models and dicts."""
    if hasattr(obj, key):
        return getattr(obj, key) or ""
    return obj.get(key, "") if isinstance(obj, dict) else ""


def _tokenize(text: str) -> set:
    """Lowercase alphanum tokens, filter single chars."""
    return {t for t in re.split(r'[^a-z0-9]+', text.lower()) if len(t) > 1}


def _extract_numbers(text: str) -> List[int]:
    return [int(n) for n in re.findall(r'\d+', text or "")]


def _fingerprint(entry: Any) -> Dict:
    name = _get(entry, "name")
    construction = _get(entry, "construction")
    combined = f"{name} {construction}"
    return {
        "name": name,
        "name_tokens": _tokenize(name),
        "construction": construction,
        "numbers": _extract_numbers(name) + _extract_numbers(construction),
        "base_type": _infer_type(combined, _BASE_TYPES),
        "transform_type": _infer_type(combined, _TRANSFORM_TYPES),
        "level": _get(entry, "level"),
    }


# ── Similarity ────────────────────────────────────────────────────────────────

def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _seq_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _similarity(fp_a: Dict, fp_b: Dict) -> float:
    name_sim = _jaccard(fp_a["name_tokens"], fp_b["name_tokens"])
    construction_sim = _seq_sim(fp_a["construction"], fp_b["construction"])
    structural_match = float(
        fp_a["base_type"] == fp_b["base_type"]
        and fp_a["transform_type"] == fp_b["transform_type"]
    )
    return NAME_W * name_sim + CONSTRUCTION_W * construction_sim + STRUCTURAL_W * structural_match


def _meaningful_param_change(cand_nums: List[int], exist_nums: List[int]) -> bool:
    """True if any numeric param differs by more than MEANINGFUL_PARAM_RATIO."""
    if not cand_nums or not exist_nums:
        return True   # no numbers → treat as different (don't suppress)
    for c, e in zip(sorted(cand_nums), sorted(exist_nums)):
        if e == 0:
            continue
        if max(c, e) / max(min(c, e), 1) > MEANINGFUL_PARAM_RATIO:
            return True
    return False


# ── Public API ────────────────────────────────────────────────────────────────

def check_novelty(
    candidate: Any,
    active_features: List[Any],
    registry: List[Dict],
) -> Dict:
    """
    Compare candidate against active features and recent registry entries.

    Returns a dict suitable for merging into TestResults.aggregate:
      novelty_score         — max combined similarity score [0, 1]
      novelty_class         — "novel" | "similar_family" | "near_duplicate"
      most_similar_feature  — name of the closest existing feature, or None
      novelty_flags         — list of "{class}:{name}" strings for each match found
      novelty_explanation   — human-readable summary
      skip_test             — True only if near_duplicate AND no meaningful param change
    """
    candidate_fp = _fingerprint(candidate)

    # Build pool: all active features + recent registry, deduplicated by name
    recent = registry[-REGISTRY_LOOKBACK:] if registry else []
    seen: set = set()
    pool: List[Any] = []
    for item in list(active_features) + list(recent):
        n = _get(item, "name")
        if n and n not in seen:
            seen.add(n)
            pool.append(item)

    max_score = 0.0
    most_similar_name: Optional[str] = None
    most_similar_fp: Optional[Dict] = None
    flags: List[str] = []

    for existing in pool:
        fp = _fingerprint(existing)
        if not fp["name"]:
            continue
        score = _similarity(candidate_fp, fp)
        if score > max_score:
            max_score = score
            most_similar_name = fp["name"]
            most_similar_fp = fp
        if score >= NEAR_DUPLICATE_THRESHOLD:
            flags.append(f"near_duplicate:{fp['name']}")
        elif score >= SIMILAR_FAMILY_THRESHOLD:
            flags.append(f"similar_family:{fp['name']}")

    # Deduplicate flags while preserving order
    flags = list(dict.fromkeys(flags))

    if max_score >= NEAR_DUPLICATE_THRESHOLD:
        novelty_class = "near_duplicate"
        skip_test = not _meaningful_param_change(
            candidate_fp["numbers"],
            most_similar_fp["numbers"] if most_similar_fp else [],
        )
        explanation = (
            f"Near-duplicate of '{most_similar_name}' (score {max_score:.2f}, "
            f"base_type={candidate_fp['base_type']}, transform_type={candidate_fp['transform_type']}). "
            + ("Test skipped — no meaningful parameter change detected."
               if skip_test else
               "Parameter change detected — test will run but improvement bar is high.")
        )
    elif max_score >= SIMILAR_FAMILY_THRESHOLD:
        novelty_class = "similar_family"
        skip_test = False
        explanation = (
            f"Similar family to '{most_similar_name}' (score {max_score:.2f}). "
            f"Allowed, but refinement must show material improvement over the existing feature."
        )
    else:
        novelty_class = "novel"
        skip_test = False
        explanation = f"No close matches found (max score {max_score:.2f}). Proceed normally."

    return {
        "novelty_score": round(max_score, 3),
        "novelty_class": novelty_class,
        "most_similar_feature": most_similar_name,
        "novelty_flags": flags,
        "novelty_explanation": explanation,
        "skip_test": skip_test,
    }
