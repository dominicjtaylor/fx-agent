# FX-Agent

A structured, decision-guided feature discovery system for FX data.

---

## Overview

FX-Agent automates the process of proposing, testing, and evaluating candidate features for quantitative FX models. It is not an autonomous agent — it is a research pipeline in which a language model (Claude) makes structured decisions about what to explore next, while Python handles all execution, evaluation, and state management.

The system exists to replace ad-hoc feature engineering with a reproducible, auditable loop that accumulates knowledge across cycles and uses past results to guide future decisions.

---

## Core Concepts

### Pipeline

Each cycle follows four deterministic steps:

```
propose → test → score → log
```

1. **Propose** — Claude selects a research action and generates a candidate feature with a working Python implementation
2. **Test** — walk-forward model validation across chronological folds
3. **Score** — composite feature score computed from performance, stability, and novelty
4. **Log** — full entry (code, results, verdict, score) written to the registry

### Decision-Making

Before proposing a feature, Claude explicitly selects one of four research actions:

| Action                | When to use                                                         |
| --------------------- | ------------------------------------------------------------------- |
| `explore_new_class`   | No features in this region yet, or repeated failures in same family |
| `refine_existing`     | Weak but consistent signal — tighten parameterisation               |
| `combine_features`    | Two complementary weak signals — build interaction or regime flag   |
| `increase_robustness` | Strong but unstable feature — add guards, widen windows             |

The action choice is grounded in structured state: recent verdicts, rejection patterns, action performance history, feature scores, and coverage gaps.

### Feedback Loop

After each cycle, the registry is updated and `AgentState` is reloaded. Future cycles see:

* Per-action success rates and improvement averages (last 20 cycles, overall and per research stage)
* Top-ranked promoted features by composite score
* Feature space coverage — which structural regions are over- or under-explored

This feedback is provided to Claude as soft guidance, not enforced as rules.

---

## Architecture

### AgentState

Loaded from disk at the start of each cycle. Contains:

```python
context            # data config: instruments, frequency, horizon
principles         # research principles P01–P06
registry           # full audit log of all tested features
active_features    # promoted features included in all future tests
research_stage     # "primitive" | "transform" | "composite"
level_counts       # count of active features by hierarchy level
cycle_count        # total entries in registry
promoted_count     # number of promoted entries
recent_verdicts    # verdicts from last 5 cycles (newest first)
recent_rejection_reasons  # triggered principles from recent rejections
action_stats       # per-action metrics (count, success_rate, avg_improvement, stability_score)
action_stats_by_stage     # same, filtered to current research stage
top_features       # top-K promoted features by feature_score
feature_space_coverage    # {base_type+transform_type: count + low/medium/high label}
```

### Decision Layer

The propose step is a single Claude API call that receives the full `AgentState` context and must output both an action choice (with rationale) and a feature proposal (with implementation) in one structured JSON response. There is no separate decision module — the action and proposal are computed together.

### Evaluation

Features are tested using walk-forward validation:

* Chronological folds with expanding training windows
* Minimum training fraction enforced per fold
* Baseline vs baseline + candidate feature comparison
* Metrics include:

  * performance improvement (e.g. RMSE or task-specific metric)
  * feature importance
  * importance drift across folds

Feature code runs in a sandboxed namespace (numpy and pandas only). Forbidden patterns (os, sys, exec, open, etc.) are rejected before execution.

### Feature Scoring

Each tested feature receives a composite score in [0, 1]:

```
base_score = 0.5 × perf_norm + 0.3 × stability + 0.2 × novelty_val

perf_norm   = normalised performance improvement (task-specific metric)
stability   = 1 - mean_importance_drift / 100
novelty_val = novel → 1.0 | similar_family → 0.4 | near_duplicate → 0.0
```

Post-processing adjustments (multiplicative):

* `near_duplicate` → `score × 0.5` — prevents duplicates dominating the ranking
* `novel` → `score × 1.1`, capped at 1.0 — small exploration bonus
* `stability < 0.5` → `score × 0.7` — guards against noisy features

Scores are stored in the registry and used to rank promoted features for future cycles.

### Novelty Detection

Before testing, each candidate is compared against active features and the last 15 registry entries using three structural signals:

```
similarity = 0.4 × name_sim + 0.3 × construction_sim + 0.3 × structural_match
```

* `name_sim` — Jaccard similarity on normalised name tokens
* `construction_sim` — SequenceMatcher ratio on construction description
* `structural_match` — 1 if both `base_type` and `transform_type` agree, else 0

`base_type` and `transform_type` are inferred by keyword matching (e.g. "returns", "spread", "volatility"; "rolling", "zscore", "lag").

Classification thresholds:

| Score     | Class            | Behaviour                                               |
| --------- | ---------------- | ------------------------------------------------------- |
| ≥ 0.85    | `near_duplicate` | Flagged; test skipped if no meaningful parameter change |
| 0.60–0.85 | `similar_family` | Allowed; flagged for reasoning step                     |
| < 0.60    | `novel`          | Proceeds normally                                       |

Novelty class and explanation are passed to the reasoning step. Claude is instructed to reject near-duplicates unless they demonstrate clearly superior performance with a distinct mechanism.

### Coverage Tracking

The system counts how many features have been tested per `(base_type, transform_type)` combination and labels each as `low`, `medium`, or `high` coverage relative to the most-tested combination. This is exposed to Claude to bias exploration toward under-represented structural regions.

---

## Feature Discovery Process

A single cycle:

1. `load_state()` — reads all persistent state from disk into `AgentState`
2. `propose_feature(state)` — Claude selects action + proposes feature
3. `check_novelty(...)` — computes similarity score and classification
4. `run_feature_test(...)` — walk-forward validation (skipped if redundant)
5. `compute_feature_score(...)` — composite score assigned
6. `reason_and_verdict(...)` — Claude evaluates against research principles
7. `make_entry(...)` — canonical registry entry constructed
8. `save_entry(...)` — appended to `outputs/registry.json`
9. If promoted → added to `outputs/active_features.json`

### Research Stage Progression

Features are organised in a three-level hierarchy:

* **Primitive** — direct measurements from price series
* **Transform** — statistical transformations of primitives
* **Composite** — combinations and interactions

The system progresses through stages automatically based on promoted features.

---

## Design Principles

**Transparency** — all decisions, scores, and reasoning are logged
**Reproducibility** — deterministic pipeline with explicit state
**Minimalism** — no embeddings, RL, or opaque heuristics
**Separation of concerns** — Claude decides; Python executes

---

## Current Capabilities

* Structured propose → test → evaluate → log loop
* Walk-forward validation with fold-level diagnostics
* Novelty detection via structural fingerprints
* Feature ranking with duplicate penalty and stability guard
* Coverage-aware exploration guidance
* Action-level performance feedback
* Full audit trail (`outputs/registry.json`)
* Persistent active feature set across cycles
* Hierarchical feature construction
* Sandboxed execution of generated feature code

---

## Limitations

* **No CLI** — currently run via script; CLI planned
* **Single-dataset setup** — not yet generalised to multiple datasets or assets
* **Model dependency** — evaluation currently tied to a fixed model pipeline
* **No automated regime detection**
* **No hyperparameter search**
* **No unit tests**
* **Requires Anthropic API access**

---

## Future Work

* CLI for experiment control and inspection
* Multi-dataset / multi-asset support
* Pluggable evaluation backends (model-agnostic testing)
* Regime-aware evaluation
* Improved search strategy (coverage + prioritisation control)
* Test suite for core components
