# FX-Agent

A structured, decision-guided feature discovery system for FX volatility forecasting.

---

## Overview

FX-Agent automates the process of proposing, testing, and evaluating candidate features for FX volatility models. It is not an autonomous agent — it is a research pipeline in which a language model (Claude) makes structured decisions about what to explore next, while Python handles all execution, evaluation, and state management.

The system exists to replace ad-hoc feature engineering with a reproducible, auditable loop that accumulates knowledge across cycles and uses past results to guide future decisions.

---

## Core Concepts

### Pipeline

Each cycle follows four deterministic steps:

```
propose → test → score → log
```

1. **Propose** — Claude selects a research action and generates a candidate feature with a working Python implementation
2. **Test** — walk-forward LightGBM validation across chronological folds
3. **Score** — composite feature score computed from performance, stability, and novelty
4. **Log** — full entry (code, results, verdict, score) written to the registry

### Decision-Making

Before proposing a feature, Claude explicitly selects one of four research actions:

| Action | When to use |
|---|---|
| `explore_new_class` | No features in this region yet, or repeated failures in same family |
| `refine_existing` | Weak but consistent signal — tighten parameterisation |
| `combine_features` | Two complementary weak signals — build interaction or regime flag |
| `increase_robustness` | Strong but unstable feature — add guards, widen windows |

The action choice is grounded in structured state: recent verdicts, rejection patterns, action performance history, feature scores, and coverage gaps.

### Feedback Loop

After each cycle, the registry is updated and `AgentState` is reloaded. Future cycles see:

- Per-action success rates and improvement averages (last 20 cycles, overall and per research stage)
- Top-ranked promoted features by composite score
- Feature space coverage — which structural regions are over- or under-explored

This feedback is provided to Claude as soft guidance, not enforced as rules.

---

## Architecture

### AgentState

Loaded from disk at the start of each cycle. Contains:

```python
context            # data config: pairs, frequency, horizon
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

Features are tested using walk-forward LightGBM validation:

- 5 chronological folds, expanding training window
- Minimum 50% training data per fold
- Baseline: rolling volatility features only
- Comparison: baseline model vs baseline + candidate feature
- Metrics: RMSE improvement (%), candidate importance (%), importance drift across folds

Feature code runs in a sandboxed namespace (numpy and pandas only). Forbidden patterns (os, sys, exec, open, etc.) are rejected before execution.

### Feature Scoring

Each tested feature receives a composite score in [0, 1]:

```
base_score = 0.5 × perf_norm + 0.3 × stability + 0.2 × novelty_val

perf_norm  = normalised RMSE improvement, clipped to [-2%, +5%] range
stability  = 1 - mean_importance_drift / 100
novelty_val = novel → 1.0 | similar_family → 0.4 | near_duplicate → 0.0
```

Post-processing adjustments (multiplicative):

- `near_duplicate` → `score × 0.5` — prevents duplicates dominating the ranking even with high performance
- `novel` → `score × 1.1`, capped at 1.0 — small exploration bonus
- `stability < 0.5` → `score × 0.7` — guards against noisy features

Scores are stored in the registry and used to rank promoted features for future cycles.

### Novelty Detection

Before testing, each candidate is compared against active features and the last 15 registry entries using three structural signals:

```
similarity = 0.4 × name_sim + 0.3 × construction_sim + 0.3 × structural_match
```

- `name_sim` — Jaccard similarity on normalised name tokens
- `construction_sim` — SequenceMatcher ratio on construction description
- `structural_match` — 1 if both `base_type` and `transform_type` agree, else 0

`base_type` and `transform_type` are inferred by keyword matching (e.g. "vol", "variance" → `volatility`; "zscore", "rank" → `normalisation`).

Classification thresholds:

| Score | Class | Behaviour |
|---|---|---|
| ≥ 0.85 | `near_duplicate` | Flagged; test skipped if no meaningful parameter change |
| 0.60–0.85 | `similar_family` | Allowed; flagged for reasoning step |
| < 0.60 | `novel` | Proceeds normally |

Novelty class and explanation are passed to the reasoning step. Claude is instructed to reject near-duplicates unless they demonstrate clearly superior performance with a distinct mechanism.

### Coverage Tracking

The system counts how many features have been tested per `(base_type, transform_type)` combination and labels each as `low`, `medium`, or `high` coverage relative to the most-tested combination. This is exposed to Claude to bias exploration toward under-represented structural regions.

---

## Feature Discovery Process

A single cycle:

1. `load_state()` — reads all persistent state from disk into `AgentState`
2. `propose_feature(state)` — Claude selects action + proposes feature; receives full state context including action stats, top features, coverage, recent rejections
3. `check_novelty(feature, active_features, registry)` — computes similarity score and classification; result merged into test results
4. `run_feature_test(feature, context, data, active_features)` — walk-forward LightGBM validation (skipped if near_duplicate with no parameter change)
5. `compute_feature_score(aggregate)` — composite score written to test results
6. `reason_and_verdict(feature, test_results, principles, context)` — Claude evaluates results against research principles (P01–P06) and issues `promoted`, `rejected`, or `modified`
7. `make_entry(feature, test_results, reasoning)` — canonical registry entry constructed
8. `save_entry(entry)` — appended to `outputs/registry.json`
9. If promoted: `add_active_feature(feature)` — added to `outputs/active_features.json` for inclusion in all future cycles

### Research Stage Progression

Features are organised in a three-level hierarchy:

- **Primitive** — direct measurements from price series (rolling vol, ATR, high-low range)
- **Transform** — statistical transformations of existing primitives (z-score, percentile rank)
- **Composite** — combinations of primitives and transforms (regime flags, interaction terms)

The system progresses through stages automatically based on what has been promoted. Hierarchy violations trigger automatic rejection.

### Research Principles

Six rejection criteria (P01–P06) are applied at the reasoning step:

| ID | Principle |
|---|---|
| P01 | Regime robustness — must hold across different market conditions |
| P02 | No look-ahead — strictly backward-looking at all horizons |
| P03 | Importance stability — consistent across folds |
| P04 | OOS improvement — must improve on LightGBM baseline out-of-sample |
| P05 | Economic motivation — grounded in market microstructure |
| P06 | Signal persistence — importance must not decay monotonically across folds |

---

## Design Principles

**Transparency** — every decision is logged with full rationale. The registry contains feature code, test results, triggered principles, novelty assessment, composite score, and Claude's reasoning.

**Reproducibility** — the pipeline is deterministic given fixed inputs. All randomness is confined to LightGBM training (fixed seed). No hidden state.

**Minimalism** — no embeddings, no correlation matrices, no reinforcement learning. All signals are computed from short text strings and scalar metrics. The system is inspectable without tooling.

**Separation of concerns** — Claude makes decisions and reasons about results; Python executes, measures, and persists. Neither role bleeds into the other.

---

## Current Capabilities

- Propose, test, and evaluate candidate features in a single structured cycle
- Walk-forward LightGBM validation with per-fold diagnostics
- Novelty detection using structural fingerprints (no correlation computation)
- Composite feature scoring with duplicate penalty and stability guard
- Coverage tracking across structural feature types
- Per-action performance feedback (success rate, average improvement, stability)
- Full audit trail in `outputs/registry.json`
- Promoted feature accumulation in `outputs/active_features.json` — active features are re-included in every subsequent test
- Hierarchy enforcement: primitives before transforms before composites
- Sandboxed execution of LLM-generated feature code

---

## Limitations

- **No CLI** — the system is invoked via `scripts/run_cycle.py`. A command-line interface is planned but not yet implemented.
- **Single data source** — currently configured for one currency pair per cycle. Multi-pair joint testing is not supported.
- **No regime-awareness** — P01 (regime robustness) is evaluated qualitatively by Claude against known structural breaks in the config. There is no automated regime detection or regime-conditional evaluation.
- **No stronger baselines** — the baseline is rolling volatility features only. EWMA and GARCH baselines are not implemented.
- **No test suite** — correctness is verified by running `scripts/run_cycle.py` end-to-end. Unit tests do not exist.
- **API dependency** — the propose and reason steps require an active Anthropic API key and incur latency and cost per cycle.
- **No hyperparameter tuning** — LightGBM parameters are fixed. The system does not search over model configurations.

---

## Future Work

- CLI for cycle control, registry inspection, and research chat
- Multi-pair joint validation (test features across pairs simultaneously)
- Regime-conditional evaluation to address P01 rejection patterns
- EWMA/GARCH baselines for contextualising LightGBM gains
- Structured exploration schedule to enforce coverage targets
- Unit tests for core pipeline components
