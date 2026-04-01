# Project Overview

A structured, decision-guided AI system for discovering predictive features in FX data using a reproducible propose → test → reason → log research loop powered by Claude and a deterministic evaluation pipeline.

---

# Original Idea

Automate the quantitative research process for FX signals.

Instead of hand-crafting features, an LLM-driven system proposes candidate features, evaluates them via walk-forward validation, and builds a validated feature registry through structured promotion/rejection decisions.

---

# Current State

### Core System

* Core agent loop implemented (`loop.py`): propose → test → reason → log (multi-step pipeline)
* Feature proposal via Claude API (`propose.py`): structured JSON output including feature definition and implementation
* Feature testing engine (`test.py`): walk-forward validation with chronological folds (baseline vs baseline + candidate comparison)
* Reasoning/verdict module (`reason.py`): Claude evaluates results against research principles and assigns promoted/rejected/modified
* Registry management (`registry.py`): persistent audit log in `outputs/registry.json`
* Active feature tracking (`active_features.py`): hierarchical structure (primitive → transform → composite), persisted to `outputs/active_features.json`
* Safety validation: blocks unsafe patterns in LLM-generated code prior to execution

---

### Search & Decision Enhancements (Recent)

* Model-driven decision layer (Claude selects action explicitly)
* Action performance tracking (`action_stats`, per-stage stats)
* Feature novelty detection (fingerprinting + similarity scoring)
* Feature ranking (performance + stability + novelty with safeguards)
* Feature space coverage tracking (base_type × transform_type)

---

### CLI (FRONTIER)

* CLI implemented at `src/agent/cli.py`, exposed as `frontier`
* Supports:

  * running cycles
  * inspecting registry
  * viewing features
  * basic research interaction

**Status:**

* CLI is functional and installable via `pip install -e .`
* Not yet fully validated against the latest agent infrastructure
* May contain assumptions from earlier volatility-specific pipeline
* Requires testing and potential alignment with:

  * updated AgentState
  * evaluation outputs
  * registry schema

---

### Configuration

* Config-driven design:

  * `config/context.yaml` — data context
  * `config/principles.yaml` — research principles

**Status:**

* Expected by system but not yet fully standardised or verified in repo

---

### Missing / Incomplete

* No comprehensive test suite
* Evaluation backend not yet modular/pluggable
* CLI integration not fully stabilised
* Limited validation across multiple datasets/instruments

---

# Goal

A robust, reproducible feature discovery system that:

* systematically explores the feature space
* prioritises high-quality signals via ranking and feedback
* avoids redundant exploration via novelty detection
* maintains full transparency via a persistent registry

Ultimately enabling:

> a continuously improving library of validated features for quantitative modelling.

---

# Constraints

* Features must be strictly backward-looking (no look-ahead)
* Rolling windows capped (unless justified)
* LLM-generated code restricted to numpy/pandas
* Walk-forward validation with chronological splits
* Hierarchical feature construction enforced:

  * primitives → transforms → composites

---

# Next Steps

1. **Stabilise CLI integration (FRONTIER)**

   * test all commands against current agent state
   * resolve schema mismatches
   * remove any residual pipeline-specific assumptions

2. **Run end-to-end validation cycles**

   * confirm registry integrity
   * validate scoring and novelty behaviour
   * check decision quality over multiple cycles

3. **Verify configuration layer**

   * ensure `context.yaml` and `principles.yaml` exist and are consistent
   * standardise required fields

4. **Decouple evaluation backend**

   * move toward model-agnostic evaluation interface

5. **Introduce basic test coverage**

   * registry integrity
   * scoring logic
   * novelty classification

6. **Refine search behaviour**

   * balance exploration vs exploitation
   * monitor coverage distribution
