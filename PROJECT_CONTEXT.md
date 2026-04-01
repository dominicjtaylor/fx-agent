# Project Overview
An autonomous AI agent that systematically discovers predictive features for FX volatility forecasting using a structured propose → test → reason → log research loop powered by Claude and LightGBM.

# Original Idea
Automate the quant research process for FX volatility signals. Instead of hand-crafting features, an LLM-driven agent proposes candidate features, tests them with walk-forward validation against a LightGBM baseline, and builds up a validated feature registry through structured promotion/rejection decisions.

# Current State
- Core agent loop implemented (`loop.py`): propose → test → reason → log, 10-step pipeline
- Feature proposal via Claude API (`propose.py`): structured JSON output with code generation
- Feature testing engine (`test.py`): walk-forward LightGBM validation, 5 chronological folds, compares full model vs baseline and naive rolling vol
- Reasoning/verdict module (`reason.py`): Claude evaluates results against principles, issues promoted/rejected/modified verdict
- Registry management (`registry.py`): persistent audit log in `outputs/registry.json`
- Active feature tracking (`active_features.py`): 3-level hierarchy (primitive → transform → composite), persists to `outputs/active_features.json`
- Safety validation in `test.py`: blocks forbidden patterns in LLM-generated code before exec
- Config-driven: `config/context.yaml` (data context) and `config/principles.yaml` (research principles) — not yet in repo
- No entrypoint/runner script visible yet; no config files, no outputs directory, no tests

# Goal
A fully autonomous feature research loop that iteratively builds a validated library of FX volatility features, starting from primitives, advancing through transforms and composites, with every decision logged and auditable.

# Constraints
- No volume data — OHLC only
- Features must be strictly backward-looking (no look-ahead)
- Rolling windows capped at 360 bars unless explicitly justified
- LLM-generated code restricted to numpy/pandas only
- Walk-forward validation: 5 chronological folds, expanding window, min 50% training data
- Hierarchy enforced: primitives before transforms before composites

# Next Steps
1. Verify config files exist (`config/context.yaml`, `config/principles.yaml`) — required to run any cycle
2. Identify and create/locate the entrypoint script that loads data and calls `run_cycle()`
3. Confirm data loading pipeline — where does the OHLC data come from and in what format
4. Run a first cycle end-to-end and validate output structure
5. Review/tune promotion thresholds and principle definitions once first results are available
