# FX-Agent

A structured, decision-guided feature discovery system for FX data.

---

## Overview

FX-Agent automates the process of proposing, testing, and evaluating candidate features for quantitative FX models. It is not an autonomous agent — it is a research pipeline in which a language model (Claude) makes structured decisions about what to explore next, while Python handles execution, evaluation, and state management.

The system replaces ad-hoc feature engineering with a reproducible, auditable loop that accumulates knowledge across cycles and uses past results to guide future decisions.

---

## CLI — FRONTIER

FX-Agent is operated via **FRONTIER**:

> **F**X **R**esearch for **O**ptimization and **N**avigation of **T**rading **I**nference via **E**xploratory **R**esearch

### Available Commands

```bash
frontier run --cycles N [--hint TEXT] [--data-dir PATH]
frontier status
frontier inspect <name>
frontier chat ["<question>"]
frontier init
```

### Command Overview

* **`run`** — execute one or more research cycles
* **`status`** — inspect registry state and active features
* **`inspect`** — view full details of a feature
* **`chat`** — interact with the system using grounded context
* **`init`** — configure data paths and API key

The CLI is a thin interface over the core research pipeline.

---

## Workflow

A typical research session follows a structured loop:

```text
Initialise → Run Cycles → Review Results → Refine Direction → Repeat
```

### Step-by-Step

**1) Initialise environment**

```bash
frontier init
```

Configure data paths and API access.

---

**2) Run research cycles**

```bash
frontier run -n 5
```

Each cycle executes the full feature discovery pipeline.

---

**3) Monitor progress**

```bash
frontier status
```

Inspect:

* number of cycles
* promoted vs rejected features
* active feature set

---

**4) Inspect features**

```bash
frontier inspect <feature_name>
```

Review hypothesis, construction, performance, and reasoning.

---

**5) Guide exploration (optional)**

```bash
frontier run -n 3 --hint "explore underrepresented feature types"
```

Hints can steer the direction of research without enforcing rules.

---

**6) Iterate**

```text
Run → Evaluate → Refine → Run
```

The system improves over time via accumulated feedback and structured state.

---

### Cycle Execution

```text
[State]
   ↓
[Decision (Claude)]
   ↓
[Feature Proposal]
   ↓
[Evaluation]
   ↓
[Scoring + Novelty]
   ↓
[Reasoning + Verdict]
   ↓
[Registry Update]
   ↓
[State (next cycle)]
```

---

## Core Concepts

### Pipeline

```
propose → test → score → log
```

1. **Propose** — select action and generate feature
2. **Test** — evaluate via walk-forward validation
3. **Score** — combine performance, stability, and novelty
4. **Log** — store full results in registry

---

### Decision-Making

The model selects one of:

* `explore_new_class`
* `refine_existing`
* `combine_features`
* `increase_robustness`

Decisions are informed by:

* past performance
* rejection patterns
* feature rankings
* coverage gaps

---

### Feedback Loop

After each cycle, the system updates:

* action performance statistics
* feature rankings
* feature space coverage

These signals guide future decisions.

---

## Architecture

* **AgentState** — central structured state (registry, stats, rankings, coverage)
* **Decision layer** — model selects actions and proposes features
* **Evaluation** — deterministic testing pipeline
* **Scoring** — prioritisation via performance, stability, novelty
* **Novelty detection** — prevents redundant exploration
* **Coverage tracking** — ensures exploration across feature space

---

## Feature Scoring

Each feature receives a bounded score:

```
score = performance + stability + novelty
```

With safeguards:

* duplicate penalty
* stability guard
* conservative novelty bonus

---

## Novelty Detection

Features are compared using structural fingerprints:

* base_type
* transform_type
* parameters and tokens

Classified as:

* **near_duplicate**
* **similar_family**
* **novel**

---

## Coverage Tracking

Exploration is tracked across:

```
(base_type, transform_type)
```

This biases the system toward underexplored regions.

---

## Design Principles

* **Transparency** — full audit trail of decisions and results
* **Reproducibility** — deterministic pipeline with explicit state
* **Minimalism** — no unnecessary abstractions or hidden systems
* **Separation of concerns** — model decides, Python executes

---

## Current Capabilities

* Decision-guided feature discovery
* Feedback-aware exploration
* Redundancy-aware search
* Coverage-aware exploration
* Persistent registry of all experiments
* CLI interface via FRONTIER

---

## Limitations

* Evaluation backend is fixed (not yet pluggable)
* Single-dataset workflow
* No automated regime detection
* No hyperparameter optimisation
* No unit tests
* Requires Anthropic API

---

## Future Work

* Pluggable evaluation backends
* Multi-dataset / multi-asset support
* Improved exploration strategies
* CLI enhancements
* Test coverage

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone <your-repo-url>
cd fx-agent
pip install -e .
```

This exposes the CLI as the `frontier` command.

### Ensure CLI is on PATH

If `frontier` is not recognised, add your Python user bin directory to your PATH.

For macOS (zsh):

```bash
export PATH="$HOME/Library/Python/3.9/bin:$PATH"
```

Then reload your shell:

```bash
source ~/.zshrc
```

Verify installation:

```bash
frontier --help
```
