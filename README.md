# Knowledge Integrity & Expert Verification Engine

## Overview

Traditional hiring assumes that correct answers imply real expertise.

That assumption no longer holds.

With modern LLMs, candidates can generate convincing answers and polished profiles without having actually operated systems in production. The failure mode is subtle: not incorrect answers, but **lack of depth disguised as competence**.

This project addresses a new class of fraud:

> Not identity fraud — but **expertise inflation**.

---

## Core Thesis

The difference between a real expert and an LLM-assisted candidate is not correctness.

It is the **depth gradient of their reasoning**.

Real experts:

- reference constraints, trade-offs, and failure modes
- expose system-level thinking such as latency, scaling, and edge cases
- remain consistent under deeper probing

LLM-assisted candidates:

- produce structurally correct but generic answers
- lack operational detail
- collapse when pushed beyond surface-level explanations

This system is designed to capture that difference.

---

## Problem Framing

We model candidate evaluation as a **sequential decision process**, not a static classification task.

Instead of asking:

> "Is this candidate good or bad?"

We ask:

> "What is the most informative next action to reduce uncertainty?"

This framing is closer to how strong interviewers and hiring panels actually work in practice.

---

## System Architecture

### State Representation

Each environment step encodes a fixed-size feature vector built from signals such as:

- **Profile Delta** — sudden skill inflation over time
- **Scar Score** — evidence of real-world operational detail
- **Genericity Score** — LLM-like phrasing patterns
- **Consistency Score** — alignment between profile and answers
- **Web Signals Score** — external validation proxy
- **Volatility** — disagreement between signals
- **Fraud Score** — aggregated suspicion signal
- **Question Ratio** — how deep the interaction has gone
- **Last Action Encoding**
- **Uncertainty** — confidence of the current estimate

### Fraud Score

```python
fraud_score = (
    0.30 * profile_delta
    + 0.25 * genericity
    + 0.20 * web_score
    + 0.10 * volatility
    - 0.10 * scar
    - 0.15 * consistency
)
```

### Interpretation

- **High score** → stronger indicators of fabricated expertise
- **Low score** → stronger consistency with real-world experience

This helps avoid signal cancellation by explicitly separating suspicion signals from credibility signals.

---

## Action Space

The system does not immediately classify candidates.

It decides **how to investigate them**.

### Available Actions

- `PASS`
- `FLAG`
- `ESCALATE`
- `CHECK_WEB`
- `ASK_DEEP_RAG`

### Design Principle

Reinforcement Learning is used for:

> **Question and investigation strategy optimization — not final classification alone**

This is important because expertise is revealed through interaction, not through a single response.

---

## Current Implementation

This repository includes:

- a custom Gymnasium environment
- a heuristic baseline agent
- a trained DQN policy
- a synthetic dataset with `fraud`, `legit`, and `gray` cases
- a side-by-side demo comparing baseline vs trained DQN
- an evaluation script for repeated seeded runs
- saved evaluation and training metrics
- unit tests for scoring-related components

---

## Results

Controlled evaluation over **300 episodes** shows that the trained DQN improves policy quality while preserving fraud coverage.

### Heuristic Baseline

- **Average reward:** `1.8316`
- **Legit pass rate:** `0.4466`
- **Fraud flag rate:** `1.0`
- **Gray investigate rate:** `1.0`

### Trained DQN

- **Average reward:** `2.2748`
- **Legit pass rate:** `1.0`
- **Fraud flag rate:** `1.0`
- **Gray investigate rate:** `1.0`

### Key Insight

The trained DQN:

- preserves fraud detection performance
- preserves investigation coverage for gray-zone profiles
- reduces false-positive escalation on legitimate candidates
- replaces generic escalation with more targeted next-step actions such as:
  - `CHECK_WEB`
  - `ASK_DEEP_RAG`

In other words, the learned policy is not only identifying suspicious patterns — it is learning **how to investigate them more efficiently**.

---

## Demo Behavior

A typical outcome pattern looks like this:

```text
fraud -> FLAG
gray  -> CHECK_WEB / ASK_DEEP_RAG
legit -> PASS
```

The heuristic baseline tends to route many ambiguous cases to a generic `ESCALATE` action, while the DQN chooses more specific next investigative actions.

---

## Repository Structure

```text
src/kie/
  agent.py
  dqn_agent.py
  environment.py
  scoring.py
  profile_delta.py
  web_signals.py
  question_bank.py
  simulator.py

artifacts/
  training_metrics.json
  eval_results.json

tests/
run_demo.py
train_dqn.py
evaluate_agents.py
README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the DQN agent

```bash
python train_dqn.py
```

This generates:

- `artifacts/training_metrics.json`
- `artifacts/dqn_q_net.pth`
- `artifacts/dqn_target_net.pth`

### 3. Run the demo

```bash
python run_demo.py
```

This prints:

- heuristic baseline decisions
- trained DQN decisions
- short natural-language explanations for each action

### 4. Run the evaluation

```bash
python evaluate_agents.py
```

This generates:

- `artifacts/eval_results.json`

### 5. Run tests

```bash
pytest -q
```

---

## Practical Interpretation

This prototype demonstrates that candidate verification can be framed as a **policy problem** rather than only a classification problem.

That distinction matters.

A good verification engine should not only estimate whether someone is suspicious. It should also decide what the **best next investigative move** is:

- validate external evidence
- probe deeper on technical depth
- escalate only when necessary
- pass when evidence is internally consistent

That makes the system more aligned with real hiring workflows and more interpretable than a single black-box fraud score.

---

## Limitations

This is still a controlled prototype.

Current limitations include:

- synthetic candidate data
- simulated web signals
- a simplified and fixed action/state interface
- evaluation in a controlled environment rather than a production hiring pipeline

So the correct interpretation is:

> this project demonstrates **policy improvement in a controlled expertise-verification environment**, not a production-ready fraud-detection platform.

---

## Future Work

Potential next steps include:

- expanding the candidate pool with more adversarial and realistic cases
- integrating richer external validation signals
- improving feature-level and action-level interpretability
- validating state and action dimensions directly from the environment
- testing over broader seeded runs and more realistic screening scenarios

---

## Author

**Raoni Medeiros**  
AI Automation & Systems Engineer
