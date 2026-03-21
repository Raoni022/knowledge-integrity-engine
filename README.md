# knowledge-integrity-engine

# Knowledge Integrity & Expert Verification Engine

## Overview

Traditional hiring assumes that correct answers imply real expertise.

That assumption no longer holds.

With LLMs, candidates can generate convincing answers and polished profiles without having actually operated systems in production. The failure mode is subtle: not incorrect answers, but lack of depth disguised as competence.

This project reframes the problem:

> The goal is not to detect wrong answers — it is to detect absence of lived experience.

---

## Core Thesis

The distinction between a real expert and an LLM-assisted candidate is not correctness.

It is the depth gradient of their reasoning.

Real experts:
- reference constraints, trade-offs, and failure modes
- show "scars" from real systems
- maintain consistency across contexts

LLM-assisted candidates:
- produce structurally correct but generic answers
- lack operational detail
- collapse under deeper probing

---

## System Design

This problem is modeled as a sequential decision process.

### State Representation

- Profile Delta
- Scar Score
- Genericity Score
- Consistency Score
- Web Signals Score
- Volatility
- Fraud Score
- Question Ratio
- Last Action
- Uncertainty

---

## Fraud Score

```python
fraud_score = (
    0.30 * profile_delta
    + 0.25 * genericity
    + 0.20 * web_score
    + 0.10 * volatility
    - 0.10 * scar
    - 0.15 * consistency
)

High → suspicious
Low → credible

Action Space

The system does not immediately classify candidates.

It decides how to investigate them.

Ask deeper questions (RAG, infra, automation)
Check profile consistency
Check external signals

Final decisions:

FLAG → fraud
PASS → legitimate
ESCALATE → uncertain
Key Design Decision

Reinforcement Learning is used for:

Question strategy, not final classification.

This avoids black-box classification and models how real experts are evaluated.

Demo Result
fraud → FLAG
gray  → ESCALATE
legit → PASS
How to Run
pip install -r requirements.txt
python run_demo.py
pytest -q
Limitations
Synthetic dataset
Heuristic agent (no trained RL yet)
Web signals are mocked

These are intentional trade-offs to prioritize interpretability.

Future Work
DQN-based agent
Real web signal integration (GitHub, StackOverflow)
Multi-modal signals (video/audio)
Adaptive questioning strategies
Author

Raoni Medeiros
AI Automation & Systems Engineer
