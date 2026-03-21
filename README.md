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
- expose system-level thinking (latency, scaling, edge cases)
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

> "What is the most informative next question to reduce uncertainty?"

---

## System Architecture

### State Representation (10 features)

Each step encodes:

- **Profile Delta** → sudden skill inflation over time  
- **Scar Score** → presence of real-world operational detail  
- **Genericity Score** → LLM-like phrasing patterns  
- **Consistency Score** → alignment between profile and answers  
- **Web Signals Score** → external validation (mocked)  
- **Volatility** → disagreement between signals  
- **Fraud Score** → aggregated suspicion signal  
- **Question Ratio** → how deep the interaction has gone  
- **Last Action Encoding**  
- **Uncertainty** → confidence of current estimate  

---

## Fraud Score (Key Signal)

```md
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
## Interpretation

 * High score → strong indicators of fabricated expertise
 * Low score → consistent with real-world experience

This separation avoids signal cancellation by explicitly distinguishing:

 * suspicion signals
 * credibility signals
   
## Action Space

The system does not immediately classify candidates.

It decides how to investigate them.

## Available Actions
 
- Ask deeper questions:
  - RAG systems
  - Infrastructure
  - Automation workflows
- Investigate signals:
  - Profile consistency
  - External validation (web signals)
    
## Final Decisions

 * FLAG → high-confidence fraud
 * PASS → high-confidence legitimate
 * ESCALATE → ambiguous case
  
## Key Design Decision

Reinforcement Learning is used for:

> Question strategy optimization — not final classification.

Because expertise is revealed through interaction, not single answers.

This design:

 * mimics real technical interviews
 * avoids black-box classification
 * prioritizes interpretability
   
## Current Implementation

This repository includes:

 * Custom Gymnasium environment
 * Heuristic baseline agent (calibrated)
 * Synthetic dataset (fraud / legit / gray cases)
 * Unit tests for scoring components
   
## Demo Result
```
fraud → FLAG
gray  → ESCALATE
legit → PASS
```

## How to Run
```
pip install -r requirements.txt
python run_demo.py
pytest -q
```

## Project Structure
```
src/kie/
  agent.py
  environment.py
  scoring.py
  profile_delta.py
  web_signals.py
  question_bank.py
tests/
run_demo.py
```

## Limitations

 * Uses synthetic data
 * Heuristic agent (no trained RL policy yet)
 * Web signals are simulated

## Future Work

 * DQN-based policy learning
 * Real web signal integration
 * Multi-modal analysis
 * Adaptive questioning strategies

## Author

Raoni Medeiros
AI Automation & Systems Engineer
