# Knowledge Integrity & Expert Verification Engine

A reinforcement learning proof of concept for detecting **expertise inflation** in technical candidate screening.

## Overview

Traditional hiring often assumes that correct answers imply real expertise.

That assumption is weaker now.

With modern LLMs, candidates can generate polished answers and credible-looking profiles without having operated real systems in production. The failure mode is often not factual incorrectness, but **surface-level correctness without operational depth**.

This project explores a different framing for that problem:

> instead of asking only whether a candidate looks suspicious, ask what the **best next investigative action** is to reduce uncertainty.

The result is a simplified environment where an agent learns when to:

- pass a candidate
- flag a candidate
- escalate to manual review
- investigate further through deeper technical probing or web validation

---

## Core Thesis

Correctness alone is no longer a reliable proxy for experience.

In practice, stronger candidates tend to expose:

- constraints
- trade-offs
- failure modes
- debugging paths
- operational details

more consistently under deeper questioning.

By contrast, LLM-assisted or overstated expertise often appears as:

- polished but generic answers
- weak linkage between claims and examples
- inconsistent depth across related topics
- poor external evidence relative to claimed experience

This project models that process as a **sequential decision problem**.

---

## Why Reinforcement Learning?

A static classifier can score a candidate as suspicious or not suspicious.

But real interview workflows are sequential. A strong interviewer does not only classify — they decide what to do next:

- ask a deeper question
- verify externally
- escalate ambiguity
- stop early when confidence is high

That makes RL a reasonable fit for the proof of concept.

The goal here is not to claim that RL is the only valid approach.  
The goal is to show that **candidate verification can be framed as a policy problem**, not only a binary classification problem.

---

## Environment Design

The repository includes a custom Gymnasium environment:

- `KnowledgeIntegrityEnv`

Each episode simulates the evaluation of one synthetic candidate.

### Candidate Labels

The synthetic environment uses three classes:

- `legit`
- `fraud`
- `gray`

These labels act as the controlled ground truth for the prototype.

### Observation Space

The environment encodes each step as a fixed-size feature vector with 10 values:

1. **Profile Delta** — suspicious career inflation over time
2. **Scar Score** — evidence of operational detail
3. **Genericity Score** — broad, polished, low-specificity phrasing
4. **Consistency Score** — alignment between profile claims and answers
5. **Web Signals Score** — contradiction between claims and public evidence
6. **Volatility** — disagreement between signals
7. **Fraud Score** — aggregated suspicion score
8. **Question Ratio** — how far the interaction has progressed
9. **Last Action Encoding** — normalized representation of prior action
10. **Uncertainty** — confidence-adjusted ambiguity estimate

### Action Space

The agent can choose among:

- `ASK_BROAD_RAG`
- `ASK_DEEP_RAG`
- `ASK_BROAD_INFRA`
- `ASK_DEEP_INFRA`
- `ASK_BROAD_AUTOMATION`
- `ASK_DEEP_AUTOMATION`
- `CHECK_PROFILE`
- `CHECK_WEB`
- `PASS`
- `FLAG`
- `ESCALATE`

The core idea is that the system does not immediately jump to a final label.  
It can investigate first.

---

## Reward Design

The reward function is designed around three goals:

1. correctly identify strong fraud indicators
2. avoid unnecessary false positives on legitimate candidates
3. encourage useful intermediate investigation before final decisions

### Intuition

- Investigative actions such as `ASK_*`, `CHECK_PROFILE`, and `CHECK_WEB` receive positive reward when uncertainty is high
- `FLAG` is strongly rewarded for fraud and penalized for legit candidates
- `PASS` is rewarded for legit candidates and penalized for fraud
- `ESCALATE` is treated as a useful fallback for ambiguous gray-zone profiles
- long investigations are mildly penalized to avoid wasteful policies

This makes the policy trade off:

- decision quality
- uncertainty reduction
- manual review burden
- investigation cost

---

## Feature Engineering

The prototype focuses on a small set of interpretable signals rather than a large set of noisy features.

### Included Signals

**Profile Delta**
- timeline anomalies
- title inflation
- compressed experience progression

**Operational "Scar" Signals**
- mentions of concrete implementation details such as:
  - chunking
  - reranking
  - latency
  - retries
  - race conditions
  - idempotency
  - queues
  - rollback

**Genericity Signals**
- polished but low-information language such as:
  - "best practices"
  - "robust"
  - "scalable"
  - "high quality"
  - "operational excellence"

**Consistency Signals**
- overlap between claimed tools and response content
- alignment between claimed experience and operational depth

**Web Signals**
- mocked proxies for:
  - GitHub activity
  - public presence
  - temporal coherence of external evidence

### Design Choice

This first version intentionally uses **interpretable and partially hand-crafted features**.

That is a limitation, but also a deliberate trade-off:
- easier to debug
- easier to explain
- better for validating the policy framing early

A more advanced version would move toward semantic consistency models and richer evidence graphs.

---

## Fraud Score

The environment builds an aggregate suspicion score:

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

- **Higher score** -> stronger fraud indicators
- **Lower score** -> stronger evidence of legitimate expertise

This is not intended as a production fraud score.  
It is an interpretable internal signal for the RL environment.

---

## Agent Design

The repository contains two agents:

### 1. Heuristic Baseline
A rules-based policy that:
- investigates early when evidence is mixed
- flags strong fraud-like patterns
- passes strong legit-like patterns
- escalates remaining ambiguity

### 2. Trained DQN Agent
A Deep Q-Network trained to optimize reward over repeated episodes.

The DQN is not presented as a magical fraud detector.  
Its role is to learn whether a more adaptive policy can outperform a fixed heuristic on the same environment.

---

## Evaluation

The evaluation compares the heuristic baseline and the trained DQN over repeated seeded episodes.

### Reported Metrics

The evaluation script reports:

- average reward
- average steps to decision
- legit pass rate
- fraud flag rate
- gray investigate rate
- false pass rate for fraud
- false flag rate for legit
- legit escalate rate
- decision counts by label

This is more useful than reward alone because the real trade-offs are:

- catching fraud
- not harming legitimate candidates
- not overloading manual review

---

## Current Repository Structure

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
  data/
    synthetic_candidates.json

artifacts/
  training_metrics.json
  eval_results.json
  dqn_q_net.pth
  dqn_target_net.pth

run_demo.py
train_dqn.py
evaluate_agents.py
requirements.txt
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
- `artifacts/training_curve.png` if matplotlib is available

### 3. Run the side-by-side demo

```bash
python run_demo.py
```

This shows:
- heuristic decisions
- trained DQN decisions
- short natural-language explanations

### 4. Run the full evaluation

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

## Reproducibility

This repository uses:

- a single canonical environment: `KnowledgeIntegrityEnv`
- fixed seeds in training and evaluation
- synthetic candidate data under `src/kie/data`

To reproduce the main artifacts:

```bash
python train_dqn.py
python evaluate_agents.py
```

---

## Demo Interpretation

A common behavior pattern is:

```text
fraud -> FLAG
gray  -> CHECK_WEB / ASK_DEEP_RAG / ESCALATE
legit -> PASS
```

The baseline tends to use more generic escalation.

The trained policy can learn more targeted intermediate actions before reaching a final decision.

That is the main point of the prototype.

---

## Ground Truth

For this proof of concept, ground truth is synthetic and controlled.

That is acceptable for a coding exercise, but obviously weaker than real deployment data.

In a more realistic system, ground truth would likely need to combine:

- expert adjudication
- disagreement-aware labeling
- paired interview outcomes
- delayed validation from technical assessments
- weak supervision from external evidence

This repository does **not** solve that problem fully.  
It only demonstrates the policy-learning framing in a controlled setup.

---

## Limitations

This project is intentionally simplified.

Current limitations include:

- synthetic candidate data
- hand-crafted feature logic
- mocked web evidence
- small state space
- no true semantic reasoning model
- no train/eval split across richer adversarial distributions
- no live interview multimodal signals yet

So the correct interpretation is:

> this is a controlled RL proof of concept for adaptive expertise verification, not a production-ready fraud detection system

---

## Bonus Challenge: Multi-Modal Live Extension

A realistic live version could extend the state space with multimodal cues from:

### Visual
- eye movement patterns
- repeated off-screen glances
- reading behavior
- abnormal hesitation before technical answers

### Audio
- answer latency
- pacing shifts
- prosody disruption
- abrupt confidence collapse under deeper probing

### Screen
- tab switching
- clipboard behavior
- window focus changes
- suspicious timing correlations with answer quality

### RL Framing in a Live Setting

In that version, the policy could decide whether to:

- continue current topic
- deepen the question
- switch topic
- request implementation detail
- trigger silent risk accumulation
- escalate to live reviewer
- stop early when evidence is strong

That would turn the agent into a real-time interview strategy layer rather than only an offline evaluator.

---

## Why This Project Is Useful

The main value of this project is not high benchmark performance.

The main value is the modeling choice:

- treat expertise verification as a sequential decision process
- optimize investigation strategy, not only final classification
- keep the signal set small and interpretable
- surface trade-offs between fraud detection, false positives, and review cost

That is the core thesis.

---

## Future Work

Potential next steps:

- add richer synthetic and adversarial candidate profiles
- improve train/eval separation
- replace lexical features with semantic consistency features
- add cross-answer consistency scoring
- calibrate reward weights more systematically
- integrate real external evidence sources
- simulate live multimodal interview signals
- add model interpretability outputs per action

---

## Results

The latest evaluation compares the heuristic baseline against the trained DQN policy.

| Metric | Heuristic | DQN |
|---|---:|---:|
| Avg Reward | 1.869 | **2.503** |
| Avg Steps | **3.0** | 4.14 |
| Legit Pass Rate | 0.47 | **1.00** |
| Fraud Flag Rate | 1.00 | **1.00** |
| False Pass Fraud | 0.00 | **0.00** |
| False Flag Legit | 0.00 | **0.00** |

---

### Interpretation

- The DQN significantly improves handling of legitimate candidates (100% pass rate)
- It maintains perfect fraud detection
- It uses slightly more steps, indicating deeper investigation
- It reduces unnecessary escalation for legit candidates

This suggests the learned policy improves **decision quality under uncertainty**, not just reward.

---

## Author

**Raoni Medeiros**  
AI Automation & Systems Engineer
