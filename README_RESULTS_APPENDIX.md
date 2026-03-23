## Results

The latest evaluation compares the heuristic baseline against the trained DQN policy over 300 seeded episodes per agent.

| Metric | Heuristic | DQN |
|---|---:|---:|
| Avg Reward | 1.8692 | **2.5036** |
| Avg Steps to Decision | **3.00** | 4.14 |
| Legit Pass Rate | 0.4712 | **1.0000** |
| Fraud Flag Rate | 1.0000 | **1.0000** |
| Gray Investigate Rate | 1.0000 | **1.0000** |
| False Pass Fraud Rate | 0.0000 | **0.0000** |
| False Flag Legit Rate | 0.0000 | **0.0000** |
| Legit Escalate Rate | 0.5288 | **0.0000** |

### Takeaways

- The DQN maintains perfect fraud flagging while materially improving treatment of legitimate candidates.
- The largest gain is in **legit pass rate**: the learned policy passes legitimate candidates consistently instead of over-escalating them.
- The DQN takes more steps on average, which is acceptable here because the extra investigation produces better final handling of ambiguity and fewer unnecessary escalations.
- Both policies handle gray-zone candidates conservatively, but the learned policy does so with stronger overall reward.

### Interpretation

This is not evidence of production-ready fraud detection. It is evidence that, in a controlled environment, a learned policy can outperform a fixed rules baseline on the operational trade-off that matters here:

- catch strong fraud indicators
- avoid harming legitimate candidates
- investigate before deciding when uncertainty is high
- use escalation as a safe fallback for ambiguous cases

## Example Episode

**Candidate:** Sarah Kim (`gray`)

**Observed pattern:**
- medium genericity
- moderate profile anomaly
- partial operational depth
- mixed external evidence

**Likely policy path:**
`ASK_DEEP_RAG -> CHECK_WEB -> ESCALATE`

**Why this matters:**
The system avoids premature pass/flag decisions and instead treats the candidate as a true ambiguity case. That is the core behavior this prototype is designed to learn.

## Why RL Helps Here

A classifier can score suspicion. It cannot naturally decide whether the next best move is to:

- probe deeper
- verify externally
- stop and pass
- stop and flag
- escalate uncertainty to a reviewer

That sequential decision logic is the reason the project uses reinforcement learning rather than only static classification.