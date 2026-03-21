# Multimodal Fraud Detection (Live Interview Scenario)

## Signals

### Visual
- Eye tracking (reading vs recalling)
- Micro-delays before speaking
- Gaze patterns (screen vs camera)

### Audio
- Response latency
- Filler patterns ("uh", "let me think")
- Confidence variance over time

### Screen
- Tab switching frequency
- Copy/paste behavior
- Prompt usage patterns

---

## RL Extension

### State Space (extended)
- visual_latency_score
- audio_confidence_score
- tab_switch_frequency
- existing 10 features from base system

### Action Space
- ask follow-up question
- increase pressure (deeper technical question)
- request real-world example
- switch topic
- escalate to human

### Reward Design

- +1 → exposes inconsistency under pressure
- -2 → false accusation (real expert flagged)
- -1.5 → missed fraud
- +0.2 → correct escalation

---

## Real-Time Decision Flow

1. Observe multimodal signals
2. Update state vector
3. Select next action via policy
4. Update belief about candidate authenticity
5. Decide: continue / escalate / finalize

---

## Key Insight

Real experts degrade differently under pressure than LLM-assisted candidates.

The signal is not correctness — it is response dynamics.
