# Strategy Memo — Knowledge Integrity & Expert Verification Engine

My thesis is simple: the strongest signal of expertise fraud is not whether a candidate sounds correct on the surface. It is whether their depth increases under pressure. Real practitioners usually reveal operational scars — failure modes, migrations, latency trade-offs, threshold tuning, rollback decisions, weird edge cases, and tool-specific constraints. LLM-assisted candidates often remain polished, but generic.

Because of that, I would not start with dozens of weak features. I would prioritize three high-yield signal groups.

First, **historical profile deltas**. Natural career growth is usually messy but coherent. Suspicious cases often show compressed title jumps, sudden skill explosions, and retrospective profile inflation that overfits market demand. A profile that goes from generalist to “senior AI/ML expert” in a short window, without supporting progression, is a useful prior.

Second, **screening-answer depth signals**. The strongest red flags are generic phrasing, lack of implementation scars, and weak coupling between claimed tools and concrete usage detail. Someone who claims deep pgvector or LangChain experience should be able to talk about chunking failures, retrieval thresholds, index rebuild pain, metadata filters, or latency trade-offs. Broadly correct answers are not enough.

Third, **web signals**. In a production system I would use public evidence carefully, not as a hard requirement, but as coherence support. GitHub activity, public technical writing, talks, issue threads, and temporal plausibility all matter. The key question is not “is this person famous online?” but “does external evidence contradict or support the timeline and tools they claim?”

I use reinforcement learning for **question strategy**, not for the final verdict. The environment state includes normalized signals such as profile delta score, scar score, genericity score, consistency score, web-signals contradiction score, volatility between signals, uncertainty, and question count. The agent’s actions are to ask broad or deep questions by topic, check profile evidence, check web evidence, or end the episode with PASS, FLAG, or ESCALATE.

The reward is asymmetric by design. False positives are more expensive than false negatives, so incorrectly flagging a legitimate expert receives the highest penalty. Informative probes receive positive reward. Premature decisions are penalized. Escalation is rewarded for gray-zone candidates where the honest answer is uncertainty.

Ground truth should come from three layers: synthetic candidates for controlled experimentation, weak supervision from known profile/answer mismatches, and human-reviewed adjudication for the ambiguous middle. The point of the model is not automated punishment. It is to make expert verification more rigorous, more explainable, and less vulnerable to polished but shallow answers.
