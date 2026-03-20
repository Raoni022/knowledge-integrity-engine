from __future__ import annotations


def web_signals_score(web_signals: dict) -> float:
    github_activity = float(web_signals.get("github_activity", 0.0))
    public_presence = float(web_signals.get("public_presence", 0.0))
    temporal_coherence = float(web_signals.get("temporal_coherence", 0.0))

    positive = 0.45 * github_activity + 0.20 * public_presence + 0.35 * temporal_coherence
    contradiction_score = 1.0 - positive
    return max(0.0, min(1.0, contradiction_score))
