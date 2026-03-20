from __future__ import annotations


def profile_delta_score(profile_claims: dict) -> float:
    timeline = float(profile_claims.get("timeline_anomaly", 0.0))
    title = float(profile_claims.get("title_inflation", 0.0))
    years = float(profile_claims.get("years_experience", 0.0))

    compression_penalty = 0.15 if years <= 2 and (timeline > 0.5 or title > 0.5) else 0.0
    score = 0.55 * timeline + 0.35 * title + compression_penalty
    return max(0.0, min(1.0, score))
