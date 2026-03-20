from __future__ import annotations

from typing import Iterable

SCAR_INDICATORS = {
    "chunking",
    "rerank",
    "latency",
    "top-k",
    "threshold",
    "hallucination",
    "lock",
    "timeout",
    "memory",
    "retry",
    "idempotency",
    "race condition",
    "queue",
    "dedup",
    "webhook",
    "payload",
    "rollback",
}

GENERIC_INDICATORS = {
    "best practice",
    "best practices",
    "high quality",
    "operational excellence",
    "scalable",
    "robust",
    "efficient",
    "efficiency",
    "streamline",
    "important",
}


def _count_hits(text: str, indicators: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(1 for token in indicators if token in lowered)



def scar_score(text: str) -> float:
    scar_hits = _count_hits(text, SCAR_INDICATORS)
    generic_hits = _count_hits(text, GENERIC_INDICATORS)
    word_count = max(len(text.split()), 1)
    raw = (scar_hits - 0.5 * generic_hits) / max(word_count / 30.0, 1.0)
    return max(0.0, min(1.0, raw))



def genericity_score(text: str) -> float:
    generic_hits = _count_hits(text, GENERIC_INDICATORS)
    word_count = max(len(text.split()), 1)
    raw = generic_hits / max(word_count / 25.0, 1.0)
    return max(0.0, min(1.0, raw))



def consistency_score(profile_tools: list[str], text: str, years_experience: int) -> float:
    lowered = text.lower()
    tool_mentions = sum(1 for tool in profile_tools if tool.lower() in lowered)
    operational_context_hits = _count_hits(text, SCAR_INDICATORS)
    exp_factor = min(max(years_experience / 5.0, 0.2), 1.0)
    raw = (0.6 * (tool_mentions / max(len(profile_tools), 1)) + 0.4 * min(operational_context_hits / 4.0, 1.0)) * exp_factor
    return max(0.0, min(1.0, raw))



def info_gain(previous_score: float, current_score: float) -> float:
    return abs(current_score - previous_score)
