from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class HeuristicAgent:
    """Stronger baseline policy over a fraud-oriented score."""

    def act(self, state: np.ndarray) -> int:
        profile_delta, scar, genericity, consistency, web_score, volatility, fraud_score, q_ratio, _, uncertainty = state

        if q_ratio < 0.4:
            if profile_delta > 0.65:
                return 6  # CHECK_PROFILE
            if web_score > 0.60:
                return 7  # CHECK_WEB
            if genericity > 0.55 and scar < 0.25:
                return 1  # ASK_DEEP_RAG
            if consistency < 0.30:
                return 5  # ASK_DEEP_AUTOMATION
            return 3  # ASK_DEEP_INFRA

        if fraud_score > 0.60 and uncertainty < 0.60:
            return 9  # FLAG

        if fraud_score < 0.32 and consistency > 0.35 and scar > 0.20 and genericity < 0.55:
            return 8  # PASS

        return 10  # ESCALATE
