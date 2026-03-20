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
    """A lightweight baseline agent.

    This is not a full neural DQN yet. It provides a defensible baseline policy so the
    repository is runnable immediately, while keeping the RL environment ready for a
    PyTorch-backed DQN extension.
    """

    def act(self, state: np.ndarray) -> int:
        profile_delta, scar, genericity, consistency, web_score, _, avg, q_ratio, _, uncertainty = state

        if q_ratio < 0.4:
            if profile_delta > 0.65:
                return 6  # CHECK_PROFILE
            if web_score > 0.6:
                return 7  # CHECK_WEB
            if scar < 0.25:
                return 1  # ASK_DEEP_RAG
            return 3  # ASK_DEEP_INFRA

        if avg > 0.72 and uncertainty < 0.45:
            return 9  # FLAG
        if avg < 0.38 and consistency > 0.45 and genericity < 0.45:
            return 8  # PASS
        return 10  # ESCALATE
