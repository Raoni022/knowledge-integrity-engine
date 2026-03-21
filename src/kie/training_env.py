from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .environment import KnowledgeIntegrityEnv


class ExpertEnv:
    """Compatibility wrapper for training loops.

    The base environment follows Gymnasium's newer API:
    - reset() -> (state, info)
    - step() -> (state, reward, terminated, truncated, info)

    This wrapper exposes the simpler interface expected by the DQN training loop:
    - reset() -> state
    - step() -> (next_state, reward, done, info)
    """

    def __init__(self, max_steps: int = 5):
        self.env = KnowledgeIntegrityEnv(max_steps=max_steps)

    def reset(self, seed: int | None = None) -> np.ndarray:
        state, _info = self.env.reset(seed=seed)
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        return next_state, float(reward), done, info
