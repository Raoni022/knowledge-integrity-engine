from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .profile_delta import profile_delta_score
from .question_bank import ACTIONS
from .scoring import consistency_score, genericity_score, info_gain, scar_score
from .web_signals import web_signals_score


DATA_PATH = Path(__file__).parent / "data" / "synthetic_candidates.json"


class KnowledgeIntegrityEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps: int = 5):
        super().__init__()
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            self.candidates = json.load(f)

        self.max_steps = max_steps
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)

        self.current_candidate: dict[str, Any] | None = None
        self.steps = 0
        self.prev_avg = 0.0
        self.last_action_norm = 0.0

    def _pick_candidate(self, seed: int | None = None) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        idx = int(rng.integers(0, len(self.candidates)))
        return self.candidates[idx]

    def _topic_text(self) -> str:
        responses = self.current_candidate["responses"]
        if self.steps % 3 == 0:
            return responses["rag"]
        if self.steps % 3 == 1:
            return responses["infra"]
        return responses["automation"]

    def _build_state(self) -> np.ndarray:
        profile = self.current_candidate["profile_claims"]
        text = self._topic_text()

        p_score = profile_delta_score(profile)
        s_score = scar_score(text)
        g_score = genericity_score(text)
        c_score = consistency_score(profile["tools"], text, profile["years_experience"])
        w_score = web_signals_score(profile["web_signals"])
        avg_score = (p_score + (1 - g_score) + c_score + (1 - w_score) + s_score) / 5.0
        volatility = abs(s_score - c_score)
        uncertainty = 1.0 - abs(avg_score - 0.5) * 2.0
        question_ratio = min(self.steps / self.max_steps, 1.0)
        info = info_gain(self.prev_avg, avg_score)

        state = np.array([
            p_score,
            s_score,
            g_score,
            c_score,
            w_score,
            volatility,
            avg_score,
            question_ratio,
            self.last_action_norm,
            uncertainty * (0.7 + 0.3 * info),
        ], dtype=np.float32)
        self.prev_avg = avg_score
        return state

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_candidate = self._pick_candidate(seed)
        self.steps = 0
        self.prev_avg = 0.0
        self.last_action_norm = 0.0
        state = self._build_state()
        return state, {"candidate": self.current_candidate["name"], "label": self.current_candidate["label"]}

    def step(self, action: int):
        action_name = ACTIONS[action]
        self.steps += 1
        self.last_action_norm = action / max(len(ACTIONS) - 1, 1)
        state = self._build_state()
        avg_score = float(state[6])
        label = self.current_candidate["label"]

        reward = 0.0
        terminated = False

        if action_name.startswith("ASK") or action_name in {"CHECK_PROFILE", "CHECK_WEB"}:
            reward += 0.15 + 0.35 * float(state[9])
            reward -= 0.05 * float(state[2])
        elif action_name == "FLAG":
            terminated = True
            if label == "fraud":
                reward += 2.0
            elif label == "gray":
                reward -= 0.4
            else:
                reward -= 2.0
        elif action_name == "PASS":
            terminated = True
            if label == "legit":
                reward += 1.8
            elif label == "gray":
                reward -= 0.3
            else:
                reward -= 1.5
        elif action_name == "ESCALATE":
            terminated = True
            if label == "gray":
                reward += 1.5
            else:
                reward -= 0.2

        if not terminated and self.steps >= self.max_steps:
            terminated = True
            if label == "gray":
                reward += 0.8
            else:
                reward -= 0.5 * abs(avg_score - (0.75 if label == "fraud" else 0.25))

        info = {
            "candidate": self.current_candidate["name"],
            "label": label,
            "action_name": action_name,
            "avg_score": avg_score,
        }
        return state, reward, terminated, False, info
