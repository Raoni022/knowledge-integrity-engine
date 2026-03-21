import numpy as np

from src.kie.environment import KnowledgeIntegrityEnv
from src.kie.question_bank import ACTIONS


def test_environment_reset_returns_valid_state_and_info():
    env = KnowledgeIntegrityEnv()
    state, info = env.reset(seed=42)

    assert isinstance(state, np.ndarray)
    assert state.shape == (10,)
    assert state.dtype == np.float32

    assert isinstance(info, dict)
    assert "candidate" in info
    assert "label" in info
    assert info["label"] in {"fraud", "legit", "gray"}


def test_environment_step_returns_gymnasium_tuple():
    env = KnowledgeIntegrityEnv()
    _state, info = env.reset(seed=7)

    next_state, reward, terminated, truncated, step_info = env.step(0)

    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (10,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    assert isinstance(step_info, dict)
    assert step_info["candidate"] == info["candidate"]
    assert step_info["label"] == info["label"]
    assert step_info["action_name"] in ACTIONS
    assert isinstance(step_info["avg_score"], float)


def test_environment_spaces_are_consistent():
    env = KnowledgeIntegrityEnv()

    assert env.action_space.n == len(ACTIONS)
    assert env.observation_space.shape == (10,)


def test_environment_terminates_within_max_steps():
    env = KnowledgeIntegrityEnv(max_steps=3)
    env.reset(seed=1)

    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(0)
        steps += 1
        assert steps <= 3

    assert terminated or truncated
