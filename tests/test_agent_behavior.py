import numpy as np

from src.kie.agent import HeuristicAgent
from src.kie.question_bank import ACTIONS


def test_agent_returns_valid_action_index():
    agent = HeuristicAgent()

    state = np.zeros(10, dtype=np.float32)
    action = agent.act(state)

    assert isinstance(action, int)
    assert 0 <= action < len(ACTIONS)


def test_agent_does_not_pass_on_high_fraud_signal():
    agent = HeuristicAgent()

    state = np.array([
        0.9,  # profile_delta
        0.1,  # scar
        0.8,  # genericity
        0.2,  # consistency
        0.7,  # web_score
        0.5,  # volatility
        0.9,  # fraud_score
        0.8,  # q_ratio
        0.0,
        0.2   # uncertainty
    ], dtype=np.float32)

    action = agent.act(state)

    # PASS = index 8
    assert action != 8


def test_agent_does_not_flag_clear_legit_signal():
    agent = HeuristicAgent()

    state = np.array([
        0.1,
        0.6,
        0.2,
        0.8,
        0.1,
        0.1,
        0.1,
        0.8,
        0.0,
        0.2
    ], dtype=np.float32)

    action = agent.act(state)

    # FLAG = index 9
    assert action != 9