from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.kie.environment import KnowledgeIntegrityEnv
from src.kie.dqn_agent import DQNAgent


OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42


def run_training(episodes: int = 200, target_update_interval: int = 10) -> Dict[str, List[float]]:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = KnowledgeIntegrityEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

    rewards: List[float] = []
    losses: List[float] = []

    for ep in range(episodes):
        state, _ = env.reset(seed=SEED + ep)
        done = False
        total_reward = 0.0
        episode_losses: List[float] = []

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += float(reward)

        agent.decay_epsilon()

        if (ep + 1) % target_update_interval == 0:
            agent.update_target()

        rewards.append(total_reward)
        losses.append(float(np.mean(episode_losses)) if episode_losses else 0.0)

        print(f"Episode {ep:03d} | reward={total_reward:.3f} | epsilon={agent.epsilon:.3f}")

    torch.save(agent.q_net.state_dict(), OUTPUT_DIR / "dqn_q_net.pth")
    torch.save(agent.target_net.state_dict(), OUTPUT_DIR / "dqn_target_net.pth")

    metrics = {
        "episodes": episodes,
        "target_update_interval": target_update_interval,
        "seed": SEED,
        "final_epsilon": agent.epsilon,
        "rewards": rewards,
        "losses": losses,
    }
    (OUTPUT_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2))

    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure()
        plt.plot(rewards)
        plt.title("Training Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "training_curve.png")
        plt.close()
    except Exception:
        pass

    return metrics


if __name__ == "__main__":
    run_training(episodes=200)
