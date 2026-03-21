from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.kie.training_env import ExpertEnv
from src.kie.dqn_agent import DQNAgent


OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_training(episodes: int = 200, target_update_interval: int = 10) -> Dict[str, List[float]]:
    env = ExpertEnv()
    state_dim = 10
    action_dim = 11

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

    rewards: List[float] = []
    losses: List[float] = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        episode_losses: List[float] = []

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += float(reward)

        agent.decay_epsilon()
        if ep % target_update_interval == 0:
            agent.update_target()

        rewards.append(total_reward)
        losses.append(float(np.mean(episode_losses)) if episode_losses else 0.0)

        print(f"Episode {ep:03d} | reward={total_reward:.3f} | epsilon={agent.epsilon:.3f}")

    metrics = {"rewards": rewards, "losses": losses}
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
