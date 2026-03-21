import json
from collections import Counter
from pathlib import Path

import torch

from src.kie.agent import HeuristicAgent
from src.kie.dqn_agent import DQNAgent
from src.kie.environment import KnowledgeIntegrityEnv


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def build_agent(mode: str):
    if mode == "heuristic":
        return HeuristicAgent()

    if mode == "dqn":
        agent = DQNAgent(state_dim=10, action_dim=11)
        model_path = ARTIFACTS_DIR / "dqn_q_net.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        agent.q_net.load_state_dict(torch.load(model_path, map_location="cpu"))
        agent.q_net.eval()
        return agent

    raise ValueError(f"Unsupported mode: {mode}")


def choose_action(agent, mode: str, state):
    if mode == "dqn":
        return agent.act(state, greedy=True)
    return agent.act(state)


def evaluate(mode: str, episodes: int = 200):
    env = KnowledgeIntegrityEnv(max_steps=5)
    agent = build_agent(mode)

    outcomes = []
    reward_sum = 0.0
    summary = Counter()

    for episode in range(episodes):
        state, meta = env.reset(seed=episode)
        done = False
        total_reward = 0.0
        final_info = None

        while not done:
            action = choose_action(agent, mode, state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            state = next_state
            final_info = info

        label = meta["label"]
        decision = final_info["action_name"]
        avg_score = float(final_info["avg_score"])

        outcomes.append({
            "candidate": meta["candidate"],
            "label": label,
            "decision": decision,
            "avg_score": avg_score,
            "reward": round(total_reward, 4),
        })

        reward_sum += total_reward
        summary[(label, decision)] += 1

    legit_total = sum(1 for x in outcomes if x["label"] == "legit")
    fraud_total = sum(1 for x in outcomes if x["label"] == "fraud")
    gray_total = sum(1 for x in outcomes if x["label"] == "gray")

    legit_pass = sum(1 for x in outcomes if x["label"] == "legit" and x["decision"] == "PASS")
    fraud_flag = sum(1 for x in outcomes if x["label"] == "fraud" and x["decision"] == "FLAG")
    gray_investigate = sum(
        1 for x in outcomes
        if x["label"] == "gray" and x["decision"] in {"ESCALATE", "CHECK_WEB", "ASK_DEEP_RAG"}
    )

    results = {
        "mode": mode,
        "episodes": episodes,
        "avg_reward": round(reward_sum / episodes, 4),
        "legit_pass_rate": round(legit_pass / legit_total, 4) if legit_total else None,
        "fraud_flag_rate": round(fraud_flag / fraud_total, 4) if fraud_total else None,
        "gray_investigate_rate": round(gray_investigate / gray_total, 4) if gray_total else None,
        "decision_counts": {
            f"{label}->{decision}": count
            for (label, decision), count in sorted(summary.items())
        },
        "sample_outcomes": outcomes[:20],
    }

    return results


def main():
    heuristic = evaluate("heuristic", episodes=300)
    dqn = evaluate("dqn", episodes=300)

    final = {
        "heuristic": heuristic,
        "dqn": dqn,
    }

    out_path = ARTIFACTS_DIR / "eval_results.json"
    out_path.write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
