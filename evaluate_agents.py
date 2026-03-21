import json
from collections import Counter
from pathlib import Path

import torch

from src.kie.agent import HeuristicAgent
from src.kie.dqn_agent import DQNAgent
from src.kie.environment import KnowledgeIntegrityEnv


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

BASE_SEED = 42


def build_agent(mode: str, state_dim: int, action_dim: int):
    if mode == "heuristic":
        return HeuristicAgent()

    if mode == "dqn":
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
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
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = build_agent(mode, state_dim, action_dim)

    outcomes = []
    reward_sum = 0.0
    summary = Counter()

    for episode in range(episodes):
        state, meta = env.reset(seed=BASE_SEED + episode)
        done = False
        total_reward = 0.0
        final_info = None
        steps = 0

        while not done:
            action = choose_action(agent, mode, state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            state = next_state
            final_info = info
            steps += 1

        if final_info is None:
            raise RuntimeError("Episode ended without final_info.")

        label = meta["label"]
        decision = final_info["action_name"]
        avg_score = float(final_info["avg_score"])

        outcomes.append({
            "candidate": meta["candidate"],
            "label": label,
            "decision": decision,
            "avg_score": avg_score,
            "reward": round(total_reward, 4),
            "steps": steps,
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

    false_pass_fraud = sum(
        1 for x in outcomes
        if x["label"] == "fraud" and x["decision"] == "PASS"
    )

    false_flag_legit = sum(
        1 for x in outcomes
        if x["label"] == "legit" and x["decision"] == "FLAG"
    )

    legit_escalate = sum(
        1 for x in outcomes
        if x["label"] == "legit" and x["decision"] == "ESCALATE"
    )

    avg_steps = round(sum(x["steps"] for x in outcomes) / len(outcomes), 4)

    results = {
        "mode": mode,
        "episodes": episodes,
        "base_seed": BASE_SEED,
        "avg_reward": round(reward_sum / episodes, 4),
        "avg_steps_to_decision": avg_steps,
        "legit_pass_rate": round(legit_pass / legit_total, 4) if legit_total else None,
        "fraud_flag_rate": round(fraud_flag / fraud_total, 4) if fraud_total else None,
        "gray_investigate_rate": round(gray_investigate / gray_total, 4) if gray_total else None,
        "false_pass_fraud_rate": round(false_pass_fraud / fraud_total, 4) if fraud_total else None,
        "false_flag_legit_rate": round(false_flag_legit / legit_total, 4) if legit_total else None,
        "legit_escalate_rate": round(legit_escalate / legit_total, 4) if legit_total else None,
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
