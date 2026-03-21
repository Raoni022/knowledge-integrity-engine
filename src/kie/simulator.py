from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from .agent import HeuristicAgent
from .dqn_agent import DQNAgent
from .environment import KnowledgeIntegrityEnv


console = Console()


def explain_decision(decision: str, avg_score: float) -> str:
    if decision == "PASS":
        return f"Low aggregate risk score ({avg_score:.3f}); profile appears internally consistent."
    if decision == "FLAG":
        return f"High risk score ({avg_score:.3f}); profile shows strong fraud indicators."
    if decision == "CHECK_WEB":
        return f"Medium-risk pattern ({avg_score:.3f}); external verification is the best next step."
    if decision == "ASK_DEEP_RAG":
        return f"Gray-zone pattern ({avg_score:.3f}); deeper evidence retrieval is needed."
    if decision == "ESCALATE":
        return f"Ambiguous pattern ({avg_score:.3f}); escalate for manual review."
    return f"Decision taken with avg_score={avg_score:.3f}."


def run_demo(episodes: int = 12, mode: str = "heuristic") -> None:
    env = KnowledgeIntegrityEnv(max_steps=5)

    if mode == "dqn":
        agent = DQNAgent(state_dim=10, action_dim=11)
        model_path = Path("artifacts") / "dqn_q_net.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        agent.q_net.load_state_dict(torch.load(model_path, map_location="cpu"))
        agent.q_net.eval()
        note = "this demo uses a trained DQN agent over the RL-ready environment."
    else:
        agent = HeuristicAgent()
        note = "this demo uses a heuristic baseline agent over the RL-ready environment."

    outcomes = []

    for episode in range(episodes):
        state, meta = env.reset(seed=episode)
        done = False
        final_info = None
        total_reward = 0.0

        while not done:
            if mode == "dqn":
                action = agent.act(state, greedy=True)
            else:
                action = agent.act(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            state = next_state
            final_info = info

        explanation = explain_decision(final_info["action_name"], float(final_info["avg_score"]))

        outcomes.append({
            "candidate": meta["candidate"],
            "label": meta["label"],
            "decision": final_info["action_name"],
            "avg_score": round(final_info["avg_score"], 3),
            "reward": round(total_reward, 3),
            "explanation": explanation,
        })

    table = Table(title=f"Knowledge Integrity Engine Demo ({mode})")
    table.add_column("Candidate")
    table.add_column("Ground Truth")
    table.add_column("Decision")
    table.add_column("Avg Score")
    table.add_column("Reward")
    table.add_column("Why")

    for row in outcomes:
        table.add_row(
            row["candidate"],
            row["label"],
            row["decision"],
            str(row["avg_score"]),
            str(row["reward"]),
            row["explanation"],
        )

    summary = Counter((row["label"], row["decision"]) for row in outcomes)

    console.print(table)
    console.print("\nDecision summary:")
    for key, count in sorted(summary.items()):
        console.print(f"- {key[0]} -> {key[1]}: {count}")

    console.print(f"\nNote: {note}")
