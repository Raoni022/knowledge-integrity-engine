from __future__ import annotations

from collections import Counter

from rich.console import Console
from rich.table import Table

from .agent import HeuristicAgent
from .environment import KnowledgeIntegrityEnv
from .question_bank import ACTIONS


console = Console()



def run_demo(episodes: int = 12) -> None:
    env = KnowledgeIntegrityEnv(max_steps=5)
    agent = HeuristicAgent()
    outcomes = []

    for episode in range(episodes):
        state, meta = env.reset(seed=episode)
        done = False
        final_info = None
        total_reward = 0.0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, info = env.step(action)
            total_reward += reward
            state = next_state
            final_info = info

        outcomes.append({
            "candidate": meta["candidate"],
            "label": meta["label"],
            "decision": final_info["action_name"],
            "avg_score": round(final_info["avg_score"], 3),
            "reward": round(total_reward, 3),
        })

    table = Table(title="Knowledge Integrity Engine Demo")
    table.add_column("Candidate")
    table.add_column("Ground Truth")
    table.add_column("Decision")
    table.add_column("Avg Score")
    table.add_column("Reward")

    for row in outcomes:
        table.add_row(row["candidate"], row["label"], row["decision"], str(row["avg_score"]), str(row["reward"]))

    summary = Counter((row["label"], row["decision"]) for row in outcomes)

    console.print(table)
    console.print("\nDecision summary:")
    for key, count in sorted(summary.items()):
        console.print(f"- {key[0]} -> {key[1]}: {count}")

    console.print("\nNote: this demo uses a heuristic baseline agent over an RL-ready environment.")
