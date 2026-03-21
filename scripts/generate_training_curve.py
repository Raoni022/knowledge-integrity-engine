import json
from pathlib import Path

import matplotlib.pyplot as plt

ARTIFACTS = Path("artifacts")
INPUT_FILE = ARTIFACTS / "training_metrics.json"
OUTPUT_FILE = ARTIFACTS / "training_curve.png"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing metrics file: {INPUT_FILE}")

    data = json.loads(INPUT_FILE.read_text())
    rewards = data.get("rewards", [])
    losses = data.get("losses", [])

    plt.figure()

    if rewards:
        plt.plot(rewards, label="reward")

    if losses:
        plt.plot(losses, label="loss")

    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("DQN Training Curve")
    plt.legend()

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_FILE)

    print(f"Saved training curve to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
