from src.kie.simulator import run_demo


if __name__ == "__main__":
    print("=== HEURISTIC BASELINE ===")
    run_demo(mode="heuristic")
    print("\n=== TRAINED DQN ===")
    run_demo(mode="dqn")
