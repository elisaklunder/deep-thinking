import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import yaml  # Add yaml import


def read_hydra_config(stats_path):
    """Read Hydra config file and extract solver parameters."""
    config_path = Path(stats_path).parent / ".hydra/config.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        mode = config.get("mazesolver_mode", "unknown")
        steps = config.get("step", "unknown")
        return f"Mode: {mode}, Steps: {steps}"
    except (FileNotFoundError, yaml.YAMLError):
        return "Config not found or invalid"


def visualize_stats(json_path):
    """
    Visualize data from stats.json file

    Args:
        json_path (str): Path to the stats.json file
        output_dir (str, optional): Directory to save visualizations
    """

    output_dir = Path(json_path).parent / "figures"

    with open(json_path, "r") as f:
        stats = json.load(f)

    sns.set_theme(style="whitegrid")

    test_stats = stats.get("test_acc", {})

    solver_params = read_hydra_config(json_path)

    if "acc_by_iter" in test_stats:
        iterations = [int(k) for k in test_stats["acc_by_iter"].keys()]
        accuracies = list(test_stats["acc_by_iter"].values())

        plt.figure(figsize=(12, 6))
        plt.plot(
            iterations, accuracies, marker="o", linestyle="-", color="cornflowerblue"
        )
        plt.axhline(
            y=test_stats.get("stable_acc", 0),
            color="r",
            linestyle="--",
            label=f"Stable Accuracy: {stats.get('stable_acc', 0):.2f}%\n{solver_params}",
        )
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy per Iteration")
        plt.legend()
        plt.grid(True)

        if output_dir:
            plt.savefig(
                f"{output_dir}/accuracy_per_iteration.png", dpi=300, bbox_inches="tight"
            )
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.histplot(accuracies, kde=True, bins=20)
        plt.xlabel("Accuracy (%)")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of Accuracies\n{solver_params}")

        if output_dir:
            plt.savefig(
                f"{output_dir}/accuracy_distribution.png", dpi=300, bbox_inches="tight"
            )
        plt.show()

    return stats


if __name__ == "__main__":
    stats_path = "/home/elisa/deep-thinking/outputs/small_mazes_outputspace_step/testing-sometime-Sevag/stats.json"
    visualize_stats(stats_path)
