import json

import matplotlib.pyplot as plt

run_path = "/home/elisa/deep-thinking/outputs/small_mazes_outputspace_step/testing-dockside-Otavia"
with open(f"{run_path}/stats.json", "r") as f:
    file = json.load(f)
    test_acc = file["test_acc"]
    train_acc = file["train_acc"]
    val_acc = file["val_acc"]

iterations = sorted(map(int, test_acc.keys()))
test_accuracies = [test_acc[str(i)] for i in iterations]
train_accuracies = [train_acc[str(i)] for i in iterations]
val_accuracies = [val_acc[str(i)] for i in iterations]

plt.figure(figsize=(10, 6))
plt.plot(iterations, test_accuracies, marker="o", label="Test", color="orange")
plt.plot(iterations, train_accuracies, marker="s", label="Train", color="blue")
plt.plot(iterations, val_accuracies, marker="^", label="Val", color="pink")
plt.title("Accuracy vs Number of Iterations")
plt.xlabel("Number of Iterations")
plt.ylabel("Accuracy (%)")

plt.xticks(range(30, max(iterations) + 1, 10))
plt.grid(axis="both", which="major")
plt.legend()
plt.tight_layout()
plt.savefig(f"{run_path}/accuracy_plot.png", dpi=300, bbox_inches="tight")
plt.show()
