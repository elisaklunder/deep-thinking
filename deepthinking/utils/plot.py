import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from deepthinking.utils.maze_solver import get_intermediate_path_masks

MAZE_INDEX = 8


def get_color_block(r: int, g: int, b: int) -> str:
    if r == 1 and g == 0 and b == 0:
        return "🟥"  # Red
    if r == 0 and g == 1 and b == 0:
        return "🟩"  # Green
    if r == 1 and g == 1 and b == 1:
        return "⬜"  # White
    return "⬛"


def display_path_mask_as_overlay(input_maze, path_mask):
    for i in range(input_maze.shape[1]):
        for j in range(input_maze.shape[2]):
            r, g, b = input_maze[0, i, j], input_maze[1, i, j], input_maze[2, i, j]
            if path_mask[i, j] == 1:
                print("🟦", end="")
            else:
                print(get_color_block(r, g, b), end="")
        print()


def display_colored_maze(file_path: str, maze_index: int = 0) -> None:
    print("Array shape:", end=" ")
    try:
        data = np.load(file_path)
        print(data.shape)
    except Exception as e:
        print(f"Error: {e}")
        return
    try:
        data = np.load(file_path)
        if maze_index >= data.shape[0]:
            raise ValueError(f"Maze index {maze_index} out of bounds")

        maze = data[maze_index]

        # Soultions
        if len(maze.shape) == 2:
            for i in range(maze.shape[0]):
                for j in range(maze.shape[1]):
                    color_code = "⬜" if maze[i, j] == 1 else "⬛"
                    print(color_code, end="")
                print()
        # Inputs
        else:
            for i in range(maze.shape[1]):
                for j in range(maze.shape[2]):
                    r, g, b = maze[0, i, j], maze[1, i, j], maze[2, i, j]
                    color_code = get_color_block(r, g, b)
                    print(color_code, end="")
                print()

    except Exception as e:
        print(f"Error: {e}")


def plot_prediction_heatmap(
    input_maze,
    probs_seq,  # (T, H, W) – probabilities for “path”
    steps,
    title_prefix="sample_0",
    dpi=120,
):
    """
    Draw a heat-map for several iterations of the networks output.
    """

    os.makedirs("figures", exist_ok=True)

    n_cols = len(steps) + 1
    fig, axs = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), dpi=dpi)

    ax0 = axs[0]
    maze_rgb = np.transpose(input_maze, (1, 2, 0))  # (H, W, 3)
    ax0.imshow(maze_rgb)
    ax0.set_title("maze (RGB)")
    ax0.axis("off")

    _BLACK_RED_WHITE = LinearSegmentedColormap.from_list(
        "black_red_white",
        [(0.0, "#000000"), (0.5, "#ff0000"), (1.0, "#ffffff")],
        N=256,
    )

    last_im = None
    for k, step in enumerate(steps, start=1):
        ax = axs[k]
        im = ax.imshow(
            probs_seq[step],  # (H, W)
            cmap=_BLACK_RED_WHITE,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        last_im = im
        ax.set_title(f"step {step + 1}")
        ax.axis("off")

    # shared colour-bar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(last_im, cax=cbar_ax, label="Path probability")

    fig.suptitle(title_prefix, fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(f"figures/heatmap_steps_{title_prefix}.png")
    plt.close()


if __name__ == "__main__":
    path = "data/maze_data_test_11/inputs.npy"
    inputs = np.load(path)
    print("Inputs shape:", inputs.shape)

    print("Original maze input:")
    display_colored_maze(path, MAZE_INDEX)

    maze_tensor = torch.tensor(inputs[MAZE_INDEX], dtype=torch.float32)
    steps = get_intermediate_path_masks(maze_tensor)

    print(f"Found path of {len(steps)} steps.")
    for i, step in enumerate(steps):
        print(f"\nStep {i + 1}")
        display_path_mask_as_overlay(maze_tensor.numpy(), step)
