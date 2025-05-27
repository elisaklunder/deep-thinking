import math
import os
from datetime import datetime

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import seaborn as sns
import torch
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap

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


def display_colored_maze(maze: np.ndarray) -> None:
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


def plot_heatmap_sequence(
    input_maze,  # (3, H, W) - input maze in RGB
    target,  # (H, W) – target
    probs_seq,  # (T, H, W) float – probabilities
    steps,  # list[int] – nbr time‑steps
    title_prefix="sample_0",
    dpi=120,
    *,
    masks_per_row=10,
    wall_colour=(0, 0, 0),
):
    """
    Draw a heatmap for several iterations of the networks output,
    masking out wall cells. Now includes target solution as second frame.
    """
    os.makedirs("figures", exist_ok=True)

    n_panels = len(steps) + 2
    n_rows = math.ceil(n_panels / masks_per_row)
    fig_w = 4 * masks_per_row
    fig_h = 4 * n_rows
    fig, axs = plt.subplots(n_rows, masks_per_row, figsize=(fig_w, fig_h), dpi=dpi)
    axs = axs.ravel()

    maze_rgb = np.transpose(input_maze, (1, 2, 0))  # (H, W, 3)
    wall_mask = np.all(maze_rgb == wall_colour, axis=-1)  # (H, W) bool

    _BLACK_RED_WHITE = LinearSegmentedColormap.from_list(
        "black_red_white", [(0.0, "#000000"), (0.5, "#ff0000"), (1.0, "#ffffff")], N=256
    )
    _BLACK_RED_WHITE.set_bad(color="#000000")

    axs[0].imshow(maze_rgb)
    axs[0].set_title("Input maze")
    axs[0].axis("off")

    if target is not None:
        masked_target = ma.array(target, mask=wall_mask)
        axs[1].imshow(masked_target, cmap=_BLACK_RED_WHITE, vmin=0.0, vmax=1.0)
        axs[1].set_title("Target solution")
        axs[1].axis("off")

    last_im = None
    for k, step in enumerate(
        steps, start=2
    ):  # Start from 2 to account for input and target
        ax = axs[k]
        masked_probs = ma.array(probs_seq[step], mask=wall_mask)
        im = ax.imshow(
            masked_probs,
            cmap=_BLACK_RED_WHITE,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        last_im = im
        ax.set_title(f"step {step + 1}")
        ax.axis("off")

    for ax in axs[n_panels:]:
        ax.set_visible(False)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(last_im, cax=cbar_ax, label="Path probability")

    fig.suptitle(title_prefix, fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    fig.savefig(f"figures/heatmap_steps_{title_prefix}.png")
    plt.close(fig)


def plot_maze_and_target(input, targets, save_str=None):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    ax = axs[0]
    ax.imshow(
        np.transpose(input.squeeze(), (1, 2, 0))
        if isinstance(input, np.ndarray)
        else input.cpu().squeeze().permute(1, 2, 0)
    )

    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])

    ax = axs[1]
    sns.heatmap(targets, ax=ax, cbar=False, linewidths=0, square=True, rasterized=True)

    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])

    plt.tight_layout()
    if save_str is None:
        save_str = (
            f"figures/maze_example_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.png"
        )
    plt.savefig(save_str, bbox_inches="tight")
    plt.close()


def plot_maze_and_intermediate_masks(
    inp, masks, masks_per_row=10, type="sometype", save_str=None
):
    n_masks = len(masks)
    n_rows = (n_masks + masks_per_row - 1) // masks_per_row  # ceil div

    fig, axs = plt.subplots(
        n_rows + 1, masks_per_row, figsize=(2.0 * masks_per_row, 2.0 * (n_rows + 1))
    )
    axs = axs.ravel()

    ax0 = axs[0]
    if isinstance(inp, np.ndarray):
        img = np.transpose(inp.squeeze(), (1, 2, 0))
    else:
        img = inp.cpu().squeeze().permute(1, 2, 0)
    ax0.imshow(img)
    ax0.axis("off")

    for idx, mask in enumerate(masks, start=1):
        ax = axs[idx]
        sns.heatmap(
            mask, ax=ax, cbar=False, xticklabels=False, yticklabels=False, linewidths=0
        )
        ax.axis("off")

    for ax in axs[len(masks) + 1 :]:
        ax.set_visible(False)

    fig.tight_layout()
    if save_str is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_str = f"figures/masks_example_{type}-{ts}.png"
    fig.savefig(save_str, bbox_inches="tight")
    plt.close(fig)


def animate_prediction_sequence(
    input_maze,
    target,
    probs_seq,
    title_prefix="sample_0",
    frame_duration=0.5,  # in seconds
    wall_colour=(0, 0, 0),
):
    """Creates an animation with static input/target and animated predictions."""
    os.makedirs("figures", exist_ok=True)

    fig, (ax_input, ax_pred, ax_target) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title_prefix, fontsize=14)

    maze_rgb = np.transpose(input_maze, (1, 2, 0))
    wall_mask = np.all(maze_rgb == wall_colour, axis=-1)

    ax_input.imshow(maze_rgb)
    ax_input.set_title("Input maze")
    ax_input.axis("off")

    _BLACK_RED_WHITE = LinearSegmentedColormap.from_list(
        "black_red_white", [(0.0, "#000000"), (0.5, "#ff0000"), (1.0, "#ffffff")], N=256
    )
    _BLACK_RED_WHITE.set_bad(color="#000000")

    masked_target = ma.array(target, mask=wall_mask)
    ax_target.imshow(masked_target, cmap=_BLACK_RED_WHITE, vmin=0.0, vmax=1.0)
    ax_target.set_title("Target solution")
    ax_target.axis("off")

    im_pred = ax_pred.imshow(
        ma.array(probs_seq[0], mask=wall_mask),
        cmap=_BLACK_RED_WHITE,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax_pred.set_title("Prediction")
    ax_pred.axis("off")

    def update(frame):
        im_pred.set_array(ma.array(probs_seq[frame], mask=wall_mask))
        return [im_pred]

    frames = len(probs_seq)
    interval = frame_duration * 1000
    anim = animation.FuncAnimation(
        fig, update, frames=frames, interval=interval, blit=True
    )

    writer = FFMpegWriter(fps=1 / frame_duration)
    anim.save(f"figures/prediction_animation_{title_prefix}.mp4", writer=writer)
    plt.close()


if __name__ == "__main__":
    in_path = "data/maze_data_test_11/inputs.npy"
    target_path = "data/maze_data_test_11/solutions.npy"
    inputs_np = np.load(in_path)
    inputs = torch.from_numpy(inputs_np).float()
    targets_np = np.load(target_path)
    targets = torch.from_numpy(targets_np).float()
    MAZE_INDEX = 1
    input = inputs[MAZE_INDEX]
    target = targets[MAZE_INDEX]
    plot_maze_and_target(input, target)
