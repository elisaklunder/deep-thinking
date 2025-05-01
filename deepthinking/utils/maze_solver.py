from collections import deque
import os
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# try to look at the actual outputs of the maze solver
# see what incorrect turns it takes
# plot the distribution of lengths that come out of the solver

Patch = Tuple[int, int]  # (row, col) on the patch grid
RGB = np.ndarray  # shape (3, H, W)


class MazeSolver:
    """
    Maze solver using BFS to find the optimal path in a maze.
    The maze is a grid of rgb patches (cells) where one patch = 2x2 pixels
    - Black (0, 0, 0) = Wall
    - White (1, 1, 1) = Free space
    - Red   (1, 0, 0) = Start patch (all 4 pixels red)
    - Green (0, 1, 0) = Goal  patch (all 4 pixels green)

    The 3 pixel outer border is automatically cropped by one pixel so the patch grid remains aligned.

    Public API:

    - get_patch_path_masks: frames that progressively add the optimal path discovered by bfs

    - get_patch_exploration_masks: starts with every patch discovered by bfs,
    then removes dead ends one by one in reverse order of discovery until only the
    shortest path remains.
    """

    def __init__(self):
        pass

    def _crop_outer_wall(self, rgb: RGB) -> Tuple[RGB, Tuple[int, int]]:
        """Crop outermost ring of pixels and return the offset."""
        return (rgb[:, 1:-1, 1:-1], (1, 1))

    def _view_as_patches(self, arr: np.ndarray) -> np.ndarray:
        """Return a view shaped (H2, 2, W2, 2) where H2 = H // 2 and W2 = W // 2."""
        h, w = arr.shape
        if (h | w) & 1:
            raise ValueError("Image dimensions must be multiples of 2.")
        return arr.reshape(h // 2, 2, w // 2, 2)

    def _parse_maze(self, rgb: RGB) -> Tuple[np.ndarray, Patch, Patch]:
        """Return free-mask, start-patch, goal-patch on the patch grid."""
        r, g, b = rgb
        pr, pg, pb = map(self._view_as_patches, (r, g, b))

        free = np.all((pr + pg + pb) != 0, axis=(1, 3))
        start = np.all((pr == 1) & (pg == 0) & (pb == 0), axis=(1, 3))
        goal = np.all((pr == 0) & (pg == 1) & (pb == 0), axis=(1, 3))

        s_idx, g_idx = map(np.argwhere, (start, goal))
        if s_idx.size == 0 or g_idx.size == 0:
            raise ValueError("Start or goal patch not found.")
        return free, tuple(s_idx[0]), tuple(g_idx[0])

    def _bfs_trace(
        self, free: np.ndarray, start: Patch, goal: Patch
    ) -> Tuple[List[Patch], List[Patch]]:
        """Perform BFS and return (optimal_path, discovery_sequence)."""
        h2, w2 = free.shape
        visited = np.zeros_like(free, bool)
        parent = {}

        q = deque([start])
        visited[start] = True
        discovered: List[Patch] = [start]

        while q:
            r, c = q.popleft()
            if (r, c) == goal:
                break
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < h2
                    and 0 <= nc < w2
                    and free[nr, nc]
                    and not visited[nr, nc]
                ):
                    visited[nr, nc] = True
                    parent[(nr, nc)] = (r, c)
                    q.append((nr, nc))
                    discovered.append((nr, nc))

        # Reconstruct shortest path
        if goal not in parent and start != goal:
            return [], discovered
        path = [goal]
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()
        return path, discovered

    def _patch_to_pixels(
        self, pr: int, pc: int, offs: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Convert patch coordinates to pixel coordinates."""
        return 2 * pr + offs[0], 2 * pc + offs[1]

    def _toggle_patch(
        self, mask: np.ndarray, patch: Patch, offs: Tuple[int, int], val: int
    ) -> None:
        """Set the pixels of a patch to a given value."""
        r0, c0 = self._patch_to_pixels(*patch, offs)
        mask[r0 : r0 + 2, c0 : c0 + 2] = val

    def _mask_from_patches(
        self, patches: Sequence[Patch], shape: Tuple[int, int], offs: Tuple[int, int]
    ) -> np.ndarray:
        """Create a mask with originall maze shape from a list of patches."""
        m = np.zeros(shape, np.uint8)
        for p in patches:
            self._toggle_patch(m, p, offs, 1)
        return m

    def _frames_remove(
        self, sequence: Sequence[Patch], offs: Tuple[int, int], start_mask: np.ndarray
    ) -> List[np.ndarray]:
        """Yield frames while clearing patches from start_mask."""
        m = start_mask.copy()
        frames = [m.copy()]
        for p in sequence:
            self._toggle_patch(m, p, offs, 0)
            frames.append(m.copy())
        return frames

    def get_intermediate_path_masks(self, input_rgb) -> List[np.ndarray]:
        """Incrementally light up the optimal path."""
        rgb, offs = self._crop_outer_wall(input_rgb)
        free, start, goal = self._parse_maze(rgb)
        path, _ = self._bfs_trace(free, start, goal)
        if not path:
            return []
        H, W = input_rgb.shape[1:]
        return [
            self._mask_from_patches(path[:i], (H, W), offs)
            for i in range(1, len(path) + 1)
        ]

    def get_reverse_exploration_masks(self, input_rgb) -> List[np.ndarray]:
        """Start with all explored, then peel off dead ends."""
        rgb, offs = self._crop_outer_wall(input_rgb)
        free, start, goal = self._parse_maze(rgb)
        path, discovered = self._bfs_trace(free, start, goal)

        H, W = input_rgb.shape[1:]
        full = self._mask_from_patches(discovered, (H, W), offs)

        if path:
            dead_ends = [p for p in reversed(discovered) if p not in set(path)]
        else:  # unsolvable
            dead_ends = list(reversed(discovered))

        frames = self._frames_remove(dead_ends, offs, full)
        final = (
            self._mask_from_patches(path, (H, W), offs)
            if path
            else np.zeros((H, W), np.uint8)
        )
        if not np.array_equal(frames[-1], final):
            frames.append(final)
        return frames

    import numpy as np


def plot_solution_length_distribution(
    dataset_path: str,
):
    """
    Plot histograms of frame lengths produced by MazeSolver on entire train dataset

    Returns
    (inter_lengths, reverse_lengths, stats) : tuple
        Lists of lengths and a pandas DataFrame with stats.
    """
    dataset_name = dataset_path.split("/")[-2]
    mazes = np.load(dataset_path)
    solver = MazeSolver()

    inter_lengths, reverse_lengths = [], []
    for maze in tqdm(mazes):
        inter_lengths.append(len(solver.get_intermediate_path_masks(maze)))
        reverse_lengths.append(len(solver.get_reverse_exploration_masks(maze)))

    stats = pd.DataFrame({"intermediate": inter_lengths, "reverse": reverse_lengths})
    print(stats.describe().to_string(float_format="%.2f"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, dpi=120)

    for ax, data, label in zip(
        axes,
        [inter_lengths, reverse_lengths],
        ["Optimal path frames", "Reverse-exploration frames"],
    ):
        bins = range(min(data), max(data) + 3, 2)
        ax.hist(data, bins=bins, edgecolor="black", alpha=0.85)

        mean, med = np.mean(data), np.median(data)
        ax.axvline(
            mean,
            color="tab:orange",
            linestyle="--",
            linewidth=2,
            label=f"mean = {mean:.1f}",
        )
        ax.axvline(
            med,
            color="tab:green",
            linestyle=":",
            linewidth=2,
            label=f"median = {med:.1f}",
        )

        ax.set_title(label, fontsize=12, pad=10)
        ax.set_xlabel("frames")
        ax.grid(alpha=0.3, linestyle=":")
        ax.legend(frameon=False)

    axes[0].set_ylabel("frequency")

    fig.suptitle(
        f"MazeSolver solution length distributions for {dataset_name}",
        fontsize=16,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if not os.path.exists("figures"):
        os.makedirs("figures")
    fig.savefig(f"figures/length_dist_{dataset_name}", dpi=300)
    
    return inter_lengths, reverse_lengths, stats


if __name__ == "__main__":
    path = "data/maze_data_test_33/inputs.npy"
    inputs = np.load(path)

    # MAZE_INDEX = 1
    # print("Original maze:")
    # from deepthinking.utils.plot import display_colored_maze

    # maze = inputs[MAZE_INDEX]
    # display_colored_maze(maze)

    # print("Path masks:")
    # maze_solver = MazeSolver()
    # masks = maze_solver.get_intermediate_path_masks(maze)
    # for mask in masks:
    #     display_colored_maze(mask)
    #     print("\n")

    # print("Exploration masks:")
    # masks = maze_solver.get_reverse_exploration_masks(maze)
    # for mask in masks:
    #     display_colored_maze(mask)
    #     print("\n")

    inter_lengths, reverse_lengths, stats = plot_solution_length_distribution(path)
