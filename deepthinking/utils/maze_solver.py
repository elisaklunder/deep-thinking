import heapq
import os
from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

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

    def __init__(self, mode: Optional[str] = "incremental"):
        self.mode = mode

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
        self,
        sequence: Sequence[Patch],
        offs: Tuple[int, int],
        start_mask: np.ndarray,
        step: int = 1,
    ) -> List[np.ndarray]:
        """Yield frames while clearing patches from start_mask."""
        m = start_mask.copy()
        frames = [m.copy()]
        for i, p in enumerate(sequence):
            self._toggle_patch(m, p, offs, 0)
            if (i + 1) % step == 0:  # append frame every step frames
                frames.append(m.copy())
        return frames + [m.copy()]  # + final state

    @staticmethod
    def _manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _astar_trace(
        self, free: np.ndarray, start: Patch, goal: Patch
    ) -> Tuple[List[Patch], List[Patch]]:
        """Perform A* search"""
        h2, w2 = free.shape
        discovered: List[Patch] = [start]

        open_set = [
            (
                self._manhattan_distance(start, goal) + 0,
                self._manhattan_distance(start, goal),
                start,
            )
        ]
        heapq.heapify(open_set)
        closed_set: Set[Patch] = set()

        open_set_elements = {start}
        g_score: Dict[Patch, int] = {start: 0}
        parent: Dict[Patch, Patch] = {}
        
        while open_set:
            _, _, current_node = heapq.heappop(open_set)
            open_set_elements.remove(current_node)
            
            if current_node in closed_set:
                continue

            closed_set.add(current_node)

            if current_node == goal:
                
                path = []
                node = current_node
                while node in parent: 
                    path.append(node)
                    node = parent[node]
                path.append(start)
                path.reverse() 
                return path, discovered
                

            for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = current_node[0] + drow, current_node[1] + dcol
                neighbor = (nr, nc)

                if not (0 <= nr < h2 and 0 <= nc < w2 and free[nr, nc]):
                    continue

                tentative_g = g_score[current_node] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    parent[neighbor] = current_node
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._manhattan_distance(neighbor, goal)

                    if neighbor not in open_set_elements:
                        heapq.heappush(
                            open_set,
                            (
                                f_score,
                                self._manhattan_distance(neighbor, goal),
                                neighbor,
                            ),
                        )
                        open_set_elements.add(neighbor)
                        discovered.append(neighbor)

        return [], discovered
     

    def get_incremental_path_masks(self, input_rgb, step) -> List[np.ndarray]:
        """Incrementally light up the optimal path."""
        rgb, offs = self._crop_outer_wall(input_rgb)
        free, start, goal = self._parse_maze(rgb)
        path, _ = self._bfs_trace(free, start, goal)

        if not path:
            return []

        H, W = input_rgb.shape[1:]
        steps = [
            self._mask_from_patches(path[:i], (H, W), offs)
            for i in range(1, len(path) + 1, step)
        ]
        for i in range(10):
            steps.append(self._mask_from_patches(path, (H, W), offs))

        return steps

    def get_incremental_path_masks_bidirectional(
        self, input_rgb, step
    ) -> List[np.ndarray]:
        rgb, offs = self._crop_outer_wall(input_rgb)
        free, start, goal = self._parse_maze(rgb)
        path, _ = self._bfs_trace(free, start, goal)

        if not path:
            return []

        H, W = input_rgb.shape[1:]
        steps = []
        path_len = len(path)

        for i in range(0, path_len // 2 + 1, step):
            current_path = path[:i] + path[-(i if i > 0 else 0) :]
            steps.append(self._mask_from_patches(current_path, (H, W), offs))

        for _ in range(1, 10):
            steps.append(self._mask_from_patches(path, (H, W), offs))

        return steps

    def get_reverse_exploration_masks(self, input_rgb, step) -> List[np.ndarray]:
        """Start with all explored, then peel off dead ends."""
        rgb, offs = self._crop_outer_wall(input_rgb)
        free, start, goal = self._parse_maze(rgb)
        path, discovered = self._bfs_trace(free, start, goal)

        H, W = input_rgb.shape[1:]
        full = self._mask_from_patches(discovered, (H, W), offs)

        dead_ends = [p for p in reversed(discovered) if p not in set(path)]
        frames = self._frames_remove(dead_ends, offs, full, step=step)

        final = self._mask_from_patches(path, (H, W), offs)
        for _ in range(10):
            frames.append(final)

        return frames

    import numpy as np

    def get_bfs_masks(self, input_rgb, step) -> List[np.ndarray]:
        """Return masks of the BFS discovery sequence."""
        rgb, offs = self._crop_outer_wall(input_rgb)
        free, start, goal = self._parse_maze(rgb)
        optimal_path, discovered = self._bfs_trace(free, start, goal)

        H, W = input_rgb.shape[1:]
        discovery_masks = [
            self._mask_from_patches(discovered[:i], (H, W), offs)
            for i in range(1, len(discovered) + 1)
        ]
        discovery_masks.append(self._mask_from_patches(optimal_path, (H, W), offs))
        return discovery_masks

    def get_astar_masks(self, input_rgb, step) -> List[np.ndarray]:
        """Masks showing A* search progression and final path."""
        rgb, offs = self._crop_outer_wall(input_rgb)
        free, start, goal = self._parse_maze(rgb)
        optimal_path, discovered = self._astar_trace(free, start, goal)
    
        H, W = input_rgb.shape[1:]
        
        exploration_masks = [
            self._mask_from_patches(discovered[:i], (H, W), offs)
            for i in range(1, len(discovered) + 1, step)
        ]
        
        optimal_path_mask = self._mask_from_patches(optimal_path, (H, W), offs)
        for _ in range(10):
            exploration_masks.append(optimal_path_mask)
            
        return exploration_masks

    def get_intermediate_supervision_masks(self, input_rgb, step=1) -> List[np.ndarray]:
        if self.mode == "incremental":
            return self.get_incremental_path_masks(input_rgb, step=step)
        elif self.mode == "reverse_exploration":
            return self.get_reverse_exploration_masks(input_rgb, step=step)
        elif self.mode == "bfs":
            return self.get_bfs_masks(input_rgb, step=step)
        elif self.mode == "incremental_bidirectional":
            return self.get_incremental_path_masks_bidirectional(input_rgb, step=step)
        elif self.mode == "astar":
            return self.get_astar_masks(input_rgb, step=step)


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

    inter_lengths, reverse_lengths, bfs_lengths = [], [], []
    for maze in tqdm(mazes):
        inter_lengths.append(len(solver.get_incremental_path_masks(maze)))
        reverse_lengths.append(len(solver.get_reverse_exploration_masks(maze)))
        bfs_lengths.append(len(solver.get_bfs_masks(maze)))

    stats = pd.DataFrame(
        {"incremental": inter_lengths, "reverse": reverse_lengths, "bfs": bfs_lengths}
    )
    print(stats.describe().to_string(float_format="%.2f"))

    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True, dpi=120)

    for ax, data, label in zip(
        axes,
        [inter_lengths, reverse_lengths, bfs_lengths],
        ["Incremental path frames", "Reverse-exploration frames", "BFS frames"],
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
    fig.savefig(f"figures/lengths_dist_{dataset_name}", dpi=300)

    return inter_lengths, reverse_lengths, bfs_lengths, stats


if __name__ == "__main__":
    from deepthinking.utils.plot import (
        display_colored_maze,
        plot_maze_and_intermediate_masks,
    )

    path = "data/maze_data_test_33/inputs.npy"
    inputs = np.load(path)

    MAZE_INDEX = 1
    maze = inputs[MAZE_INDEX]
    maze_solver = MazeSolver()

    # masks_incremental = maze_solver.get_incremental_path_masks(maze)
    # masks_reverse = maze_solver.get_reverse_exploration_masks(maze)
    # masks_bfs = maze_solver.get_bfs_masks(maze)
    # masks_bidirectional = maze_solver.get_incremental_path_masks_bidirectional(
    #     maze, step=1
    # )
    # plot_maze_and_intermediate_masks(maze, masks_bidirectional, type="bidirectional")

    # Plot the masks
    masks_astar = maze_solver.get_astar_masks(maze, step=3)
    plot_maze_and_intermediate_masks(maze, masks_astar, type="astar")

    # Plot the masks
    # plot_maze_and_intermediate_masks(maze, masks_incremental, type="intermediate")
    # plot_maze_and_intermediate_masks(maze, masks_reverse, type="reverse")
    # plot_maze_and_intermediate_masks(maze, masks_bfs, type="bfs")

    # inter_lengths, reverse_lengths, bfs_lengths, stats = (
    #     plot_solution_length_distribution(path)
    # )
