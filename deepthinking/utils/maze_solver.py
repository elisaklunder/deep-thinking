import heapq
import os
from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from deepthinking.utils.plot import plot_maze_and_intermediate_masks_ultra_fast

Patch = Tuple[int, int]  # (row, col) on the patch grid
RGB = np.ndarray  # shape (3, H, W)

class MazeSolver:
    """
    Maze solver using different algorithms to find the optimal path in a maze.
    The maze is a grid of rgb patches (cells) where one patch = 2x2 pixels
    - Black (0, 0, 0) = Wall
    - White (1, 1, 1) = Free space
    - Red   (1, 0, 0) = Start patch (all 4 pixels red)
    - Green (0, 1, 0) = Goal  patch (all 4 pixels green)
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

    def _mask_from_patches_with_values(
        self,
        patches_values: List[Tuple[Patch, int]],
        shape: Tuple[int, int],
        offs: Tuple[int, int],
    ) -> np.ndarray:
        """Create a mask with different values for different patch types."""
        m = np.zeros(shape, np.uint8)
        for patch, value in patches_values:
            self._toggle_patch(m, patch, offs, value)
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

    def _astar_trace_enhanced(
        self, free: np.ndarray, start: Patch, goal: Patch
    ) -> Tuple[List[Patch], List[Dict[str, List[Patch]]]]:
        """
        Returns:
        - optimal_path: The shortest path found
        - algorithm_states: List of dictionaries containing state at each step
        """
        h2, w2 = free.shape
        algorithm_states = []

        open_set = [
            (
                self._manhattan_distance(start, goal),
                self._manhattan_distance(start, goal),
                start,
            )
        ]
        heapq.heapify(open_set)
        closed_set: Set[Patch] = set()
        open_set_elements = {start}
        g_score: Dict[Patch, int] = {start: 0}
        parent: Dict[Patch, Patch] = {}

        step = 0

        while open_set:
            current_state = {
                "open_set": list(open_set_elements),
                "closed_set": list(closed_set),
                "current_node": None,
                "neighbors_explored": [],
                "step": step,
            }

            _, _, current_node = heapq.heappop(open_set)
            open_set_elements.remove(current_node)
            current_state["current_node"] = current_node

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
                current_state["path_found"] = True
                algorithm_states.append(current_state)
                return path, algorithm_states

            neighbors_this_step = []
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
                        neighbors_this_step.append(neighbor)

            current_state["neighbors_explored"] = neighbors_this_step
            algorithm_states.append(current_state)
            step += 1

        return [], algorithm_states

    def get_dfs_masks_colored(self, input_rgb, step=1) -> List[np.ndarray]:
        """
        - 0: Unvisited
        - 1: DFS-explored patches - Blue
        - 2: Dead ends - Red
        - 3: Final path - Green
        """

        rgb, offs = self._crop_outer_wall(input_rgb)
        free, start, goal = self._parse_maze(rgb)
        H, W = input_rgb.shape[1:]

        h2, w2 = free.shape
        visited = np.zeros_like(free, bool)
        parent = {}
        path = []
        discovered: List[Patch] = []
        dead_ends: Set[Patch] = set()
        found = False

        def dfs(patch: Patch):
            nonlocal found
            if found:
                return
            visited[patch] = True
            discovered.append(patch)

            if patch == goal:
                found = True
                return

            is_dead_end = True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = patch[0] + dr, patch[1] + dc
                neighbor = (nr, nc)
                if (
                    (0 <= nr < h2)
                    and (0 <= nc < w2)
                    and free[nr, nc]
                    and not visited[nr, nc]
                ):
                    parent[neighbor] = patch
                    dfs(neighbor)
                    if found:
                        return
                    is_dead_end = False

            if is_dead_end:
                dead_ends.add(patch)

        dfs(start)

        if goal not in parent and start != goal:
            return []

        path = [goal]
        while path[-1] != start:
            path.append(parent[path[-1]])
        path.reverse()
        path_set = set(path)

        frames = []
        for i in range(1, len(discovered) + 1, step):
            patches_values = []
            for patch in discovered[:i]:
                if patch in dead_ends:
                    patches_values.append((patch, 2))
                else:
                    patches_values.append((patch, 1))
            frame = self._mask_from_patches_with_values(patches_values, (H, W), offs)
            frames.append(frame)

        final_frame = np.zeros((H, W), np.uint8)
        for patch in discovered:
            if patch in dead_ends:
                val = 2
            elif patch in path_set:
                val = 3
            else:
                val = 1
            self._toggle_patch(final_frame, patch, offs, val)
        for _ in range(10):
            frames.append(final_frame)

        return frames

    def get_dfs_masks(self, input_rgb, step=1) -> List[np.ndarray]:
        """
        DFS with backtracking
        """

        rgb, offs = self._crop_outer_wall(input_rgb)
        free, start, goal = self._parse_maze(rgb)
        H, W = input_rgb.shape[1:]

        h2, w2 = free.shape
        visited = np.zeros_like(free, bool)
        current_path = []
        frames = []
        frame_counter = 0
        found = False

        def dfs(patch: Patch):
            nonlocal found, frame_counter

            if found:
                return True

            visited[patch] = True
            current_path.append(patch)

            frame_counter += 1
            if frame_counter % step == 0:
                frame = self._mask_from_patches(current_path, (H, W), offs)
                frames.append(frame)

            if patch == goal:
                found = True
                return True

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = patch[0] + dr, patch[1] + dc
                neighbor = (nr, nc)

                if (
                    (0 <= nr < h2)
                    and (0 <= nc < w2)
                    and free[nr, nc]
                    and not visited[nr, nc]
                ):
                    if dfs(neighbor):
                        return True

            current_path.pop()

            frame_counter += 1
            if frame_counter % step == 0:
                frame = self._mask_from_patches(current_path, (H, W), offs)
                frames.append(frame)

            return False

        dfs(start)

        if found:
            solution_visited = np.zeros_like(free, bool)
            solution_path = []

            def find_solution_path(patch: Patch, path: List[Patch]):
                solution_visited[patch] = True
                path.append(patch)

                if patch == goal:
                    return True

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = patch[0] + dr, patch[1] + dc
                    neighbor = (nr, nc)

                    if (
                        (0 <= nr < h2)
                        and (0 <= nc < w2)
                        and free[nr, nc]
                        and not solution_visited[nr, nc]
                    ):
                        if find_solution_path(neighbor, path):
                            return True

                path.pop()
                solution_visited[patch] = False
                return False

            find_solution_path(start, solution_path)

            final_frame = self._mask_from_patches(solution_path, (H, W), offs)

            for _ in range(10):
                frames.append(final_frame)

        return frames

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

    def get_astar_masks_colored(self, input_rgb, step=1) -> List[np.ndarray]:
        """
        0: Unexplored (walls/unvisited)
        1: Open set (candidates for exploration) - Cyan
        2: Closed set (already explored) - Orange
        3: Current node (being processed) - Yellow
        4: Optimal path (final solution) - Magenta
        5: Dead ends (explored but not optimal) - Purple
        """
        rgb, offs = self._crop_outer_wall(input_rgb)
        free, start, goal = self._parse_maze(rgb)
        optimal_path, algorithm_states = self._astar_trace_enhanced(free, start, goal)

        H, W = input_rgb.shape[1:]
        masks = []

        for i, state in enumerate(algorithm_states):
            if i % step != 0:
                continue

            patches_values = []

            for patch in state["closed_set"]:
                patches_values.append((patch, 2))

            for patch in state["open_set"]:
                patches_values.append((patch, 1))

            if state["current_node"]:
                patches_values.append((state["current_node"], 3))

            for patch in state["neighbors_explored"]:
                patches_values.append((patch, 1))

            mask = self._mask_from_patches_with_values(patches_values, (H, W), offs)
            masks.append(mask)

        if optimal_path:
            patches_values = []
            optimal_set = set(optimal_path)

            if algorithm_states:
                all_explored = set()
                for state in algorithm_states:
                    all_explored.update(state["closed_set"])

                for patch in all_explored:
                    if patch not in optimal_set:
                        patches_values.append((patch, 5))

            for patch in optimal_path:
                patches_values.append((patch, 4))

            final_mask = self._mask_from_patches_with_values(
                patches_values, (H, W), offs
            )

            for _ in range(10):
                masks.append(final_mask)

        return masks

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
        elif self.mode == "dfs":
            return self.get_dfs_masks(input_rgb, step=step)
        

def colormap_for_astar():
    colors = [
        [0, 0, 0],  # 0: Black - unexplored
        [0, 1, 1],  # 1: Cyan - open set
        [1, 0.5, 0],  # 2: Orange - closed set
        [1, 1, 0],  # 3: Yellow - current node
        [1, 0, 0],  # 4: Red - optimal path
        [0.8, 0.8, 0.8],  # 5: Light gray - dead ends
    ]
    return colors

def colormap_for_dfs():
    colors = [
        [0, 0, 0],  # 0: Black
        [0, 0, 1],  # 1: Blue
        [1, 0, 0],  # 2: Red
        [0, 1, 0],  # 3: Green
    ]
    return colors

def plot_colored_masks(masks, colormap, save_path=None):
    num_masks = len(masks)
    cols = 10
    rows = (num_masks + cols - 1) // cols

    fig = plt.figure(figsize=(16, 2.0 * rows))
    gs = fig.add_gridspec(rows, cols, hspace=0, wspace=0.1)
    axes = []
    for i in range(rows):
        for j in range(cols):
            axes.append(fig.add_subplot(gs[i, j]))

    for i, mask in enumerate(masks):
        if i < len(axes):
            ax = axes[i]

            h, w = mask.shape
            rgb_img = np.zeros((h, w, 3))
            for val in range(6):
                rgb_img[mask == val] = colormap[val]

            ax.imshow(rgb_img)
            ax.set_title(f"Step {i}")
            ax.axis("off")

    for i in range(num_masks, len(axes)):
        axes[i].axis("off")

    plt.suptitle(
        "A* Algorithm Learning Progression\n(Cyan=Open, Orange=Closed, Yellow=Current, Magenta=Path, Purple=Dead End)",
        fontsize=14,
        y=0.98,
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig

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
    path = "data/maze_data_test_11/inputs.npy"
    inputs = np.load(path)

    MAZE_INDEX = 2
    maze = inputs[MAZE_INDEX]
    maze_solver = MazeSolver()

    # masks = maze_solver.get_astar_masks_colored(maze, step=3)
    # masks = maze_solver.get_dfs_masks_colored(maze, step=1)
    # colors = colormap_for_astar()
    # fig = plot_colored_masks(masks, colors, save_path="figures/dfs_colored_masks.png")

    masks_dfs = maze_solver.get_dfs_masks_binary(maze, step=1)
    plot_maze_and_intermediate_masks_ultra_fast(maze, masks_dfs, type="dfs_binary")
