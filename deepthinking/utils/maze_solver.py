import numpy as np
from collections import deque

def parse_maze_input(rgb_maze):
    r, g, b = rgb_maze[0], rgb_maze[1], rgb_maze[2]
    wall_mask = (r + g + b != 0).astype(np.uint8)

    start_mask = (r == 1) & (g == 0) & (b == 0)
    goal_mask = (r == 0) & (g == 1) & (b == 0)

    try:
        start = tuple(np.argwhere(start_mask)[0])
        goal = tuple(np.argwhere(goal_mask)[0])
    except IndexError:
        raise ValueError("Start or goal not founddd")

    return wall_mask, start, goal

def is_valid_block(wall_mask, r, c):
    H, W = wall_mask.shape
    if r + 1 >= H or c + 1 >= W:
        return False
    return (wall_mask[r, c] and wall_mask[r+1, c] and wall_mask[r, c+1] and wall_mask[r+1, c+1])

def find_valid_block_start(wall_mask, pos):
    """Find a nearby top-left coordinate (r, c) of a valid 2by2 block around given position"""
    r0, c0 = pos
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            rr, cc = r0 + dr, c0 + dc
            if is_valid_block(wall_mask, rr, cc):
                return (rr, cc)
    raise ValueError("No valid 2x2 block found near position: ", pos)

def bfs_2x2_blocks(wall_mask, start_px, goal_px):
    H, W = wall_mask.shape
    visited = np.full((H-1, W-1), False)  # track top-lefts of visited 2x2 blocks
    parent = {}

    start = find_valid_block_start(wall_mask, start_px)
    goal = find_valid_block_start(wall_mask, goal_px)

    q = deque([start])
    visited[start] = True

    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break

        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:  # move one block in 2x2 grid
            nr, nc = r + dr, c + dc
            if 0 <= nr < H - 1 and 0 <= nc < W - 1 and is_valid_block(wall_mask, nr, nc) and not visited[nr, nc]:
                visited[nr, nc] = True
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))

    path = []
    curr = goal
    while curr != start:
        if curr not in parent:
            return []  # No path found
        path.append(curr)
        curr = parent[curr]
    path.append(start)
    path.reverse()

    # Generate step masks
    steps = []
    for i in range(1, len(path)+1):
        mask = np.zeros((H, W), dtype=np.uint8)
        for r, c in path[:i]:
            mask[r:r+2, c:c+2] = 1
        steps.append(mask)

    return steps

def get_intermediate_path_masks(rgb_maze_tensor):
    maze_np = rgb_maze_tensor.detach().cpu().numpy()
    wall_mask, start_px, goal_px = parse_maze_input(maze_np)
    return bfs_2x2_blocks(wall_mask, start_px, goal_px)
