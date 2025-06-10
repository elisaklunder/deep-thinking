# import os

# pylint: disable=C0415, C0114, E0606


def test_easy_to_hard_data():
    """Test the easy-to-hard-data maze dataset loading"""
    try:
        from easy_to_hard_data import MazeDataset as EasyToHardMazeDataset

        print("Successfully imported easy-to-hard-data")

        # Create a small test dataset
        dataset = EasyToHardMazeDataset("./data", train=True, size=9, download=True)
        print(f"Dataset loaded with {len(dataset)} mazes")

        # Check a sample
        x, y = dataset[0]
        print(f"Sample maze shape: {x.shape}, solution shape: {y.shape}")
        print(
            f"Value ranges - maze: [{x.min():.4f}, {x.max():.4f}], solution: [{y.min():.4f}, {y.max():.4f}]"
        )

        return True
    except Exception as e:
        print(f"Error testing easy-to-hard-data: {e}")
        return False


def test_maze_dataset():
    """Test the maze-dataset library loading"""
    try:
        from maze_dataset import set_serialize_minimal_threshold
        from maze_dataset.dataset.rasterized import (
            MazeDataset,
            MazeDatasetConfig,
            RasterizedMazeDataset,
        )
        from maze_dataset.generation import LatticeMazeGenerators

        set_serialize_minimal_threshold(
            int(10**7)
        )  # set this threshold to prevent crashing on large datasets. Will be fixed soon.
        print("Successfully imported maze-dataset")

        grid_n = 9
        base_dataset = MazeDataset.from_config(
            MazeDatasetConfig(
                name="test",
                grid_n=grid_n,
                n_mazes=5,
                seed=42,
                maze_ctor=LatticeMazeGenerators.gen_dfs,
                maze_ctor_kwargs={},
                endpoint_kwargs=dict(deadend_start=True, endpoints_not_equal=True),
            ),
            local_base_path="./data/maze-dataset/",
        )

        maze_dataset = RasterizedMazeDataset.from_base_MazeDataset(
            base_dataset=base_dataset,
            added_params=dict(
                remove_isolated_cells=True,
                extend_pixels=True,
            ),
        )

        print(f"Dataset loaded with {len(maze_dataset)} mazes")

        x, y = maze_dataset[0]
        x = x / 255.0
        y = y / 255.0
        print(f"Sample maze shape: {x.shape}, solution shape: {y.shape}")
        print(
            f"Value ranges - maze: [{x.min():.4f}, {x.max():.4f}], solution: [{y.min():.4f}, {y.max():.4f}]"
        )

        return True
    except Exception as e:
        print(f"Error testing maze-dataset: {e}")
        return False


if __name__ == "__main__":
    print("\n===== Testing Maze Dataset Libraries =====\n")

    easy_to_hard_success = test_easy_to_hard_data()
    print("\n" + "-" * 50 + "\n")
    maze_dataset_success = test_maze_dataset()

    print("\n===== Test Results =====")
    print(f"easy-to-hard-data: {'SUCCESS' if easy_to_hard_success else 'FAILED'}")
    print(f"maze-dataset: {'SUCCESS' if maze_dataset_success else 'FAILED'}")

    if not maze_dataset_success:
        print("\nSuggestions for maze-dataset:")
        print("1. Try increasing Python's recursion limit:")
        print("   import sys; sys.setrecursionlimit(3000)")
        print("2. Check if your maze-dataset installation is correct")
        print("3. Try using a smaller maze size for testing")
