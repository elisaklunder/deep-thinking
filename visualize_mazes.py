import os
import torch
import numpy as np
from deepthinking.utils.plot import plot_maze_and_target
import matplotlib.pyplot as plt
import sys

# Create figures directory if it doesn't exist
os.makedirs("figures", exist_ok=True)

def visualize_easy_to_hard_mazes():
    """Load and visualize mazes from the easy-to-hard-data dataset"""
    try:
        from easy_to_hard_data import MazeDataset as EasyToHardMazeDataset
        print("Loading easy-to-hard mazes...")
        
        # Test different maze sizes
        for maze_size in [9, 11, 15, 19]:
            # Create dataset
            dataset = EasyToHardMazeDataset("./data", train=False, size=maze_size, download=True)
            print(f"Loaded {maze_size}x{maze_size} maze dataset with {len(dataset)} mazes")
            
            # Plot a few samples
            for i in range(min(3, len(dataset))):
                x, y = dataset[i]
                save_path = f"figures/easy_to_hard_maze_{maze_size}x{maze_size}_sample_{i}.png"
                plot_maze_and_target(x, y, save_str=save_path)
                print(f"Saved {save_path}")
                
        return True
    except Exception as e:
        print(f"Error visualizing easy-to-hard mazes: {e}")
        return False

def visualize_maze_dataset():
    """Load and visualize mazes from the maze-dataset library"""
    try:
        from maze_dataset import set_serialize_minimal_threshold
        from maze_dataset.dataset.rasterized import (
            MazeDataset, MazeDatasetConfig, RasterizedMazeDataset
        )
        from maze_dataset.generation import LatticeMazeGenerators
        
        # Increase recursion limit to handle maze generation
        sys.setrecursionlimit(15000)
        
        # Set threshold to prevent memory issues
        set_serialize_minimal_threshold(int(10**7))
        print("Loading maze-dataset mazes...")
        
        # Test different grid sizes and generation methods
        grid_sizes = [5, 7, 9]  # Will produce different maze sizes
        generators = [
            ("DFS", LatticeMazeGenerators.gen_dfs, {}),
            ("DFS with Percolation", LatticeMazeGenerators.gen_dfs_percolation, {"p": 0.2}),
            ("Percolation", LatticeMazeGenerators.gen_percolation, {"p": 0.6})
        ]
        
        for grid_n in grid_sizes:
            for gen_name, gen_func, gen_kwargs in generators:
                print(f"Generating {grid_n}x{grid_n} grid maze using {gen_name}")
                
                # Create dataset
                base_dataset = MazeDataset.from_config(
                    MazeDatasetConfig(
                        name=f"viz_{gen_name.lower().replace(' ', '_')}_{grid_n}",
                        grid_n=grid_n,
                        n_mazes=3,  # Just 3 samples
                        seed=42,
                        maze_ctor=gen_func,
                        maze_ctor_kwargs=gen_kwargs,
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
                
                # Get batch of mazes
                batch = maze_dataset.get_batch(idxs=None)
                
                # Process and plot each maze
                # Fix: Check if batch is numpy array and handle accordingly
                inputs = batch[0, :, :, :] / 255.0  # Normalize
                if isinstance(inputs, np.ndarray):
                    # Use numpy's transpose for numpy arrays
                    inputs = np.transpose(inputs, (0, 3, 1, 2))  # CHW format
                    # Convert to tensor to use with plot_maze_and_target
                    inputs = torch.from_numpy(inputs).float()
                else:
                    # Use permute for torch tensors
                    inputs = inputs.permute(0, 3, 1, 2)  # CHW format
                
                solutions = batch[1, :, :, :] / 255.0  # Normalize
                if isinstance(solutions, np.ndarray):
                    # Use numpy's transpose for numpy arrays
                    solutions = np.transpose(solutions, (0, 3, 1, 2))  # CHW format
                    # Convert to tensor
                    solutions = torch.from_numpy(solutions).float()
                    # Max along channel dimension
                    solutions = solutions.max(dim=1)[0]
                else:
                    # Use permute for torch tensors
                    solutions = solutions.permute(0, 3, 1, 2)  # CHW format
                    solutions, _ = torch.max(solutions, dim=1)  # Flatten channels
                
                # Plot each maze in the batch
                for i in range(len(inputs)):
                    save_path = f"figures/maze_dataset_{grid_n}_grid_{gen_name.lower().replace(' ', '_')}_sample_{i}.png"
                    plot_maze_and_target(inputs[i], solutions[i], save_str=save_path)
                    print(f"Saved {save_path}")
                
        return True
    except Exception as e:
        print(f"Error visualizing maze-dataset mazes: {e}")
        import traceback
        traceback.print_exc()  # Print full stack trace for better debugging
        return False

if __name__ == "__main__":
    print("\n===== Visualizing Maze Datasets =====\n")
    
    print("1. Easy-to-Hard Mazes")
    easy_to_hard_success = visualize_easy_to_hard_mazes()
    
    print("\n2. Maze-Dataset Library Mazes")
    maze_dataset_success = visualize_maze_dataset()
    
    print("\n===== Visualization Results =====")
    print(f"easy-to-hard-data visualization: {'SUCCESS' if easy_to_hard_success else 'FAILED'}")
    print(f"maze-dataset visualization: {'SUCCESS' if maze_dataset_success else 'FAILED'}")
    
    # Display the first few images with matplotlib
    print("\nTo view more mazes, check the 'figures' directory.")
