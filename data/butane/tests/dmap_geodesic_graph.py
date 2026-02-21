"""
Script to load diffusion map coordinates and construct geodesic graph.

This script:
1. Loads the diffusion map from butane_diff_map.npz
2. Extracts the first three coordinates
3. Computes euclidean kernel and geodesic graph using functions from kernel_construction.py
"""

import numpy as np
from kernel_construction import create_gaussian_kernel_euclidean, create_graph_kernel_geodesic
import os


def main():
    """Main function to load dmap and construct geodesic graph."""
    print("=" * 60)
    print("Loading diffusion map data...")
    print("=" * 60)
    
    # Load the diffusion map data
    fname = "butane_diff_map.npz"
    if not os.path.exists(fname):
        raise FileNotFoundError(f"File {fname} not found in current directory.")
    
    data = np.load(fname)
    print("Keys in data:")
    print(list(data.keys()))
    
    # Load the diffusion map coordinates
    dmap = data['dmap']
    print(f"Diffusion map shape: {dmap.shape}")
    
    # Extract first three coordinates
    dmap_3d = dmap[:, :3]
    print(f"Using first 3 coordinates, shape: {dmap_3d.shape}")
    
    print("\n" + "=" * 60)
    print("Computing euclidean kernel and distances...")
    print("=" * 60)
    
    # Set parameters
    epsilon = 0.1  # Kernel bandwidth
    
    # Compute euclidean kernel and squared distances using the function from kernel_construction
    K_euclidean, sqdists = create_gaussian_kernel_euclidean(dmap_3d, epsilon)
    print(f"Squared distances shape: {sqdists.shape}")
    print(f"Distance range: min={np.sqrt(sqdists[sqdists > 0].min()):.6f}, "
          f"max={np.sqrt(sqdists.max()):.6f}")
    print(f"Euclidean kernel shape: {K_euclidean.shape}")
    print(f"Euclidean kernel entries: min={K_euclidean.min():.6f}, max={K_euclidean.max():.6f}")
    
    print("\n" + "=" * 60)
    print("Constructing geodesic graph and kernel...")
    print("=" * 60)
    
    n_neighbors = min(50, dmap_3d.shape[0] - 1)  # Use 50 neighbors or all if less
    
    # Create geodesic kernel using the function from kernel_construction
    K_geodesic, geodesic_dists, graph_adj = create_graph_kernel_geodesic(
        sqdists, epsilon, n_neighbors=n_neighbors
    )
    
    print(f"Geodesic kernel shape: {K_geodesic.shape}")
    print(f"Geodesic kernel entries: min={K_geodesic.min():.6f}, max={K_geodesic.max():.6f}")
    print(f"Geodesic distances: min={geodesic_dists[geodesic_dists > 0].min():.6f}, "
          f"max={geodesic_dists.max():.6f}")
    
    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)
    
    np.savez('geodesic_data_dmap.npz',
             sqdists=sqdists,
             geodesic_dists=geodesic_dists)
    print("Saved results to geodesic_data_dmap.npz")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    
    return {
        'dmap_3d': dmap_3d,
        'sqdists': sqdists,
        'K_euclidean': K_euclidean,
        'K_geodesic': K_geodesic,
        'geodesic_dists': geodesic_dists,
        'graph_adj': graph_adj
    }


if __name__ == '__main__':
    results = main()
