"""
Script to compute delta nets on geodesic distances.

This script:
1. Loads geodesic distances from geodesic_data_euclidean.npz and geodesic_data_dmap.npz
2. Computes delta nets with delta tuned to the distance to the 50th nearest neighbor
3. Loads original butane data and dihedrals, subsamples by factor of 10
4. Extracts delta net indices and corresponding dihedrals
5. Visualizes the dihedrals corresponding to delta net points
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os
import time


def section_print(message):
    """Print a section header with decorative lines."""
    print("\n" + "=" * 60)
    print(message)
    print("=" * 60)


def compute_delta_net(distances, delta):
    """
    Compute delta net from a distance matrix.
    
    Parameters:
    -----------
    distances : array, shape (n_samples, n_samples)
        Pairwise distance matrix
    delta : float
        Delta threshold for delta net
        
    Returns:
    --------
    delta_idx : array
        Indices of points in the delta net
    """
    n_samples = distances.shape[0]
    delta_idx = []
    remaining = set(range(n_samples))
    
    while remaining:
        # Pick the first remaining point
        center = min(remaining)
        delta_idx.append(center)
        
        # Remove all points within delta of this center
        to_remove = set()
        for j in remaining:
            if distances[center, j] <= delta:
                to_remove.add(j)
        
        remaining -= to_remove
    
    return np.array(delta_idx)


def compute_adaptive_delta(distances, k=50):
    """
    Compute adaptive delta as the distance to the k-th nearest neighbor.
    Returns the minimum such distance across all points.
    
    Parameters:
    -----------
    distances : array, shape (n_samples, n_samples)
        Pairwise distance matrix
    k : int
        Number of nearest neighbors to consider
        
    Returns:
    --------
    delta : float
        Adaptive delta value
    """
    n_samples = distances.shape[0]
    k = min(k, n_samples - 1)
    
    # Use NearestNeighbors with precomputed distances
    # We need k+1 neighbors because the first one is always the point itself
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='precomputed')
    nbrs.fit(distances)
    
    # Get distances to k+1 nearest neighbors (including self)
    knn_distances, knn_indices = nbrs.kneighbors(distances)
    
    # The k-th nearest neighbor distance is at index k (0-indexed, first is self at 0)
    kth_distances = knn_distances[:, k]
    
    # Use the minimum k-th distance as delta
    delta = np.min(kth_distances)
    print(f"Adaptive delta (min distance to {k}-th neighbor): {delta:.6f}")
    print(f"Mean distance to {k}-th neighbor: {np.mean(kth_distances):.6f}")
    print(f"Max distance to {k}-th neighbor: {np.max(kth_distances):.6f}")
    
    return delta


def main():
    """Main function to compute delta nets on geodesic distances."""
    section_print("Loading geodesic distance data...")
    
    # Load geodesic distances from euclidean data
    if not os.path.exists('geodesic_data_euclidean.npz'):
        raise FileNotFoundError("geodesic_data_euclidean.npz not found. Run kernel_construction.py first.")
    
    euclidean_data = np.load('geodesic_data_euclidean.npz')
    geodesic_dists_euclidean = euclidean_data['geodesic_dists']
    print(f"Euclidean geodesic distances shape: {geodesic_dists_euclidean.shape}")
    
    # Load geodesic distances from dmap data
    if not os.path.exists('geodesic_data_dmap.npz'):
        raise FileNotFoundError("geodesic_data_dmap.npz not found. Run dmap_geodesic_graph.py first.")
    
    dmap_data = np.load('geodesic_data_dmap.npz')
    geodesic_dists_dmap = dmap_data['geodesic_dists']
    print(f"Dmap geodesic distances shape: {geodesic_dists_dmap.shape}")
    # breakpoint()
    section_print("Computing adaptive delta values...")
    
    # Set k as 2.5% of the data size
    k_neighbors = max(2,int(0.01 * geodesic_dists_euclidean.shape[0]))
    print(f"Using k = {k_neighbors} neighbors (2.5% of {geodesic_dists_euclidean.shape[0]} points)")
    
    # Compute adaptive delta for euclidean geodesic distances
    print("\nFor Euclidean geodesic distances:")
    start_time = time.time()
    delta_euclidean = compute_adaptive_delta(geodesic_dists_euclidean, k=k_neighbors)
    time_delta_euclidean = time.time() - start_time
    print(f"Time to compute delta: {time_delta_euclidean:.4f} seconds")
    # breakpoint()
    # Compute adaptive delta for dmap geodesic distances
    print("\nFor Dmap geodesic distances:")
    start_time = time.time()
    delta_dmap = compute_adaptive_delta(geodesic_dists_dmap, k=k_neighbors)
    time_delta_dmap = time.time() - start_time
    print(f"Time to compute delta: {time_delta_dmap:.4f} seconds")
    # breakpoint()    
    section_print("Computing delta nets...")
    
    # Compute delta nets
    print("\nComputing delta net on Euclidean geodesic distances...")
    start_time = time.time()
    delta_idx_euclidean = compute_delta_net(geodesic_dists_euclidean, delta_euclidean)
    time_deltanet_euclidean = time.time() - start_time
    print(f"Delta net size (Euclidean): {len(delta_idx_euclidean)} / {geodesic_dists_euclidean.shape[0]}")
    print(f"Compression ratio: {len(delta_idx_euclidean) / geodesic_dists_euclidean.shape[0]:.4f}")
    print(f"Time to compute delta net: {time_deltanet_euclidean:.4f} seconds")
    # breakpoint()

    print("\nComputing delta net on Dmap geodesic distances...")
    start_time = time.time()
    delta_idx_dmap = compute_delta_net(geodesic_dists_dmap, delta_dmap)
    time_deltanet_dmap = time.time() - start_time
    print(f"Delta net size (Dmap): {len(delta_idx_dmap)} / {geodesic_dists_dmap.shape[0]}")
    print(f"Compression ratio: {len(delta_idx_dmap) / geodesic_dists_dmap.shape[0]:.4f}")
    print(f"Time to compute delta net: {time_deltanet_dmap:.4f} seconds")
    # breakpoint()
    section_print("Loading original butane data...")
    
    # Load butane data
    cwd = os.getcwd()
    fname = os.path.join(cwd, "data", "butane", "butane_metad.npz")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"File {fname} not found.")
    
    butane_data = np.load(fname)
    print("Keys in butane data:")
    print(list(butane_data.keys()))
    
    # Get data and dihedrals
    data = butane_data["data"]
    print(f"Original data shape: {data.shape}")
    
    # Check if dihedrals exist in the data
    if "dihedrals" in butane_data:
        dihedrals = butane_data["dihedrals"]
    else:
        # If dihedrals not in file, assume data are the dihedrals
        print("Note: 'dihedrals' key not found, using 'data' as dihedrals")
        dihedrals = data
    
    print(f"Original dihedrals shape: {dihedrals.shape}")
    
    # Subsample by factor of 10 (consistent with kernel_construction.py)
    data_subsampled = data[::10, :]
    dihedrals_subsampled = dihedrals[::10]
    print(f"Subsampled data shape: {data_subsampled.shape}")
    print(f"Subsampled dihedrals shape: {dihedrals_subsampled.shape}")
    
    # Verify shapes match
    if data_subsampled.shape[0] != geodesic_dists_euclidean.shape[0]:
        print(f"Warning: Subsampled data size ({data_subsampled.shape[0]}) != "
              f"geodesic distances size ({geodesic_dists_euclidean.shape[0]})")
    # breakpoint()
    section_print("Extracting delta net dihedrals...")
    
    # Get dihedrals corresponding to delta net indices
    dihedrals_deltanet_euclidean = dihedrals_subsampled[delta_idx_euclidean, None]
    dihedrals_deltanet_dmap = dihedrals_subsampled[delta_idx_dmap, None]
    
    print(f"Delta net dihedrals (Euclidean): {dihedrals_deltanet_euclidean.shape}")
    print(f"Delta net dihedrals (Dmap): {dihedrals_deltanet_dmap.shape}")
    # breakpoint()
    section_print("Visualizing dihedrals...")
    
    # Visualize dihedrals as histograms (1D data)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot overlaid histograms
    ax.hist(dihedrals_subsampled, bins=50, alpha=0.3, color='gray', 
            label=f'All subsampled (n={len(dihedrals_subsampled)})', density=True)
    ax.hist(dihedrals_deltanet_euclidean, bins=30, alpha=0.5, color='red', 
            label=f'Euclidean geodesic delta net (n={len(delta_idx_euclidean)})', density=True)
    ax.hist(dihedrals_deltanet_dmap, bins=30, alpha=0.5, color='blue', 
            label=f'Dmap geodesic delta net (n={len(delta_idx_dmap)})', density=True)
    
    ax.set_xlabel('Dihedral angle (rad)', fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)
    ax.set_title('Comparison of Dihedral Distributions', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('deltanets_geodesic_dihedrals.png', dpi=300, bbox_inches='tight')
    print("Saved plot to deltanets_geodesic_dihedrals.png")
    plt.close()
    # breakpoint()
    section_print("Saving results...")
    
    # Save delta net indices and results
    np.savez('butane_data_with_nets.npz',
             data=data_subsampled,
             dihedrals=dihedrals_subsampled,
             delta_idx_euclidean=delta_idx_euclidean,
             delta_idx_dmap=delta_idx_dmap,
             delta_euclidean=delta_euclidean,
             delta_dmap=delta_dmap,
             sqdists=euclidean_data['sqdists'],
             potential=butane_data['potential'][::10],
             kbT=butane_data['kbT'])
    print("Saved results to deltanets_geodesic_results.npz")
    
    # Print timing summary
    print("\n" + "-" * 60)
    print("TIMING SUMMARY")
    print("-" * 60)
    print(f"Euclidean Geodesic:")
    print(f"  - Delta computation:    {time_delta_euclidean:.4f} seconds")
    print(f"  - Delta net computation: {time_deltanet_euclidean:.4f} seconds")
    print(f"  - Total:                 {time_delta_euclidean + time_deltanet_euclidean:.4f} seconds")
    print(f"\nDmap Geodesic:")
    print(f"  - Delta computation:    {time_delta_dmap:.4f} seconds")
    print(f"  - Delta net computation: {time_deltanet_dmap:.4f} seconds")
    print(f"  - Total:                 {time_delta_dmap + time_deltanet_dmap:.4f} seconds")
    print("-" * 60)
    
    section_print("Done!")


if __name__ == '__main__':
    results = main()
