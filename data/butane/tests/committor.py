"""
Committor analysis on butane data.

This script:
1. Loads data from butane_data_with_nets.npz (already subsampled by 10)
2. Subsamples using delta_net_idx
3. Computes epsilon as largest distance to k-th nearest neighbor (k = 10% of data size)
4. Computes committor using target measure diffusion map (from potential)
5. Visualizes committor over dihedral angles
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from diffusion_map import TargetMeasureDiffusionMap


def section_print(message):
    """Print a section header with decorative lines."""
    print("\n" + "=" * 60)
    print(message)
    print("=" * 60)


def compute_adaptive_epsilon(sq_dists, k_percent=0.1):
    """
    Compute epsilon as the largest distance to the k-th nearest neighbor.
    
    Parameters:
    -----------
    sq_dists : array, shape (n_samples, n_samples)
        Squared pairwise distance matrix
    k_percent : float
        Percentage of data size to use for k (default 0.1 = 10%)
        
    Returns:
    --------
    epsilon : float
        Largest distance to k-th nearest neighbor across all points
    k : int
        Number of neighbors used
    """
    n_samples = sq_dists.shape[0]
    k = int(k_percent * n_samples)
    k = max(1, min(k, n_samples - 1))  # Ensure k is valid
    
    # Compute regular distances from squared distances
    dists = np.sqrt(sq_dists)
    
    # Use NearestNeighbors with precomputed distances
    # We need k+1 neighbors because the first one is always the point itself
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='precomputed')
    nbrs.fit(dists)
    
    # Get distances to k+1 nearest neighbors (including self)
    knn_distances, knn_indices = nbrs.kneighbors(dists)
    
    # The k-th nearest neighbor distance is at index k (0-indexed, first is self at 0)
    kth_distances = knn_distances[:, k]
    
    # Use the LARGEST k-th distance as epsilon
    epsilon = np.max(kth_distances)
    
    print(f"Using k = {k} neighbors ({k_percent*100}% of {n_samples} points)")
    print(f"Epsilon (max distance to {k}-th neighbor): {epsilon:.6f}")
    print(f"Mean distance to {k}-th neighbor: {np.mean(kth_distances):.6f}")
    print(f"Min distance to {k}-th neighbor: {np.min(kth_distances):.6f}")
    
    return epsilon, k


def main():
    """Main function to compute and visualize committor."""
    section_print("Loading butane data with nets...")
    
    # Load data
    if not os.path.exists('butane_data_with_nets.npz'):
        raise FileNotFoundError("butane_data_with_nets.npz not found.")
    
    data_file = np.load('butane_data_with_nets.npz')
    print("Keys in data file:")
    print(list(data_file.keys()))
    
    # Load required arrays
    data = data_file['data']
    potential = data_file['potential'] - min(data_file['potential'])  # Shift potential to be non-negative
    sqdists = data_file['sqdists']
    delta_net_idx = data_file['delta_idx_dmap']
    
    print(f"Original data shape: {data.shape}")
    print(f"Original potential shape: {potential.shape}")
    print(f"Original sqdists shape: {sqdists.shape}")
    print(f"Delta net indices shape: {delta_net_idx.shape}")
    breakpoint()
    section_print("Subsampling using delta_net_idx...")
    
    # Subsample using delta net indices
    data_deltanet = data[delta_net_idx, :]
    potential_deltanet = potential[delta_net_idx]
    sqdists_deltanet = sqdists[delta_net_idx, :][:, delta_net_idx]
    
    print(f"Delta net data shape: {data_deltanet.shape}")
    print(f"Delta net potential shape: {potential_deltanet.shape}")
    print(f"Delta net sqdists shape: {sqdists_deltanet.shape}")
    breakpoint()
    section_print("Computing adaptive epsilon...")
    
    # Compute epsilon as largest distance to k-th nearest neighbor (k = 10% of data size)
    epsilon, k_neighbors = compute_adaptive_epsilon(sqdists_deltanet, k_percent=0.1)
    breakpoint()
    section_print("Converting potential to target measure...")
    
    # Load kbT if available, otherwise use default

    kbT = data_file['kbT']
    print(f"Using kbT from file: {kbT}")
    
    # Convert potential to target measure
    target_measure = np.exp(-potential_deltanet / kbT)
    print(f"Target measure shape: {target_measure.shape}")
    print(f"Target measure range: [{target_measure.min():.6e}, {target_measure.max():.6e}]")
    breakpoint()
    section_print("Constructing diffusion map and generator...")
    
    # Initialize diffusion map object
    # Use alpha=0.5 for target measure normalization
    dmap = TargetMeasureDiffusionMap(epsilon=epsilon, num_evecs=10, target_measure=target_measure)
    
    # Set the precomputed squared distances and flag
    import scipy.sparse as sps
    dmap.sq_dists = sps.csr_matrix(sqdists_deltanet)
    dmap.flag = True
    
    # Set the target measure as density
    dmap.density = target_measure
    breakpoint()
    # Construct generator without recomputing distances
    print("Constructing generator from precomputed distances...")
    dmap.construct_generator(data_deltanet)
    
    L = dmap.get_generator()
    print(f"Generator shape: {L.shape}")
    breakpoint()
    section_print("Defining reactant and product sets...")
    
    # Define reactant (B) and product (A) sets based on dihedral angles
    # Assuming data contains dihedral angles or we need to extract them
    # For butane, typically we have central dihedral angle
    
    # Check if dihedrals are in the file
    if 'dihedrals' in data_file:
        dihedrals_full = data_file['dihedrals']
        dihedrals_deltanet = dihedrals_full[delta_net_idx]
    else:
        # Assume data are the dihedrals
        print("Note: 'dihedrals' key not found, using data as dihedrals")
        dihedrals_deltanet = data_deltanet.flatten()
    
    print(f"Dihedrals shape: {dihedrals_deltanet.shape}")
    print(f"Dihedral range: [{dihedrals_deltanet.min():.4f}, {dihedrals_deltanet.max():.4f}]")
    
    # Define sets based on dihedral angles
    # Reactant B: angles around -pi (gauche-)
    # Product A: angles around +pi (gauche+)
    # Transition region C: everything else
    
    threshold_B = 0.2  # Define regions within ±threshold from ±pi
    threshold_A = 0.1
    B_bool = np.abs(dihedrals_deltanet-np.pi) < threshold_B
    A_bool_left = np.abs(dihedrals_deltanet-np.pi/3) < threshold_A
    A_bool_right = np.abs(dihedrals_deltanet-5*np.pi/3) < threshold_A
    A_bool = np.logical_or(A_bool_left, A_bool_right)
    C_bool = np.logical_not(np.logical_or(A_bool, B_bool))  
    
    print(f"Reactant set B size: {B_bool.sum()}")
    print(f"Product set A size: {A_bool.sum()}")
    print(f"Transition region C size: {C_bool.sum()}")
    breakpoint()
    section_print("Computing committor function...")
    
    # Compute committor using the static method from TargetMeasureDiffusionMap
    q = dmap.construct_committor(L, B_bool, C_bool)
    
    print(f"Committor shape: {q.shape}")
    print(f"Committor range: [{q.min():.4f}, {q.max():.4f}]")
    print(f"Committor at B (should be ~1): {q[B_bool].mean():.4f}")
    print(f"Committor at A (should be ~0): {q[A_bool].mean():.4f}")
    breakpoint()
    section_print("Visualizing committor...")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Committor vs dihedral angle
    scatter = axes[0].scatter(dihedrals_deltanet, q, c=q, cmap='RdYlBu_r', 
                             s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
    axes[0].axhline(y=0, color='blue', linestyle='--', linewidth=2, label='Product A (q=0)')
    axes[0].axhline(y=1, color='red', linestyle='--', linewidth=2, label='Reactant B (q=1)')
    axes[0].set_xlabel('Dihedral angle (rad)', fontsize=12)
    axes[0].set_ylabel('Committor q', fontsize=12)
    axes[0].set_title('Committor Function vs Dihedral Angle', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label='Committor value')
    
    # Plot 2: Histogram of committor values
    axes[1].hist(q, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(x=0, color='blue', linestyle='--', linewidth=2, label='Product A')
    axes[1].axvline(x=1, color='red', linestyle='--', linewidth=2, label='Reactant B')
    axes[1].set_xlabel('Committor value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Committor Values', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('committor_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved plot to committor_analysis.png")
    plt.close()
    breakpoint()
    section_print("Saving results...")
    
    # Save results
    np.savez('committor_results.npz',
             committor=q,
             dihedrals_deltanet=dihedrals_deltanet,
             data_deltanet=data_deltanet,
             potential_deltanet=potential_deltanet,
             epsilon=epsilon,
             k_neighbors=k_neighbors,
             B_bool=B_bool,
             A_bool=A_bool,
             C_bool=C_bool,
             kbT=kbT)
    print("Saved results to committor_results.npz")
    
    section_print("Done!")
    
    return {
        'committor': q,
        'dihedrals_deltanet': dihedrals_deltanet,
        'epsilon': epsilon,
        'generator': L
    }


if __name__ == '__main__':
    results = main()
