"""
Committor analysis on butane all-atom data.

This script:
1. Loads all-atom data from butane_metad_nonaligned.npz (data_all_atom key)
2. Subsamples by every 10th point to reduce size
3. Converts to mdtraj trajectory and aligns structures
4. Computes epsilon as largest distance to k-th nearest neighbor (k = 10% of data size)
5. Computes committor using target measure diffusion map (from potential)
6. Visualizes committor over dihedral angles
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import sys
import os
import mdtraj as md

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from helpers import compute_squared_distances, align_trajectory, section_print
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


def align_trajectory(xyz_data):
    """
    Convert xyz coordinates to mdtraj trajectory and align all frames.
    
    Parameters:
    -----------
    xyz_data : array, shape (n_frames, n_atoms, 3)
        XYZ coordinates in nanometers
        
    Returns:
    --------
    aligned_xyz : array, shape (n_frames, n_atoms, 3)
        Aligned XYZ coordinates
    """
    section_print("Creating mdtraj trajectory and aligning...")
    
    # Load topology from PDB file
    pdb = md.load('butane.pdb')
    topology = pdb.topology
    
    # Create trajectory with topology
    traj = md.Trajectory(xyz_data, topology)
    print(f"Created trajectory with {traj.n_frames} frames and {traj.n_atoms} atoms")
    
    # Align trajectory to first frame
    traj.superpose(traj, frame=0)
    print("Aligned trajectory")
    
    return traj.xyz


def get_metastable_states(dihedrals, threshold_B=0.2, threshold_A=0.1):
    """
    Define reactant (B), product (A), and transition (C) sets based on dihedral angles.
    
    Parameters:
    -----------
    dihedrals : array, shape (n_samples,)
        Dihedral angles in radians
    threshold_B : float
        Threshold around pi for reactant set B
    threshold_A : float
        Threshold around pi/3 and 5pi/3 for product set A
        
    Returns:
    --------
    B_bool : array, shape (n_samples,)
        Boolean array for reactant set B
    A_bool : array, shape (n_samples,)
        Boolean array for product set A
    C_bool : array, shape (n_samples,)
        Boolean array for transition region C
    """
    B_bool = np.abs(dihedrals - np.pi) < threshold_B
    A_bool_left = np.abs(dihedrals - np.pi/3) < threshold_A
    A_bool_right = np.abs(dihedrals - 5*np.pi/3) < threshold_A
    A_bool = np.logical_or(A_bool_left, A_bool_right)
    C_bool = np.logical_not(np.logical_or(A_bool, B_bool))
    return B_bool, A_bool, C_bool

def main():
    """Main function to compute and visualize committor."""
    section_print("Loading butane all-atom data...")
    
    # Load data
    if not os.path.exists('butane_metad_nonaligned.npz'):
        raise FileNotFoundError("butane_metad_nonaligned.npz not found.")
    
    data_file = np.load('butane_metad_nonaligned.npz')
    print("Keys in data file:")
    print(list(data_file.keys()))
    
    # Load and subsample by every 10th point
    data = data_file['data_all_atom'][::10]
    potential = (data_file['potential'] - min(data_file['potential']))[::10]  # Shift to be non-negative
    breakpoint()
    print(f"Data shape (flat): {data.shape}")
    print(f"Potential shape: {potential.shape}")
    
    # Reshape data to (n_frames, n_atoms, 3)
    n_frames = data.shape[0]
    n_atoms = data.shape[1] // 3
    data = data.reshape(n_frames, n_atoms, 3)
    print(f"Data shape (reshaped): {data.shape}")
    breakpoint()
    # Align trajectory
    aligned_xyz = align_trajectory(data)
    
    # Flatten aligned coordinates for distance computation
    # Shape: (n_frames, n_atoms * 3)
    data_flat = aligned_xyz.reshape(aligned_xyz.shape[0], -1)
    print(f"Flattened aligned data shape: {data_flat.shape}")
    breakpoint()
    # Compute squared distances
    sqdists = compute_squared_distances(data_flat)
    breakpoint()
    section_print("Computing adaptive epsilon...")
    
   

if __name__ == '__main__':
    results = main()
