"""
Butane carbon analysis script.

Loads butane_metad_deltanet.npz, extracts carbon-only data,
subsamples by every 2nd point (10K points), reshapes to 3D coordinates,
and computes a diffusion map based on pairwise alignment RMSD.
"""

from tqdm import tqdm
import numpy as np
import os
import sys
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from diffusion_map import DiffusionMap


def load_carbon_data(data_path='data/butane/butane_metad_deltanet.npz', subsample_step=2):
    """
    Load butane carbon-only data, subsample, and reshape to 3D coordinates.
    
    Parameters
    ----------
    data_path : str
        Path to the butane_metad_deltanet.npz file.
    
    Returns
    -------
    xyz_coords : array, shape (10000, 4, 3)
        Subsampled (every 2nd point) and reshaped coordinate matrix.
        10K frames, 4 carbon atoms, 3 coordinates (x, y, z) each.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found.")

    data_file = np.load(data_path)
    print("Keys in data file:")
    print(list(data_file.keys()))

    # Extract carbon-only data from 'data' key
    data = data_file['data_all_atom']
    print(f"Original data shape: {data.shape}")

    # Subsample by every 2nd point (resulting in 10K points)
    data_subsampled = data[::subsample_step]
    print(f"Subsampled data shape (every {subsample_step}th point): {data_subsampled.shape}")

    # Reshape 12-dim vector to (n_frames, -1, 3) - four 3D xyz coordinates
    n_frames = data_subsampled.shape[0]
    xyz_coords = data_subsampled.reshape(n_frames, -1, 3)
    print(f"Reshaped to xyz coordinates: {xyz_coords.shape}")
    print(f"  - {n_frames} frames")
    print(f"  - 4 carbon atoms per frame")
    print(f"  - 3 coordinates (x, y, z) per atom")
    
    return xyz_coords


# Load the data
print("="*60)
print("Loading butane carbon data")
print("="*60)
xyz_coords = load_carbon_data(subsample_step=20) # Subsample every 20th point to get 1K frames

print("\nData ready for further analysis.")
breakpoint()


def center_structure(X):
    """Center structure by removing the centroid."""
    return X - np.mean(X, axis=0)


def kabsch_rmsd(X, Y):
    """
    Compute RMSD after optimal alignment using Kabsch algorithm.
    
    Parameters
    ----------
    X : array, shape (n_atoms, 3)
        Reference structure (centered).
    Y : array, shape (n_atoms, 3)
        Structure to align (centered).
    
    Returns
    -------
    rmsd : float
        RMSD after optimal rigid-body alignment.
    """
    # Center structures
    X = center_structure(X)
    Y = center_structure(Y)
    
    # Compute optimal rotation via SVD
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Align Y to X and compute RMSD
    Y_aligned = Y @ R.T
    rmsd = np.sqrt(np.mean(np.sum((X - Y_aligned)**2, axis=1)))
    
    return rmsd


def compute_pairwise_alignment_rmsd(xyz_coords):
    """
    Compute pairwise RMSD between all configurations after optimal alignment.
    
    Parameters
    ----------
    xyz_coords : array, shape (n_frames, n_atoms, 3)
        Coordinate array.
    
    Returns
    -------
    rmsd_matrix : array, shape (n_frames, n_frames)
        Symmetric matrix of pairwise RMSD values.
    """
    n_frames = xyz_coords.shape[0]
    rmsd_matrix = np.zeros((n_frames, n_frames))
    
    print(f"Computing pairwise alignment RMSD ({n_frames} frames)...")
    for i in tqdm(range(n_frames)):
        # if i % 1000 == 0:
        #     print(f"  {i}/{n_frames}")
        for j in range(i, n_frames):
            rmsd = kabsch_rmsd(xyz_coords[i], xyz_coords[j])
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd
    
    print(f"RMSD matrix shape: {rmsd_matrix.shape}")
    print(f"RMSD range: [{rmsd_matrix.min():.6e}, {rmsd_matrix.max():.6e}]")
    
    return rmsd_matrix


def run_ksum_test(sq_dists_sparse, dummy_data):
    """
    Run k-sum test to find optimal epsilon for diffusion map kernel.
    
    Parameters
    ----------
    sq_dists_sparse : sparse matrix, shape (n_samples, n_samples)
        Precomputed squared distance matrix (sparse).
    dummy_data : array, shape (n_samples, 1)
        Dummy data array (used by ksums method).
    
    Returns
    -------
    optimal_epsilon : float
        Optimal epsilon selected by maximum derivative method.
    eps_range : array
        Range of epsilon values tested.
    kernel_sums : array
        Kernel sum for each epsilon value.
    """
    print("\n" + "="*60)
    print("Running k-sum test to find optimal epsilon")
    print("="*60)
    
    # Create a temporary DiffusionMap object for the k-sum test
    dmap_test = DiffusionMap(epsilon=0.1, num_evecs=10)
    dmap_test.sq_dists = sq_dists_sparse
    dmap_test.flag = True
    
    # Perform k-sum test
    eps_range, kernel_sums = dmap_test.ksums(dummy_data)
    print(f"K-sum test completed for {len(eps_range)} epsilon values")
    print(f"Epsilon range: [{eps_range.min():.6e}, {eps_range.max():.6e}]")
    
    # Find optimal epsilon using maximum derivative
    optimal_eps, max_deriv_idx, max_deriv_val, _, discrete_deriv = dmap_test.max_derivative(eps_range, kernel_sums)
    
    print(f"\nOptimal epsilon (max derivative): {optimal_eps:.6e}")
    print(f"Maximum discrete derivative: {max_deriv_val:.6e}")
    
    return optimal_eps, eps_range, kernel_sums


def compute_alignment_diffusion_map(xyz_coords, num_evecs=10, epsilon='ksum'):
    """
    Compute diffusion map using alignment-based kernel (pairwise RMSD).
    
    Parameters
    ----------
    xyz_coords : array, shape (n_frames, n_atoms, 3)
        Coordinate data.
    num_evecs : int
        Number of eigenvectors to compute.
    epsilon : str or float
        Diffusion map bandwidth parameter. If 'ksum', run k-sum test first.
    
    Returns
    -------
    dmap : DiffusionMap
        Fitted diffusion map object.
    rmsd_matrix : array
        Pairwise RMSD distance matrix used for kernel.
    """
    # Compute pairwise alignment RMSD
    rmsd_matrix = compute_pairwise_alignment_rmsd(xyz_coords)
    
    # Square RMSD to get squared distances
    sq_dists = rmsd_matrix ** 2
    
    # Convert to sparse CSR matrix
    sq_dists_sparse = csr_matrix(sq_dists)
    breakpoint()
    
    # Run k-sum test if epsilon='ksum', otherwise use provided epsilon
    if epsilon == 'ksum':
        dummy_data = np.zeros((xyz_coords.shape[0], 1))
        optimal_eps, eps_range, kernel_sums = run_ksum_test(sq_dists_sparse, dummy_data)
        epsilon = optimal_eps
    
    # Initialize DiffusionMap and set precomputed distances
    print(f"\nInitializing DiffusionMap with {num_evecs} eigenvectors and epsilon={epsilon:.6e}...")
    dmap = DiffusionMap(epsilon=epsilon, num_evecs=num_evecs)
    dmap.sq_dists = sq_dists_sparse
    dmap.flag = True  # Signal that distances are precomputed
    
    # Fit using dummy data (distances already computed)
    dummy_data = np.zeros((xyz_coords.shape[0], 1))
    dmap.fit(dummy_data)
    
    print(f"Diffusion map computed.")
    print(f"Eigenvalues shape: {dmap.evals.shape}")
    print(f"Eigenvectors shape: {dmap.evecs.shape}")
    print(f"Diffusion map coordinates shape: {dmap.dmap.shape}")
    breakpoint()
    
    return dmap, rmsd_matrix


# Compute diffusion map on alignment-based kernel
print("\n" + "="*60)
print("Computing diffusion map with alignment-based kernel")
print("="*60)
dmap, rmsd_matrix = compute_alignment_diffusion_map(xyz_coords, num_evecs=10, epsilon='ksum')

print("\nAnalysis complete.")

