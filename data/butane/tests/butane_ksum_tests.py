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
from diffusion_map import TargetMeasureDiffusionMap
from helpers import epsilon_net
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import csr_matrix


def section_print(message):
    """Print a section header with decorative lines."""
    print("\n" + "=" * 60)
    print(message)
    print("=" * 60)

def create_laplacian_dense(data, target_measure, epsilon):

    num_features = data.shape[1]
    num_samples = data.shape[0]

    ### Create distance matrix
    sqdists = cdist(data, data, 'sqeuclidean') 
    
    ### Create Kernel
    K = np.exp(-sqdists / (2.0*epsilon))

    ### Create Graph Laplacian
    kde = K.sum(axis=1)
    u = (target_measure**(0.5)) / kde
    U = np.diag(u)
    W = U @ K @ U
    stationary = W.sum(axis=1)
    P = np.diag(stationary**(-1)) @ W 
    L = (P - np.eye(num_samples))/epsilon

    return [stationary, K, L]

def solve_committor_dense(L, B, C, num_samples):

    Lcb = L[C, :][:, B]
    Lcc = L[C, :][:, C]

    q = np.zeros(num_samples)
    q[B] = 1
    row_sum = Lcb.sum(axis=1).ravel()
    q[C] = np.linalg.solve(Lcc, -row_sum)
    return q

def kabsch_algorithm(P,Q):
    """Compute aligned distance between x,y 
    x: numpy array
    y: numpy array 
    returns the pairwise euclidean distance between x,y after aligning x to y. 
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Optimal translation
    t = centroid_Q - centroid_P

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = np.dot(p.T, q)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Validate right-handed coordinate system
    if np.linalg.det(np.dot(Vt.T, U.T)) < 0.0:
        Vt[-1, :] *= -1.0

    # Optimal rotation
    R = np.dot(Vt.T, U.T)

    # RMSD
    rmsd = np.sqrt(np.sum(np.square(np.dot(p, R.T) - q)) / P.shape[0])

    return R, t, rmsd

def aligned_distance(x, y):
    N = x.shape[0] // 3
    P = x.reshape((N, 3))
    Q = y.reshape((N, 3))
    _, _, rmsd = kabsch_algorithm(P, Q)
    return rmsd

def gram_matrix(X):
    """Compute the Gram matrix (inner products) of data X.
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_atoms, n_coordinates)
        Input data
        
    Returns:
    --------
    G : array, shape (n_samples, n_atoms, n_atoms)
        Gram matrix
    """
    recentered_data = X - np.mean(X, axis=1, keepdims=True)
    G = np.einsum('ijk,ilk->ijl', recentered_data, recentered_data)
    return G
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

def run_ksum_delta_tests(data_flat, deltas=None, outdir='ksum_test_plots', num_evecs=10, target_measure=None):
    """Run k-sum tests over a range of delta epsilon-nets, save plots.

    Parameters
    ----------
    data_flat : array, shape (n_samples, n_features)
        Flattened coordinate data (rows = samples).
    deltas : iterable or None
        Sequence of delta values to use for epsilon-net (defaults 0.1..1.0 step 0.1).
    outdir : str
        Directory to save plots.
    num_evecs : int
        Number of eigenvectors passed to diffusion map (used to construct object).
    target_measure : array-like or None
        Optional target measure to pass to `TargetMeasureDiffusionMap`.

    Returns
    -------
    results : dict
        Mapping delta -> dict with keys `eps_range`, `kernel_sums`, `max_eps`, `net_idx`.
    """
    if deltas is None:
        deltas = np.arange(0.1, 1.01, 0.1)

    # Precompute full pairwise squared-distance matrix once (rows = samples)
    section_print("Precomputing full squared-distance matrix for k-sum tests...")
    full_sq = squareform(pdist(data_flat, 'sqeuclidean'))

    # Make output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    results = {}

    section_print("Running k-sum test over delta nets and saving plots...")
    # helpers.epsilon_net expects data in shape (features, samples)
    for delta in deltas:
        delta = float(np.round(delta, 2))
        section_print(f"Delta = {delta}")

        net_idx, _ = epsilon_net(data_flat.T, delta)
        net_idx = np.array(net_idx, dtype=int)

        # Subsample squared-distance matrix and make sparse
        subsq = full_sq[np.ix_(net_idx, net_idx)]
        subsq_csr = csr_matrix(subsq)

        # Initialize diffusion map with optional target measure
        if target_measure is not None:
            dmap = TargetMeasureDiffusionMap(epsilon=1e-1, num_evecs=num_evecs, target_measure=target_measure)
        else:
            dmap = TargetMeasureDiffusionMap(epsilon=1e-1, num_evecs=num_evecs)
        dmap.sq_dists = subsq_csr
        dmap.flag = True

        # Run k-sum test on the subsampled data (rows = samples)
        subsampled_data = data_flat[net_idx, :]
        eps_range, kernel_sums = dmap.ksums(subsampled_data)

        # Compute optimal epsilon (max derivative)
        max_eps, _, _, _, _ = dmap.max_derivative(eps_range, kernel_sums)

        # Plot using Axes object
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(eps_range, np.log2(kernel_sums), '-o', ms=3)
        ax.set_xscale('log')
        ax.set_xlabel('epsilon')
        ax.set_ylabel('log2(kernel sums)')
        ax.set_title(f'ksum test, delta={delta:.2f}, net size={len(net_idx)}')
        ax.grid(True, which='both')
        ax.axvline(max_eps, color='r', linestyle='--', label=f'max deriv eps={max_eps:.3e}')
        ax.legend()
        fig.tight_layout()
        fname = os.path.join(outdir, f"ksum_delta_{int(delta*10):02d}_n{len(net_idx)}.png")
        fig.savefig(fname, dpi=300)
        plt.close(fig)
        print(f"Saved: {fname}")

def main():
    """Load and prepare data for analysis.

    Returns
    -------
    data_flat : array
        Flattened, aligned coordinate array (rows = samples).
    target_measure : array
        Target measure computed from potential.
    data_file : numpy.lib.npyio.NpzFile
        Loaded npz file (for additional metadata if needed).
    """
    section_print("Loading butane all-atom data...")

    # Load data
    if not os.path.exists('butane_metad_nonaligned.npz'):
        raise FileNotFoundError("butane_metad_nonaligned.npz not found.")

    data_file = np.load('butane_metad_nonaligned.npz')
    print("Keys in data file:")
    print(list(data_file.keys()))

    # Subsample by every 10th point (same behavior as previous script)
    data = data_file['data_all_atom'][::10]
    potential = (data_file['potential'] - np.min(data_file['potential']))[::10]
    print(f"Data shape (flat): {data.shape}")
    print(f"Potential shape: {potential.shape}")

    # compute aligned distance instead 
    sq_dist = pdist(data, metric=aligned_distance)
    breakpoint()
    # Reshape data to (n_frames, n_atoms, 3)
    n_frames = data.shape[0]
    n_atoms = data.shape[1] // 3
    data = data.reshape(n_frames, n_atoms, 3)
    print(f"Data shape (reshaped): {data.shape}")

    # Align trajectory (uses mdtraj topology file `butane.pdb` in working dir)
    aligned_xyz = align_trajectory(data) # Select only heavy atoms (C)
    print(f"Aligned data shape: {aligned_xyz.shape}")
    breakpoint()
    gram_matrices = gram_matrix(aligned_xyz)
    print(f"Gram matrices shape: {gram_matrices.shape}")

    # Flatten aligned coordinates for downstream processing
    # data_flat = aligned_xyz.reshape(aligned_xyz.shape[0], -1)
    data_flat = gram_matrices.reshape(gram_matrices.shape[0], -1)
    data_flat *= (1/np.sqrt(data_flat.shape[1]))
    print(f"Flattened aligned data shape: {data_flat.shape}")
    # compute a square distance matrix 
    # sq_dists = squareform(pdist(data_flat, 'sqeuclidean'))
    breakpoint()

    # Load kbT if available, otherwise use default
    if 'kbT' in data_file:
        kbT = data_file['kbT']
        print(f"Using kbT from file: {kbT}")
    else:
        kbT = 2.479  # Default value (kJ/mol at 300K)
        print(f"Using default kbT: {kbT}")

    # Convert potential to target measure
    target_measure = np.exp(-potential / kbT)
    print(f"Target measure shape: {target_measure.shape}")
    print(f"Target measure range: [{target_measure.min():.6e}, {target_measure.max():.6e}]")

    section_print("Data loaded and prepared. Call run_ksum_delta_tests(data_flat, ...) to run k-sum tests.")
    breakpoint()
    # run_ksum_delta_tests(data_flat, \
    #                      deltas=np.arange(0.1, 1.0, 0.1), \
    #                         outdir='ksum_test_plots_carbons', num_evecs=10, target_measure=target_measure)
    
    delta = 0.2
    epsilon = 0.038
    net_idx, _ = epsilon_net(data_flat.T, delta)
    # subsq = squareform(pdist(data_flat[net_idx, :], 'sqeuclidean'))
    # subsq_csr = csr_matrix(subsq)
    # dmap = TargetMeasureDiffusionMap(epsilon=epsilon, num_evecs=10, n_neigh=None, target_measure=target_measure[net_idx])
    # # dmap.sq_dists = subsq_csr
    # # dmap.flag = True
    # dmap.construct_generator(data_flat[net_idx, :])
    # L = dmap.L
    _, _, L = create_laplacian_dense(data_flat[net_idx, :], target_measure[net_idx], epsilon)
    print(f"Constructed diffusion map generator matrix L with shape: {L.shape}")
    breakpoint()
    dihedrals = data_file['dihedrals'][::10]
    B_bool, A_bool, C_bool = get_metastable_states(dihedrals)
    B_bool_net = B_bool[net_idx]
    A_bool_net = A_bool[net_idx]
    C_bool_net = C_bool[net_idx]
    print(f"Set B size (net): {np.sum(B_bool_net)}")
    print(f"Set A size (net): {np.sum(A_bool_net)}")
    print(f"Set C size (net): {np.sum(C_bool_net)}")
    # Compute committor
    breakpoint()
    committor = solve_committor_dense(L, B_bool_net, C_bool_net, len(net_idx))

    breakpoint()

    # plot it 
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(dihedrals[net_idx], committor, c=committor, cmap='viridis', s=20, edgecolor='k')
    ax.set_xlabel('Dihedral angle (rad)')
    ax.set_ylabel('Committor')
    ax.set_title('Committor vs Dihedral Angle (Delta Net)')
    plt.colorbar(sc, ax=ax, label='Committor value')
    plt.tight_layout()
    plt.savefig('committor_vs_dihedral_delta_net.png', dpi=300)
    print("Saved plot to committor_vs_dihedral_delta_net.png")

if __name__ == '__main__':
    pass

