"""
Kernel construction on butane data and diffusion map coordinates.

This script loads butane data and constructs:
1. A standard gaussian kernel using euclidean distances
2. A graph on the data using euclidean distances, and a kernel matrix given by 
   the geodesic distance on the graph
3. Computes a diffusion map like in run/butane.py, then constructs the above 
   two types of kernels on the diffusion map coordinates
"""

import numpy as np
import scipy.sparse as sps
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os
from scipy.sparse import issparse

def load_butane_data():
    """Load butane data from /data/ directory."""
    # Get the current working directory (should be TMDmaps/)
    cwd = os.getcwd()
    fname = os.path.join(cwd, "data", "butane", "butane_metad.npz")
    
    inData = np.load(fname)
    print("Keys in data:")
    print(list(inData.keys()))
    
    data = inData["data"]
    print(f"Data shape: {data.shape}")
    
    # Load delta net indices if available
    # delta_fname = os.path.join(cwd, "data", "butane", "butane_metad_deltanet.npz")
    # if os.path.exists(delta_fname):
    #     delta_idx = np.load(delta_fname)["delta_idx"]
    #     data = data[delta_idx, :]
    #     print(f"Data shape after delta net subsampling: {data.shape}")
    data = data[::10, :]
    return data


def create_kernel_euclidean(data, epsilon):
    """
    Create a standard gaussian kernel using euclidean distances.
    
    Parameters:
    -----------
    data : array, shape (n_samples, n_features)
        Input data
    epsilon : float
        Kernel bandwidth parameter
        
    Returns:
    --------
    K : array, shape (n_samples, n_samples)
        Gaussian kernel matrix
    sqdists : array, shape (n_samples, n_samples)
        Squared euclidean distances
    """
    # Compute squared euclidean distances
    sqdists = cdist(data, data, 'sqeuclidean')
    
    return sqdists


def create_kernel_geodesic(sq_dists, n_neighbors=None, k_threshold=None):
    """
    Create a graph on the data using euclidean distances, then compute 
    a kernel matrix using geodesic distances on the graph.
    
    Parameters:
    -----------
    sq_dists : array, shape (n_samples, n_samples)
        Precomputed squared euclidean distances
    epsilon : float
        Kernel bandwidth parameter for geodesic distances
    n_neighbors : int, optional
        Number of neighbors for k-nearest neighbors graph
        If None, uses all points (fully connected)
    k_threshold : float, optional
        Threshold for edge creation (points within this distance are connected)
        If None and n_neighbors is None, uses all points
        
    Returns:
    --------
    K : array, shape (n_samples, n_samples)
        Kernel matrix from geodesic distances
    geodesic_dists : array, shape (n_samples, n_samples)
        Geodesic distances on the graph
    graph_adj : sparse matrix, shape (n_samples, n_samples)
        Adjacency matrix of the graph (euclidean distances as edge weights)
    """
    n_samples = sq_dists.shape[0]
    
    # Use precomputed squared distances - no recomputation needed
    euc_dists = np.sqrt(sq_dists)
    
 
    print("Computing geodesic distances...")
    
    if n_neighbors is not None:
        # Use sklearn's NearestNeighbors with precomputed distance matrix
        print("Constructing k-nearest neighbors graph from precomputed distances...")
        neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='precomputed')
        neigh.fit(euc_dists)
        graph_adj = neigh.kneighbors_graph(euc_dists, mode='distance')
        # Symmetrize
        graph_adj = 0.5 * (graph_adj + graph_adj.T)
    elif k_threshold is not None:
        # Connect points within threshold
        graph_adj = sps.csr_matrix(np.where(euc_dists < k_threshold, euc_dists, 0))
    else:
        # Fully connected graph with euclidean distances as edge weights
        graph_adj = sps.csr_matrix(euc_dists)
    
    # Compute geodesic distances (shortest paths) on the graph
    # Use Dijkstra's algorithm
    print(f"Graph sparsity: {issparse(graph_adj)}")
    print("Computing shortest paths from pairwise distances...")
    geodesic_dists = shortest_path(
        graph_adj, 
        method='D', 
        directed=False,
        return_predecessors=False
    )
    
    # Handle disconnected components (set to large distance)
    geodesic_dists[geodesic_dists == np.inf] = np.max(geodesic_dists[geodesic_dists != np.inf]) * 2
    
    # Create kernel from geodesic distances
    print("Creating kernel from geodesic distances...")
    # K = np.exp(-geodesic_dists**2 / (2.0 * epsilon))
    
    return geodesic_dists, graph_adj


def compute_diffusion_map(data, target_measure, epsilon, num_eigvecs=10, sq_dists=None):
    """
    Compute diffusion map like in run/butane.py.
    
    Parameters:
    -----------
    data : array, shape (n_samples, n_features)
        Input data
    target_measure : array, shape (n_samples,)
        Target measure for reweighting (e.g., exp(-potential/kbT))
    epsilon : float
        Kernel bandwidth parameter
    num_eigvecs : int
        Number of eigenvectors to compute
    sq_dists : array, shape (n_samples, n_samples), optional
        Precomputed squared euclidean distances. If None, will be computed from data.
        
    Returns:
    --------
    dmap : array, shape (n_samples, num_eigvecs-1)
        Diffusion map coordinates
    evecs : array, shape (n_samples, num_eigvecs-1)
        Eigenvectors of the generator
    evals : array, shape (num_eigvecs-1,)
        Eigenvalues of the generator
    L : array, shape (n_samples, n_samples)
        Generator matrix
    K : array, shape (n_samples, n_samples)
        Kernel matrix
    """
    num_samples = data.shape[0]
    
    # Create distance matrix
    if sq_dists is None:
        sqdists = cdist(data, data, 'sqeuclidean')
    else:
        sqdists = sq_dists
    
    # Create Kernel
    K = np.exp(-sqdists / (2.0 * epsilon))
    
    # Create Graph Laplacian (following butane.py create_laplacian_dense)
    kde = K.sum(axis=1)
    u = (target_measure**(0.5)) / kde
    U = np.diag(u)
    W = U @ K @ U
    stationary = W.sum(axis=1)
    P = np.diag(stationary**(-1)) @ W
    L = (P - np.eye(num_samples)) / epsilon
    
    # Compute diffusion map coordinates
    # Symmetrize the generator
    Dinv_onehalf = np.diag(stationary**(-0.5))
    D_onehalf = np.diag(stationary**(0.5))
    Lsymm = D_onehalf @ L @ Dinv_onehalf
    
    # Compute eigvals, eigvecs
    evals, evecs = sps.linalg.eigsh(Lsymm, k=num_eigvecs, which='SM')
    
    # Convert back to L^2 norm-1 eigvecs of L
    evecs = (Dinv_onehalf) @ evecs
    evecs /= (np.sum(evecs**2, axis=0))**(0.5)
    
    # Ignore first eigval/eigfunc (constant eigenvector)
    idx = evals.argsort()[::-1][1:]
    evals = np.real(evals[idx])
    evecs = np.real(evecs[:, idx])
    
    # Create diffusion map coordinates
    dmap = np.dot(evecs, np.diag(np.sqrt(-1. / evals)))
    
    return dmap, evecs, evals, L, K


def main():
    """Main function to construct all kernels."""
    print("=" * 60)
    print("Loading butane data...")
    print("=" * 60)
    data = load_butane_data()
    breakpoint()  # Breakpoint 1: After loading data
    
    # Set up parameters
    epsilon = 0.1  # From butane.py for metad + deltanet
    # For target measure, we'll use a simple uniform measure
    # In practice, you might load potential from the data file
    target_measure = np.ones(data.shape[0])
    
    print("\n" + "=" * 60)
    print("1. Constructing standard gaussian kernel using euclidean distances")
    print("=" * 60)
    K_euclidean, sqdists_euclidean = create_gaussian_kernel_euclidean(data, epsilon)
    sqdists_euclidean *= 12.0 # mass correction factor
    print(f"Gaussian kernel shape: {K_euclidean.shape}")
    print(f"Kernel entries: min={K_euclidean.min():.6f}, max={K_euclidean.max():.6f}")
    # plt.figure(figsize=(10, 8)); plt.imshow(K_euclidean, cmap='viridis', aspect='auto'); plt.colorbar(label='Kernel value'); plt.title('Euclidean Gaussian Kernel'); plt.savefig('K_euclidean.png', dpi=300, bbox_inches='tight'); plt.close()
    breakpoint()  # Breakpoint 2: After constructing euclidean gaussian kernel
    
    print("\n" + "=" * 60)
    print("2. Constructing graph and kernel from geodesic distances")
    print("=" * 60)
    # Use k-nearest neighbors for graph construction
    n_neighbors = min(50, data.shape[0] - 1)  # Use 50 neighbors or all if less
    K_geodesic, geodesic_dists, graph_adj = create_graph_kernel_geodesic(
        sqdists_euclidean, epsilon, n_neighbors=n_neighbors
    )
    print(f"Geodesic kernel shape: {K_geodesic.shape}")
    print(f"Geodesic kernel entries: min={K_geodesic.min():.6f}, max={K_geodesic.max():.6f}")
    print(f"Geodesic distances: min={geodesic_dists[geodesic_dists > 0].min():.6f}, "
          f"max={geodesic_dists.max():.6f}")
    # save geodesic distances for future use
    np.savez('geodesic_data_euclidean.npz', geodesic_dists=geodesic_dists, sqdists=sqdists_euclidean)
    breakpoint()  # Breakpoint 3: After constructing geodesic kernel
    
    print("\n" + "=" * 60)
    print("3. Computing diffusion map")
    print("=" * 60)
    # Load target measure if available
    cwd = os.getcwd()
    fname = os.path.join(cwd, "data", "butane", "butane_metad.npz")
    inData = np.load(fname)
    if "potential" in inData and "kbT_roomtemp" in inData:
        potential = inData["potential"]
        kbT_roomtemp = inData["kbT_roomtemp"]
        # Subsample if using delta net
        # delta_fname = os.path.join(cwd, "data", "butane", "butane_metad_deltanet.npz")
        # if os.path.exists(delta_fname):
        #     delta_idx = np.load(delta_fname)["delta_idx"]
        #     potential = potential[delta_idx]
        target_measure = np.exp(-potential / kbT_roomtemp)
        print(f"Using target measure from data (kbT_roomtemp={kbT_roomtemp})")
    
    dmap, evecs, evals, L, K_dmap = compute_diffusion_map(
        data, np.ones_like(target_measure), epsilon, num_eigvecs=10, sq_dists=sqdists_euclidean
    )
    print(f"Diffusion map shape: {dmap.shape}")
    print(f"Eigenvalues: {evals[:5]}")
    # save diffusion map
    np.savez('butane_diff_map.npz', data=data, sq_dists=sqdists_euclidean, dmap=dmap, evecs=evecs, evals=evals)
    breakpoint()  # Breakpoint 4: After computing diffusion map
    
    print("\n" + "=" * 60)
    print("4. Constructing kernels on diffusion map coordinates")
    print("=" * 60)
    
    # Gaussian kernel on diffusion map
    print("4a. Gaussian kernel on diffusion map...")
    K_dmap_euclidean, sqdists_dmap_euclidean = create_gaussian_kernel_euclidean(
        dmap, epsilon
    )
    print(f"Gaussian kernel on dmap shape: {K_dmap_euclidean.shape}")
    print(f"Kernel entries: min={K_dmap_euclidean.min():.6f}, "
          f"max={K_dmap_euclidean.max():.6f}")
    
    # Geodesic kernel on diffusion map
    print("\n4b. Geodesic kernel on diffusion map...")
    K_dmap_geodesic, geodesic_dists_dmap, graph_adj_dmap = create_graph_kernel_geodesic(
        sqdists_dmap_euclidean, epsilon, n_neighbors=n_neighbors
    )
    print(f"Geodesic kernel on dmap shape: {K_dmap_geodesic.shape}")
    print(f"Geodesic kernel entries: min={K_dmap_geodesic.min():.6f}, "
          f"max={K_dmap_geodesic.max():.6f}")
    # save 
    np.savez('geodesic_data_dmap', geodesic_dists=geodesic_dists_dmap, sqdists=sqdists_dmap_euclidean)
    breakpoint()  # Breakpoint 5: After constructing kernels on diffusion map
    
    print("\n" + "=" * 60)
    print("All kernels constructed successfully!")
    print("=" * 60)
    
    # Return results for potential further use
    results = {
        'data': data,
        'K_euclidean': K_euclidean,
        'K_geodesic': K_geodesic,
        'dmap': dmap,
        'K_dmap_euclidean': K_dmap_euclidean,
        'K_dmap_geodesic': K_dmap_geodesic,
        'epsilon': epsilon
    }
    
    return results


if __name__ == '__main__':
    pass
    # Uncomment to run main()
    # results = main()
