"""
Script to compute delta nets on geodesic distances.

This script:
1. Loads geodesic distances from geodesic_data_euclidean.npz and geodesic_data_dmap.npz
2. Computes delta nets with delta tuned to the distance to the 50th nearest neighbor
3. Loads original butane data and dihedrals, subsamples by factor of 10
4. Extracts delta net indices and corresponding dihedrals
5. Visualizes the dihedrals corresponding to delta net points
"""

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os
import time
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from helpers import epsilon_net, alignment_rmsd_distance
from butane_ksum_tests import align_trajectory, get_metastable_states
from scipy.spatial.distance import pdist, cdist, squareform
from diffusion_map import TargetMeasureDiffusionMap
import pickle 
from multiprocessing import Pool, cpu_count
from itertools import product
import traceback

def section_print(message):
    """Print a section header with decorative lines."""
    print("\n" + "=" * 60)
    print(message)
    print("=" * 60)

def get_max_min_epsilon(sq_dists, k_alpha=0.25): 
    k = int(k_alpha*sq_dists.shape[0])
    neigh = NearestNeighbors(n_neighbors=k+1,
                            metric='precomputed')
    neigh.fit(sq_dists)
    [neigh_dist, neigh_ind] = neigh.kneighbors(sq_dists)
    max_epsilon = np.max(neigh_dist[:, k])
    min_epsilon = np.min(neigh_dist[:, 1])
    return max_epsilon, min_epsilon, k 

def convert_to_knn(sq_dists, knn): 
    neigh = NearestNeighbors(n_neighbors=knn,
                                 metric='precomputed')
    neigh.fit(sq_dists)
    knn_sq_dists = neigh.kneighbors_graph(sq_dists, mode='distance')
    knn_sq_dists.sort_indices()
    return knn_sq_dists

def compute_committor(data, sq_dist, dihedrals, target_measure, epsilon, knn, delta): 
    # get committor for a specific knn, deltanet value 
    net_idx, _ = epsilon_net(data.T, delta) # get delta net 
    knn = min(knn, net_idx.shape[0]-1) # reset knn
    data_net, target_measure_net, dihedrals_net = data[net_idx, :], target_measure[net_idx], dihedrals[net_idx] # get new data
    dmap = TargetMeasureDiffusionMap(target_measure=target_measure_net, n_neigh=knn, epsilon=epsilon)

    # get boundary states 
    B_bool, A_bool, C_bool = get_metastable_states(dihedrals_net)
    breakpoint()
    print(f"Boundary states: B={np.sum(B_bool)}, A={np.sum(A_bool)}, C={np.sum(C_bool)}")

    # get Laplacian, solve committor 
    dmap.sq_dists = convert_to_knn(sq_dist[net_idx,:][:, net_idx], knn)
    print(f'Type of dmap.sq_dists: {type(dmap.sq_dists)}')
    dmap.flag = True
    dmap.construct_generator(data_net)
    L = dmap.L 
    print(f'Type of L: {type(L)}')
    committor = dmap.construct_committor(L, B_bool, C_bool)

    return L, committor, dihedrals_net

def plot_committor_vs_dihedrals(dihedrals_net, committor, delta, epsilon, knn):
    fig, ax = plt.subplots()
    ax.scatter(dihedrals_net, committor)
    ax.set_xlabel("Dihedrals")
    ax.set_ylabel("Committor")
    ax.set_title(f"Committor vs Dihedrals (delta={delta}, epsilon={epsilon}, knn={knn})")
    plt.savefig(f'sim_figures/committor_vs_dihedrals_delta{delta:.4f}_epsilon{epsilon:.4f}.png')
    plt.close(fig)

def butane_data_subsampled(n_target=5000): 
    data_dir = np.load("butane_metad_nonaligned.npz", allow_pickle=True)
    total = data_dir['data_all_atom'].shape[0]
    stride = max(1, total // n_target)
    data = data_dir['data_all_atom'][::stride][:n_target]
    potential = data_dir['potential'][::stride][:n_target]
    dihedrals = data_dir['dihedrals'][::stride][:n_target]
    beta = 1/data_dir['kbT']
    potential = potential - np.min(potential)
    target_measure = np.exp(-beta * potential)
    return data, dihedrals, target_measure, beta 

# Define at module level so it can be pickled
def simulate_wrapper(args):
    """Wrapper to unpack parameters and run simulation"""
    epsilon, delta, knn, data, sq_dists, dihedrals, target_measure = args
    try:
        L, committor, dihedrals_net = compute_committor(data, sq_dists, dihedrals, target_measure, epsilon, knn, delta)
        plot_committor_vs_dihedrals(dihedrals_net, committor, delta, epsilon, knn)
        return {'L': L, 'committor': committor, 'epsilon': epsilon, 'delta': delta, 'knn': knn}
    except Exception as e:
        print(f"Error with params (eps={epsilon}, delta={delta}, knn={knn}): {e}")
        traceback.print_exc()
        return None

def main():
    # define main simulation function 

    data_all_atom, dihedrals, target_measure, beta = butane_data_subsampled()

    # flatten to (n_frames, 42)
    n_frames = data_all_atom.shape[0]
    data = data_all_atom.reshape(n_frames, -1)
    print(f"Data shape: {data.shape}")
    breakpoint()
    # # Align trajectory (uses mdtraj topology file `butane.pdb` in working dir)
    # data_3d = data_all_atom.reshape(n_frames, -1, 3)
    # aligned_xyz = align_trajectory(data_3d)[:, [3,6,9,13], :] # Select only heavy atoms (C)
    # data = aligned_xyz.reshape(n_frames, -1)
    # print(f"Aligned data shape: {data.shape}")

    print('Computing squared distance..')
    # sq_dists = squareform(pdist(data, 'euclidean'))
    deltas = np.linspace(0.06, 0.3, 10)
    epsilons = np.linspace(0.4,8.0,1)

    # Sweep over deltas using alignment RMSD distance in epsilon_net
    def aligned_dist_fn(x_col, Y_cols):
        return alignment_rmsd_distance(x_col.T, Y_cols.T)
    breakpoint()
    net_sizes = np.zeros(len(deltas), dtype=int)
    for i, delta in enumerate(deltas):
        net_idx, _ = epsilon_net(data.T, delta, distance_fn=aligned_dist_fn)
        net_sizes[i] = len(net_idx)
        print(f"\n  delta={delta:.4f} -> N={net_sizes[i]}")
    breakpoint()
    print("\nDelta -> Net size summary:")
    for d, n in zip(deltas, net_sizes):
        print(f"  {d:.4f}  {n}")

    # # Use last delta for downstream processing
    # net_idx, _ = epsilon_net(data.T, deltas[-1], distance_fn=aligned_dist_fn)
    # max_epsilon, min_epsilon, k = get_max_min_epsilon(sq_dists[net_idx,:][:, net_idx], k_alpha=0.5)
    # print(f"Delta net size for delta={delta}: {net_idx.shape[0]}")
    # np.savez('butane_metad_deltanet_dihedrals.npz',  data_all_atom = data_all_atom, data=data, dihedrals=dihedrals, target_measure=target_measure, \
    #          net_idx=net_idx, delta=delta, beta=beta, eps_range=(min_epsilon, max_epsilon))
    # # Write configuration file
    # with open('butane_run_config.yaml', 'w') as f:
    #     f.write(f"data_file: butane_metad_deltanet_dihedrals.npz\n")
    #     f.write(f"epsilon_min: {min_epsilon}\n")
    #     f.write(f"epsilon_max: {max_epsilon}\n")
    #     f.write(f"flag: True # to use the delta net or not? \n")
    #     f.write(f"log_scale: False\n")
    # knns = np.array([1024, 2048, 4096, 8192, 16384])
    # knns = np.array([32])
    # # Create wrapper function that unpacks parameters
    #     # Generate all parameter combinations, including data that each worker needs
    # param_combinations = [(eps, delta, knn, data, sq_dists, dihedrals, target_measure) 
    #                       for eps, delta, knn in product(epsilons, deltas, knns)]
    # print(f"Total combinations to run: {len(param_combinations)}")
    
    # # Run simulations sequentially with progress bar
    # results = []
    # for params in tqdm(param_combinations, desc="Running simulations"):
    #     result = simulate_wrapper(params)
    #     if result is not None:
    #         results.append(result)
    
    # print(f"\nCompleted {len(results)} successful simulations out of {len(param_combinations)}")

    # # Run simulations in parallel
    # n_processes = cpu_count()
    # print(f"Running with {n_processes} processes...")
    
    # # simulate_wrapper(param_combinations[0])
    # with Pool(processes=n_processes) as pool:
    #     results = pool.map(simulate_wrapper, param_combinations)
    
    # # Filter out None results (failed simulations)
    # results = [r for r in results if r is not None]
    # print(f"Completed {len(results)} successful simulations")
    
    # # Save as pickle
    # with open('simulation_results.pkl', 'wb') as f:
    #     pickle.dump(results, f)
    # print(f"Saved results to 'simulation_results.pkl'")
    
    # return results


if __name__ == '__main__':
    results = main()
