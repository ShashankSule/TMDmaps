import logging
import uuid

import numpy as np 
from scipy.spatial.distance import cdist
import scipy 
import os
import datetime
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import hydra 
from omegaconf import DictConfig, OmegaConf, open_dict
from src.helpers import compute_pairwise_alignment_rmsd

log = logging.getLogger(__name__)


# ============================================================================
# Distance function implementations
# ============================================================================
def compute_euclidean_sqdists(data, **kwargs):
    """Compute squared Euclidean distance matrix.
    
    Parameters
    ----------
    data : array, shape (n_samples, n_features)
        Data array
    **kwargs : unused
        Additional parameters (for compatibility)
    
    Returns
    -------
    sqdists : array, shape (n_samples, n_samples)
        Squared Euclidean distance matrix
    """
    return cdist(data, data, 'sqeuclidean')


def compute_aligned_sqdists(data, **kwargs):
    """Compute squared RMSD distance matrix after optimal alignment (Kabsch).
    
    Parameters
    ----------
    data : array, shape (n_samples, n_features)
        Data array. Will reshape to (n_samples, -1, 3) automatically.
    **kwargs : unused
        Additional parameters (for compatibility)
    
    Returns
    -------
    sqdists : array, shape (n_samples, n_samples)
        Squared RMSD distance matrix after alignment
    """
    n_samples = data.shape[0]
    # Reshape to (n_samples, n_atoms, 3) automatically
    data_reshaped = data.reshape(n_samples, -1, 3)
    # Compute pairwise RMSD via alignment
    rmsd_matrix = compute_pairwise_alignment_rmsd(data_reshaped)
    return rmsd_matrix ** 2  # Return squared distances


def get_distance_function(distance_metric):
    """Get distance computation function by name.
    
    Parameters
    ----------
    distance_metric : str
        Name of distance metric: 'euclidean' or 'aligned'
    
    Returns
    -------
    distance_fn : callable
        Function that computes distance matrix
    """
    distance_functions = {
        'euclidean': compute_euclidean_sqdists,
        'aligned': compute_aligned_sqdists,
    }
    
    if distance_metric not in distance_functions:
        raise ValueError(f"Unknown distance metric: {distance_metric}. Available: {list(distance_functions.keys())}")
    
    return distance_functions[distance_metric]


def get_distance_fn_from_cfg(cfg):
    """Extract distance function and kwargs from hydra config.
    
    Parameters
    ----------
    cfg : DictConfig or None
        Hydra config. If None or missing 'distance' key,
        defaults to Euclidean squared distances.
    
    Returns
    -------
    distance_fn : callable
    distance_kwargs : dict
    """
    if cfg is not None and 'distance' in cfg:
        distance_fn = get_distance_function(cfg['distance']['metric'])
        distance_kwargs = dict(cfg['distance'].get('kwargs', {}))
    else:
        distance_fn = compute_euclidean_sqdists
        distance_kwargs = {}
    return distance_fn, distance_kwargs


def get_or_compute_sqdists(cfg, data):
    """Load a pre-computed squared distance matrix or compute (and save) a new one.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config.  Looks for ``cfg.distance.precomputed_path``.
        If that key is present and not None, the matrix is loaded from disk.
        Otherwise it is computed with the distance function specified in
        ``cfg.distance.metric`` (defaulting to Euclidean), and saved to
        ``data/butane/dist_matrices/<timestamp>_<hex>/``.
    data : array, shape (n_samples, n_features)
        Raw coordinate data (only used when computing fresh).

    Returns
    -------
    sqdists : array, shape (n_samples, n_samples)
        Squared pairwise distance matrix.
    """
    # --- Try loading a pre-computed matrix ---
    precomputed_path = None
    if cfg is not None and 'distance' in cfg:
        precomputed_path = cfg['distance'].get('precomputed_path', None)

    if precomputed_path is not None:
        log.info(f"Loading pre-computed distance matrix from {precomputed_path}")
        sqdists = np.load(precomputed_path)['sqdists']
        log.info(f"Loaded distance matrix shape: {sqdists.shape}")
        return sqdists

    # --- Compute fresh ---
    distance_fn, distance_kwargs = get_distance_fn_from_cfg(cfg)
    log.info("Computing pairwise distance matrix...")
    sqdists = distance_fn(data, **distance_kwargs)
    log.info(f"Distance matrix shape: {sqdists.shape}")

    # --- Save to disk ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    hex_key = uuid.uuid4().hex[:8]
    folder_name = f"{timestamp}_{hex_key}"
    save_dir = os.path.join(os.getcwd(), "data", "butane", "dist_matrices", folder_name)
    os.makedirs(save_dir, exist_ok=True)

    sqdists_path = os.path.join(save_dir, "sqdists.npz")
    np.savez(sqdists_path, sqdists=sqdists)
    log.info(f"Saved distance matrix to {sqdists_path}")

    config_path = os.path.join(save_dir, "config.yaml")
    OmegaConf.save(cfg, config_path)
    log.info(f"Saved config snapshot to {config_path}")

    return sqdists


@hydra.main(version_base=None, config_path="../data/butane/configs", config_name="butane_run_config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    simulate_rate(cfg)

def rate(sqdists, target_measure, epsilon, states, kbT, dihedrals, logger):
    """Compute committor, transition rate, and escape rate for a single epsilon.

    Parameters
    ----------
    sqdists : array, shape (n_samples, n_samples)
        Pre-computed squared pairwise distance matrix.
    target_measure : array, shape (n_samples,)
        Target (reweighting) measure.
    epsilon : float
        Bandwidth parameter.
    states : dict
        Metastable state boolean masks with keys 'A', 'B', 'C'.
    kbT : float
        Thermal energy k_B T.
    dihedrals : array, shape (n_samples,)
        Dihedral angles for committor interpolation.
    logger : logging.Logger
        Logger instance for status messages.

    Returns
    -------
    solns : dict
        Keys: 'committor', 'epsilon', 'rate', 'rho_A'.
    """
    B, C = states['B'], states['C']
    n_samples = sqdists.shape[0]

    # set up diffusion map DENSE 
    [stationary, K, L] = create_laplacian_dense(sqdists, target_measure, epsilon=epsilon)

    # solve the committor problem 
    try: 
        q = solve_committor_dense(L, B, C, n_samples) 

        # compute rate
        rate_val = compute_transition_rate(L, K, target_measure, q, beta=(1/kbT), effective_dim=12, epsilon=epsilon, C=C, dense=True)
        rho_A = compute_rho_interpolation(q, dihedrals) 

        solns = {'committor': q, 'epsilon': epsilon, 'rate': rate_val, 'rho_A': rho_A}
        logger.info(f"epsilon={epsilon:.6e}  rate={rate_val:.6e}  rho_A={rho_A:.6e}")
    except Exception as e:
        logger.error(f"Exception at epsilon={epsilon}: {e}")
        solns = {'committor': None, 'epsilon': epsilon, 'rate': None, 'rho_A': None}
    return solns  

def get_metadynamicsdata(deltanet): 
    # Load data
    fname = os.getcwd() + "/data/butane/butane_metad_deltanet_dihedrals.npz"
    inData = np.load(fname)
    print("Keys in data:")
    print(list(inData.keys()))

    data = inData["data"]
    #data = inData["data_all_atom"]
    
    print("Data shape from trajectory:")
    print(data.shape)
    dihedrals = inData["dihedrals"]
    # potential = inData["potential"]
    kbT = 1/inData["beta"]
    print(f"kbT for data:{kbT}")
    kbT_roomtemp = 1/inData["beta"]

    print(f"kbT for room temperature:{kbT_roomtemp}")

    # Load up delta net indices
    delta_idx = inData["net_idx"]

    # Define Target Measure
    # target_measure = np.exp(-potential/(kbT_roomtemp))
    target_measure = inData["target_measure"]
    # Subsample dataset (in time or in space(deltanet) )
    indices = np.arange(data.shape[0])
    if deltanet: 
        sub_indices = delta_idx
    else:
        sub_indices = indices[-np.shape(delta_idx)[0]:]
    
    new_data = data[sub_indices, :]
    # target_measure = np.exp(-potential[sub_indices]/(kbT_roomtemp))
    target_measure = target_measure[sub_indices]
    num_samples = new_data.shape[0]
    num_features = new_data.shape[1]
    print(f"number of samples for subsampled data:{num_samples}")

    dihedrals = dihedrals[sub_indices]
    return new_data, dihedrals, target_measure, kbT 

def create_laplacian_dense(sqdists, target_measure, epsilon):
    """Build the TMDmap graph Laplacian from a pre-computed distance matrix.

    Parameters
    ----------
    sqdists : array, shape (n_samples, n_samples)
        Squared pairwise distance matrix.
    target_measure : array, shape (n_samples,)
        Target (reweighting) measure.
    epsilon : float
        Bandwidth parameter.

    Returns
    -------
    stationary : array, shape (n_samples,)
    K : array, shape (n_samples, n_samples)
        Kernel matrix.
    L : array, shape (n_samples, n_samples)
        Generator (graph Laplacian).
    """
    num_samples = sqdists.shape[0]

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

def compute_transition_rate(L, K, target_measure, q, beta, effective_dim, epsilon, C, dense):
    N = L.shape[0]
    kde = np.asarray(K.sum(axis=1)).ravel()
    #kde *=  (1.0/N)*(2*np.pi*epsilon)**(-effective_dim/2) 
    kde *= (1./np.sum(kde))
    Z_dmap = (1.0/N)*np.sum(target_measure / kde)
    weight_Zdmap = (target_measure/(kde*Z_dmap)).flatten()
    if dense: 
        A = L 
    else: 
        A = L.toarray()
    # Note: for i in C, both sum_j L_ij = 0 and sum_j Lij qj = 0, so sum_ij L_ij (q_j - q_i)**2 = sum_ij L_ij q_j*2 
    rate = (1/beta)*(1/np.count_nonzero(C))*np.sum(weight_Zdmap[C]*A[C, :].dot(q**2))
    return rate

def getboolz(dihedrals, ra, rb): 
    """Compute metastable state boolean masks.

    Returns
    -------
    states : dict
        Keys 'A', 'B', 'C' mapping to boolean arrays.
    """
    Acenter = np.pi
    A = np.abs(dihedrals - Acenter) < ra
    B = np.logical_or(np.abs(dihedrals - np.pi/3) < rb, np.abs(dihedrals - 5*(np.pi/3)) < rb)
    C = np.ones(dihedrals.shape[0], dtype=bool)
    C[A] = False
    C[B] = False
    return {'A': A, 'B': B, 'C': C}

# ============================================================================
# Escape rate computation modules
# ============================================================================
def load_equilibrium_data(fname=None, subsample=10):
    """Load equilibrium (unbiased) trajectory data.
    
    Parameters
    ----------
    fname : str, optional
        Path to the equilibrium .npz file. Defaults to
        'data/butane/butane_300K.npz' relative to cwd.
    subsample : int
        Take every `subsample`-th frame.
    
    Returns
    -------
    dihedrals_eq : array, shape (n_samples,)
        Dihedral angles wrapped to [0, 2*pi).
    beta : float
        Inverse temperature 1/kbT.
    """
    if fname is None:
        fname = os.getcwd() + "/data/butane/butane_300K.npz"
    inData = np.load(fname)
    dihedrals_eq = inData["dihedrals"][::subsample]
    dihedrals_eq[dihedrals_eq < 0] += 2 * np.pi
    beta = 1.0 / inData["kbT"]
    return dihedrals_eq, beta


def interpolate_committor(q, dihedrals_from, dihedrals_to):
    """Interpolate a committor from one set of dihedrals to another.
    
    Parameters
    ----------
    q : array, shape (n_biased,)
        Committor values on the biased (source) dihedrals.
    dihedrals_from : array, shape (n_biased,)
        Dihedral angles where `q` is defined.
    dihedrals_to : array, shape (n_eq,)
        Dihedral angles to interpolate onto.
    
    Returns
    -------
    q_interpolated : array, shape (n_eq,)
        Committor evaluated at `dihedrals_to`.
    """
    interp_fn = scipy.interpolate.interp1d(
        dihedrals_from.flatten(), q.flatten(), kind='linear'
    )
    return interp_fn(dihedrals_to).flatten()


def compute_escape_rate_from_committor(q_mu):
    """Compute the escape rate from an interpolated committor.
    
    Parameters
    ----------
    q_mu : array, shape (n_eq,)
        Committor evaluated on the equilibrium sample.
    
    Returns
    -------
    escape_rate : float
        Mean of (1 - q) over the equilibrium sample.
    """
    return np.mean(1.0 - q_mu)


def compute_rho_interpolation(q, dihedrals):
    """Estimate escape rate by interpolating the committor onto equilibrium data.
    
    Parameters
    ----------
    q : array, shape (n_biased,)
        Committor on the biased dataset.
    dihedrals : array, shape (n_biased,)
        Dihedral angles corresponding to `q`.
    
    Returns
    -------
    escape_rate : float
        Mean of (1 - q) over equilibrium samples.
    """
    dihedrals_eq, _ = load_equilibrium_data()
    q_eq = interpolate_committor(q, dihedrals, dihedrals_eq)
    return compute_escape_rate_from_committor(q_eq)


def simulate_rate(cfg): 
    flag = cfg['flag']
    min_epsilon, max_epsilon = cfg['epsilon_min'], cfg['epsilon_max']
    n_epsilons = cfg.get('n_epsilons', 100)
    if cfg['log_scale']:
        epsilons = np.logspace(np.log10(min_epsilon), np.log10(max_epsilon), n_epsilons)
    else:
        epsilons = np.linspace(min_epsilon, max_epsilon, n_epsilons)

    # --- Load data and metastable states once ---
    new_data, dihedrals, target_measure, kbT = get_metadynamicsdata(flag)
    states = getboolz(dihedrals, 0.2, 0.1)

    # --- Compute pairwise distances once (or load pre-computed) ---
    sqdists = get_or_compute_sqdists(cfg, new_data)

    # --- Sweep over epsilons ---
    trials = [] 
    log.info("Starting trials...")
    for epsilon in epsilons:
        soln = rate(sqdists, target_measure, epsilon, states, kbT, dihedrals, log)
        soln['deltanet'] = flag
        trials.append(soln)
    trials = np.array(trials)

    # --- Save ---
    log.info("Now saving...")
    filename = 'rate_butane' + '_' + str(datetime.datetime.now())
    save_directory = os.getcwd() + '/data/butane/' + 'rate_butane' + '_' + str(datetime.datetime.now())
    np.savez(save_directory, trials=trials)
    with open_dict(cfg):
        cfg.date = str(datetime.datetime.now())
        cfg.saved_filename = filename
    OmegaConf.save(cfg, os.getcwd() + "/data/butane/" + filename + "_config.yaml")
    log.info("Save finished!")

if __name__ == '__main__':    
    main()




