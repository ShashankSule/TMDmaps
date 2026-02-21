import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sp_linalg
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import scipy.interpolate as scinterp
import scipy.spatial
import sys
from tqdm import tqdm
import scipy.sparse as sps
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors
sys.path.append("..")
try:
    from . import model_systems as model_systems
except ImportError:
    import model_systems as model_systems
##############################################################
# Functions helping with committor error, plotting committor, etc.
##############################################################
def committor_contours(zeroset=1e-7, oneset=1.0):

    my_levels = np.arange(0.1, 0.6, 0.1)
    my_levels = np.concatenate((my_levels, np.arange(0.6, 1.0, 0.1)))    

    return my_levels 


def is_in_ABC(data, centerx_A, centery_A, rad_A, centerx_B, centery_B, rad_B):
    A_bool = is_in_circle(data[0, :], data[1, :], centerx_A, centery_A, rad_A)
    B_bool = is_in_circle(data[0, :], data[1, :], centerx_B, centery_B, rad_B)
    C_bool = np.logical_not(np.logical_or(A_bool, B_bool))
    return A_bool, B_bool, C_bool


def is_in_circle(x, y, centerx=0, centery=0, rad=1.0): 
    return ((x - centerx)**2 + (y-centery)**2 <= rad**2)


def RMSerror(approx, truth, weights=None, checknans=True):
    if weights is not None:
        diff = weights*(approx - truth)**2
    else:
        diff = (approx - truth)**2 
    if checknans:
        diff = np.delete(diff, np.where(np.isnan(diff))) 
    output = np.sqrt(np.mean(diff))
    return output

##############################################################
# Some linear algebra helpers
##############################################################
def cholesky_hack(C):
    #Computes the (not necessarily unique) Cholesky decomp. for a symmetric positive SEMI-definite matrix, C = LL.T, returns L
    # NOTE: this is a bit more expensive than regular cholesky, should only be used if input matrix is likely not positive definite but it is semi-definite

    # C = MM^T, M^T = QR ---> MM^T = R^T R, so L = R^T
    M = sp_linalg.sqrtm(C)
    R = np.real(np.linalg.qr(M.T)[1])
    return R.T

def remove_zero_row_col(mat):
    # Removes zero rows and columns from an array
    keep_row_bool = mat.any(axis=1)
    mat = mat[keep_row_bool, :]
    mat = mat[:, keep_row_bool]

################################################################################
# Data Plotting Funtions
################################################################################

def plot_cov_ellipse(cov, x, plot_scale=1, color='k', plot_evecs=False,
                     quiver_scale=1):
    """Plots the ellipse corresponding to a given covariance matrix at x (2D) 

    Args:
        cov (array): 2 by 2 symmetric positive definite array.
        x (vector): 2 by 1 vector.
        scale (float, optional): Scale for ellipse size. Defaults to 1.
    """

    evals, evecs = np.linalg.eig(cov)
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    t = np.linspace(0, 2*np.pi)
    val = plot_scale*evecs*evals
    if plot_evecs:
        plt.quiver(*x, *val[:, 0], color=color, angles='xy', scale_units='xy',
                    scale=quiver_scale, width=0.002)
        plt.quiver(*x, *val[:, 1], color=color, angles='xy', scale_units='xy',
                   scale=quiver_scale, width= 0.002)
    else:
        a = np.dot(val, np.vstack([np.cos(t), np.sin(t)]))
    
        plt.plot(a[0, :] + x[0], a[1, :] + x[1], linewidth=0.5, c=color)


def confidence_ellipse(cov, x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`
    
    See how and why this works: https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html
    
    This function has made it into the matplotlib examples collection:
    https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py
    
    Or, once matplotlib 3.1 has been released:
    https://matplotlib.org/gallery/index.html#statistics
    
    I update this gist according to the version there, because thanks to the matplotlib community
    the code has improved quite a bit.
    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    Returns
    -------
    matplotlib.patches.Ellipse
    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
    # render plot with "plt.show()".


def gen_plot_data(potential, plot_params):
    r"""Evaluates potential on a grid for plotting

    Parameters
    ----------
    potential: function,
        Potential function which takes a 2-dim vector as input
        and returns a scalar
    
    plot_params: array-like,
        Python list of values for grid,
        plot_params must be of form [nx, ny, xmin, xmax, ymin, ymax]
        as defined below

        nx, ny : scalar
            Number of steps in x and y direction respectively.
        xmin, xmax : scalar
            Interval bounds [xmin, xmax] for the x-bounds of the grid
        xmin, xmax : scalar
            Interval bounds [ymin, ymax] for the y-bounds of the grid

    Returns
    -------
        plot_data : array-like,
        Python list of cartesian grids for plotting metad data,
        of form [pot, xx, yy]

        pot : array-like
            nx by ny array, cartesian grid of potential function
        xx, yy : array-like
            meshgrid coordinates the potential and bias were evaluated on

    """

    # Read in input lists
    nx, ny, xmin, xmax, ymin, ymax = plot_params
    
    # Compute potential energy contours on grid
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(x, y)
    pot = np.zeros((nx, ny))
    for j in range(nx):
        for i in range(ny):
            a = xx[i, j]
            b = yy[i, j]
            v = np.array([a, b])
            pot[i, j] = potential(v)
    
    plot_data = [pot, xx, yy]
    
    return plot_data

##############################################################################
#  Interpolate Finite Difference Results to data set 
##############################################################################
def grid2traj_bilinterp(xmin, xmax, ymin, ymax, funcgrid, traj):

    # Calculate grid parameters 
    nx = funcgrid.shape[1]
    ny = funcgrid.shape[0]
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
     
    # Build Interpolating function 
    my_interp = scinterp.RectBivariateSpline(x, y, funcgrid.T, kx=1, ky=1)
    N = traj.shape[1]
    functraj = np.zeros(N)
    for n in range(N):
        functraj[n] = my_interp(traj[0, n], traj[1, n])
    return functraj


def traj2traj_bilinterp(xvals, yvals, funcvals, traj):
    my_interp = scinterp.SmoothBivariateSpline(xvals, yvals, funcvals, s=100, kx=1, ky=1)
    N = traj.shape[1]
    functraj = np.zeros(N)
    for n in range(N):
        functraj[n] = my_interp(traj[0, n], traj[1, n])
    return functraj

def traj2traj_kdtree(lookup_data, funcvals, traj):

    my_kd = scipy.spatial.KDTree(lookup_data)
    _, idx = my_kd.query(traj.T)
    N = traj.shape[1]
    functraj = np.zeros(N)
    for n in range(N):
        functraj[n] = funcvals[idx[n]]
    return functraj

###############################################################################
# For doing discrete carre du champ or dirichlet form 
###############################################################################
def duchamp(A, f, g, pbc_dims=None, PBC_ARG1=False, PBC_ARG2=False):
    
    # F, G are pairwise diff matrices for vectors f,g:
    # F[i, j] = f[j] - f[i]
    F = f[np.newaxis, ...] - f[:, np.newaxis, ...]
    G = g[np.newaxis, ...] - g[:, np.newaxis, ...]
    if pbc_dims is not None:
        if PBC_ARG1:
            F = periodic_restrict(F, pbc_dims)
        if PBC_ARG2:
            G = periodic_restrict(G, pbc_dims)
    if np.shape(A) == np.shape(F): 
        out = np.sum(A*F*G, axis=1).ravel()
        print("correct shape")
    else:
        out = np.dot(A, f*g) - f*np.dot(A, g) - g*np.dot(A ,f)
    # If A has row sums 0 and there's no periodic restrictions, above is equivalent to
    #print("using explicit carre du champ")
    #out = (np.dot(A, f*g) - f*np.dot(A, g) - g*np.dot(A, f))    
    return out


def dirichlet_form(A, f, g, pbc_dims=None):
    N = f.shape[0]
    return (1/N)*np.sum(duchamp(A, f, g, pbc_dims=pbc_dims)) 

###############################################################################
# Example Systems
###############################################################################
def muller_potential(x):

    # Define list of parameters
    A = np.array([-200., -100., -170., 15.])
    nu = np.array([[1., 0.], 
                   [0., 0.5], 
                   [-0.5, 1.5], 
                   [-1., 1.]])
    sig_inv = np.array([[[1., 0.],
                         [0., 10.]],
                        [[1., 0.],
                         [0., 10.]],
                        [[6.5, -5.5],
                         [-5.5, 6.5]],
                        [[-0.7, -0.3],
                         [-0.3, -0.7]]])
    V = 0.
    for i in range(4):
        u = x - nu[i, :]
        M = sig_inv[i, :, :]
        V += A[i]*np.exp(-u.dot(M.dot(u)))
    return V

def muller_drift(x):

    # Define list of parameters
    A = np.array([-200., -100., -170., 15.])
    nu = np.array([[1., 0.], 
                   [0., 0.5], 
                   [-0.5, 1.5], 
                   [-1., 1.]])
    sig_inv = np.array([[[1., 0.],
                         [0., 10.]],
                        [[1., 0.],
                         [0., 10.]],
                        [[6.5, -5.5],
                         [-5.5, 6.5]],
                        [[-0.7, -0.3],
                         [-0.3, -0.7]]])
    force = np.array([0., 0.])
    for i in range(4):
        u = x - nu[i, :]
        M = sig_inv[i, :, :]
        force += 2*A[i]*np.exp(-u.dot(M.dot(u)))*M.dot(u)
    return force

def fast_slow_drift(x, epsilon):
    r""" One sentence description
    More detailed description
    
    Parameters
    ----------
    
    Returns
    -------
    """
    out = np.array([3.0, -(1.0 / epsilon) * x[1]])
    
    return out

def simple_potential(x):
    out = 0.5*np.sum(x**2)
    return out

def simple_drift(x):
    out = -x
    return out

###############################################################################
# Periodic boundary Condition functions
###############################################################################
def pairwise_periodic_dist(X, boundary):
    """Computes pairwise distance matrix with periodic boundary

    Args:
        X (array): num samples by num features data array
        boundary (array): one dimensional array [b1,b2,...b_numfeatures] b_i is length of boundary in i-th dimension

    Returns:
     dists (array): num_samples by num samples array of pairwise periodic distances between examples in columns of X
    """
    diffs = X[np.newaxis, ...] - X[:, np.newaxis, ...]
    restrict_diffs = periodic_restrict(diffs, boundary)
    dists = np.sqrt(np.sum(restrict_diffs**2, axis=-1))
    return dists


def periodic_dist(x, y, boundary):
    """Computes eucl. distance between x and y with periodic boundary conditions

    Args:
        x ([type]): [description]
        y ([type]): [description]
        boundary ([type]): [description]

    Returns:
        [type]: [description]
    """
    diff = periodic_restrict(x - y, boundary)
    dist = np.sqrt(diff**2).sum(axis=-1)
    return dist


def periodic_restrict(x, boundary):
    """Restricts a vector x to comply with periodic boundary conditions

    Args:
        x ([type]): [description]
        boundary ([type]): [description]

    Returns:
        [type]: [description]
    """

    while (x > 0.5*boundary).any():
        x = np.where(x > 0.5*boundary, x - boundary, x) 
    while (x < -0.5*boundary).any(): 
        x = np.where(x < -0.5*boundary, x + boundary, x) 
    return x


def periodic_matrix_vec_mult(A, x, boundary):
    """Restricts a matrix-vector product to comply with periodic boundary conditions

    Args:
        A ([type]): [description]
        x ([type]): [description]

        boundary ([type]): [description]

    Returns:
        [type]: [description]
    """
    N = A.shape[0]
    out = np.zeros(N)
    for i in range(N):
        out = periodic_add(out,periodic_restrict(x[i]*A[:, i], boundary))
    return out


def periodic_add(x, y, boundary):
    y = np.where(x - y > 0.5*boundary, y - boundary, y)
    y = np.where(x - y < -0.5*boundary, y + boundary, y)
    return periodic_restrict(x + y, boundary)


if __name__ == '__main__':
    main()

def epsilon_net(data, ϵ, distance_fn=None):

    #initialize the net

    dense = True # parameter that checks whether the net is still dense
    # ϵ = 0.005
    it = 0 
    ϵ_net = np.array(range(data.shape[1]))
    current_point_index = ϵ_net[0]

    #fill the net

    while dense:
        sys.stdout.write("\r{0}".format(f'Iteration {it}: Net size = {ϵ_net.shape[0]}'))
        sys.stdout.flush()
        current_point = data[:,current_point_index] # set current point
        if distance_fn is None:
            distances = np.linalg.norm(
                data - np.tile(current_point.reshape(current_point.shape[0], 1), (1, data.shape[1])),
                axis=0,
            )
        else:
            distances = distance_fn(current_point.reshape(-1, 1), data).ravel()
        ϵ_ball = np.where(distances <= ϵ)[0] # get indices for ϵ-ball
        ϵ_net = np.delete(ϵ_net, np.where(np.isin(ϵ_net, ϵ_ball))) # kill elements from the ϵ-ball from the net
        ϵ_net = np.append(ϵ_net, current_point_index) # add the current point at the BACK OF THE QUEUE. THIS IS KEY
        current_point_index = ϵ_net[0] # set current point for killing an epsilon ball in the next iteration
        if current_point_index == 0: # if the current point is the initial one, we are done! 
            dense = False
        it += 1
    return ϵ_net, data[:,ϵ_net]

def twowell_potential(x): return model_systems.twowell_potential(x)

def twowell_density(x): return np.exp(-twowellbeta*twowell_potential(x))/twowellZ

twowellbeta = 1

beta = 1/20

Z = 214.03364928462958 # z for muller with beta = 1/20 

twowellZ = 4284.628955358415 # z for twowell with beta = 1 

def potential(x): return model_systems.muller_potential(x)

def drift(x): return model_systems.muller_drift(x)

def density(x): return np.exp(-beta*potential(x))/Z

def laplacian(x): return model_systems.muller_laplacian(x)

def laplacian_density(x): return (-beta*laplacian(x) + (beta**2)*np.sum(drift(x)**2, axis=None))*density(x)

def abs_laplacian_density(x): return np.abs(laplacian_density(x))

def plot_density(): 
  # plot density for beta = 1/20 with muller's potential
  # Setting inverse temperature for plotting
  beta = 1/20

  # Normalizing Constant for density
  xmin, xmax = -1.75, 1.5
  ymin, ymax = -0.5, 2.25
  nx, ny = 128, 128
  volume = (xmax - xmin)*(ymax - ymin)

  # Plot potential on a grid
  plt.figure()
  plot_params = [nx, ny, xmin, xmax, ymin, ymax]
  [potential_grid, xx, yy] = gen_plot_data(potential, plot_params)
  grid_min = np.min(potential_grid)
  #print("grid minimum is: %d" % grid_min)
  contour_levels = np.linspace(grid_min, 0.4, 50)
  plt.contour(xx, yy, potential_grid, levels=contour_levels)
  plt.colorbar()
  plt.xlim([xmin, xmax])
  plt.ylim([ymin, ymax])
  plt.title("Potential")

  # Gibbs Density
  plt.figure()
  plot_params = [nx, ny, xmin, xmax, ymin, ymax]
  [density_grid, xx, yy] = gen_plot_data(density, plot_params)
  grid_min = np.min(density_grid)
  grid_max = np.max(density_grid)
  #print("grid minimum is: %d" % grid_min)
  print("grid maximum is: %d" % grid_min)
  contour_levels = np.linspace(grid_min, grid_max, 1000)
  #plt.contour(xx, yy, density_grid, levels=contour_levels)
  plt.contour(xx, yy, density_grid, 500)
  plt.colorbar()
  plt.xlim([xmin, xmax])
  plt.ylim([ymin, ymax])
  plt.title("Gibbs Density")

def throwing_pts_muller(data, Vbdry):
  # Throws points away from data depending on > Vbdry condition 
  # and potential well condition 
  N = data.shape[1]
  centerx_A, centery_A, rad_A = [-0.558, 1.441, 0.1]
  centerx_B, centery_B, rad_B = [0.623, 0.028, 0.1]
  A_bool, B_bool, C_bool = is_in_ABC(data, centerx_A, centery_A, rad_A, 
                                          centerx_B, centery_B, rad_B)
  A_test = data[:, A_bool]
  B_test= data[:, B_bool]
  C_test = data[:, C_bool]
  transition_bool = np.logical_not(np.logical_or(A_bool, B_bool))
  #Vbdry = -50     # !! tunable threshold
  outliers = np.zeros(N)
  for n in range(N):
      if model_systems.muller_potential(data[:, n]) > Vbdry:
          outliers[n] = True
  # Define what points to keep and not for error
  error_bool = np.logical_not(outliers)

  # Throw away A,B points from 'error' set
  error_bool_AB = np.logical_not(np.logical_or(A_bool, B_bool)) 
  error_bool = np.logical_and(error_bool, error_bool_AB)

  throwing_bool = np.logical_not(error_bool)
  return dict([('error_bool', error_bool), 
               ('throwing_bool', throwing_bool), 
               ('A_bool', A_bool), 
               ('B_bool', B_bool), 
               ('C_bool', C_bool)])

def throwing_pts_twowell(data, Vbdry):
    # Throws points away from data depending on > Vbdry condition 
  # and potential well condition 
  N = data.shape[1]
  centerx_A, centery_A, rad_A = [-1.0 , 0., 0.15]
  centerx_B, centery_B, rad_B = [1.0, 0., 0.15]
  A_bool, B_bool, C_bool = is_in_ABC(data, centerx_A, centery_A, rad_A, 
                                          centerx_B, centery_B, rad_B)
  A_test = data[:, A_bool]
  B_test= data[:, B_bool]
  C_test = data[:, C_bool]
  transition_bool = np.logical_not(np.logical_or(A_bool, B_bool))
  #Vbdry = -50     # !! tunable threshold
  outliers = np.zeros(N)
  for n in range(N):
      if model_systems.twowell_potential(data[:, n]) > Vbdry:
          outliers[n] = True
  # Define what points to keep and not for error
  error_bool = np.logical_not(outliers)

  # Throw away A,B points from 'error' set
  error_bool_AB = np.logical_not(np.logical_or(A_bool, B_bool)) 
  error_bool = np.logical_and(error_bool, error_bool_AB)

  throwing_bool = np.logical_not(error_bool)
  return dict([('error_bool', error_bool), 
               ('throwing_bool', throwing_bool), 
               ('A_bool', A_bool), 
               ('B_bool', B_bool), 
               ('C_bool', C_bool)])

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


def alignment_rmsd_distance(X, Y):
    """
    Compute alignment-based RMSD between structures in X and Y.

    Computes RMSD after optimal rigid-body alignment using the Kabsch algorithm.
    Follows the same reshape convention as compute_aligned_sqdists in butane.py:
    flat coordinates are reshaped to (n_samples, n_atoms, 3).

    Parameters
    ----------
    X : array, shape (n_samples_X, 3*n_atoms)
        Flattened XYZ coordinates, one sample per row.
    Y : array, shape (n_samples_Y, 3*n_atoms)
        Flattened XYZ coordinates, one sample per row.

    Returns
    -------
    distances : array, shape (n_samples_X, n_samples_Y)
        Matrix of alignment-based RMSD values between all pairs.

    Notes
    -----
    - The number of features (columns) must be divisible by 3.
    - Uses Kabsch algorithm for optimal alignment before computing RMSD.
    """
    n_samples_X, n_features = X.shape
    n_samples_Y = Y.shape[0]

    # Check that features are divisible by 3
    if n_features % 3 != 0:
        raise ValueError(f"Number of features ({n_features}) must be divisible by 3 for XYZ coordinates")

    n_atoms = n_features // 3

    # Reshape to (n_samples, n_atoms, 3) -- same as compute_aligned_sqdists
    X_reshaped = X.reshape(n_samples_X, n_atoms, 3)
    Y_reshaped = Y.reshape(n_samples_Y, n_atoms, 3)

    # Compute pairwise alignment RMSD
    distances = np.zeros((n_samples_X, n_samples_Y))
    for i in range(n_samples_X):
        for j in range(n_samples_Y):
            distances[i, j] = kabsch_rmsd(X_reshaped[i], Y_reshaped[j])

    return distances

