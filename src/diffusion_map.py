"""
Diffusion map class, closely following ``pydiffmap'' library 
by Erik Thiede, Zofia Trstanova and Ralf Banisch, 
Github: https://github.com/DiffusionMapsAcademics/pyDiffMap/blob/master/docs/usage.rst
"""
import numpy as np 
import scipy.sparse as sps 
# from scipy.linalg.lapack import clapack as cla  # Deprecated/unavailable in newer scipy
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance as sp_dist
import scipy.linalg as sp_linalg
from tqdm import tqdm
try:
    from . import helpers as helpers
except ImportError:
    import helpers as helpers

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

def cholesky_hack(C):
    #Computes the (not necessarily unique) Cholesky decomp. for a symmetric positive SEMI-definite matrix, C = LL.T, returns L
    # NOTE: this is a bit more expensive than regular cholesky, should only be used if input matrix is likely not positive definite but it is semi-definite

    # C = MM^T, M^T = QR ---> MM^T = R^T R, so L = R^T
    M = sp_linalg.sqrtm(C)
    R = np.real(np.linalg.qr(M.T)[1])
    return R.T

class DiffusionMap(object):
    r"""
    Class for computing the diffusion map of a given data set. 
    """

    def __init__(self, alpha=0, epsilon="MAX_MIN", num_evecs=1, pbc_dims=None,
                 n_neigh=None, density=None):
        r""" Initialize diffusion map object with basic hyperparameters."""    

        self.alpha = alpha
        self.epsilon = epsilon
        self.num_evecs = num_evecs
        self.pbc_dims = pbc_dims
        self.n_neigh = n_neigh
        self.density = density
        self.flag = False

    def construct_generator(self, data):
        r""" Construct the generator approximation corresponding to input data
        
        Parameters
        ----------
        data: array (num features, num samples)
        
        """  
        K = self._construct_kernel(data)
        N = K.shape[-1]
        print("done with kernel!")

        if self.density is not None:
            q = self.density
        else:
            q = np.array(K.sum(axis=1)).ravel()
        
        # Make right normalizing vector
        q_alpha = np.power(q, -self.alpha) 
        Q_alpha = sps.spdiags(q_alpha, 0, N, N)
        K_rnorm = K.dot(Q_alpha)
        
        # Make left normalizing vector 
        q = np.array(K_rnorm.sum(axis=1)).ravel()
        q_alpha = np.power(q, -1)
        D_alpha = sps.spdiags(q_alpha, 0, N, N)
        P = D_alpha.dot(K_rnorm)
        
        # Transform Markov Matrix P to get discrete generator L 
        L = (P - sps.eye(N, N))/self.epsilon

        self.L = L
        self.K_rnorm = K_rnorm
        #print("switching L for P")
        return self
    
    def fit(self, data):
        r""" Computes the generator and diffusion map for input data

        Parameters
        ----------
        data: array (num features, num samples)

        """  
        self.construct_generator(data) 
        dmap, evecs, evals = self._construct_diffusion_coords(self.L)

        self.dmap = dmap
        self.evecs = evecs
        self.evals = evals

        return self

    def fit_transform(self, data):
        r""" Fits the data as in fit() method, and returns diffusion map 

        Parameters
        ----------
        data: array (num features, num samples)

        """  
        
        self.fit(data) 

        return self.dmap

    def _construct_kernel(self, data):
        r"""Construct kernel matrix of a given data set

        Takes an input data set with structure num_features x
        num_observations, constructs squared distance
        matrix and gaussian kernel

        Parameters
        ----------
        data: array (num features, num samples)
       
        Returns
        -------
        K : array (num samples, num samples)
          pair-wise kernel matrix K(i, j) is kernel
          applied to data points i, j

        """  
        if not self.flag:
            K = self._compute_knn_sq_dists(data)
        else: 
            K = self.sq_dists

        # Construct kernel from data matrix
        K.data = np.exp(-K.data / (self.epsilon))

        # Symmetrizing the kernel due to KNN making a non-symmetric kernel  
        K = 0.5*(K + K.T)

        self.K = K
        return K
    
    def _construct_renormalized_kernel(self, data):
        r""" Construct the renormalized kernel corresponding to input data
        
        Parameters
        ----------
        data: array (num features, num samples)
        
        """  
        K = self._construct_kernel(data)
        N = K.shape[-1]
        print("done with kernel!")

        if self.density is not None:
            q = self.density
        else:
            q = np.array(K.sum(axis=1)).ravel()
        
        # Make right normalizing vector
        q_alpha = np.power(q, -self.alpha) 
        Q_alpha = sps.spdiags(q_alpha, 0, N, N)
        K_rnorm = K.dot(Q_alpha)
        self.K_rnorm = K_rnorm
        return self

    def max_min_epsilon(self, k_alpha = 0.25):
        if not self.flag:
            assert self.sq_dists is None, "Need to compute sq_dists first"
        else: 
            K = self.sq_dists
        k = int(k_alpha*K.shape[0])
        neigh = NearestNeighbors(n_neighbors=k+1,
                                metric='precomputed')
        neigh.fit(self.sq_dists)
        [neigh_dist, neigh_ind] = neigh.kneighbors(self.sq_dists)
        max_epsilon = np.max(neigh_dist[:, k])
        min_epsilon = np.min(neigh_dist[:, 0])
        return max_epsilon, min_epsilon, k 
    
    def choose_epsilon(self):
        r""" Function for automatically choosing epsilon, work in progress

        Parameters
        ----------
        sq_dists: array (num samples, num samples)
            pair-wise squared distance matrix, entry (i, j) 
            is squared euclidean distance between data points i, j
        
        """
        if self.epsilon == "MAX_MIN":
            self.epsilon, k = self.max_min_epsilon()
            print("choosing min_max epsilon with k=%d" % k) 
        
        # otherwise, keep the epsilon value chosen by the user
        #print("epsilon = %f" % self.epsilon) 
        return self

    def ksums(self, data):
        " Compute the sum of the kernel for a range of epsilon values"
        "input: dmap: diffusion_map object, data: (num_features, num_samples) data to compute the kernel sum"
        # first check if sq_dists is computed
        if self.sq_dists is None:
            self._compute_knn_sq_dists(data)
            print("computed sq_dists!")
        eps_max, eps_min, _ = self.max_min_epsilon()
        eps_range = np.logspace(np.log2(0.25*eps_min), np.log2(4.0*eps_max), num=100, base=2)
        kernel_sums = []
        for i in tqdm(range(100)):
            self.epsilon = eps_range[i]
            K = -(1/self.epsilon)*self.sq_dists
            K = K.expm1() # compute exp(K) - 1
            kernel_sum = np.mean(K) + np.mean(np.ones(K.shape))
            kernel_sums.append(kernel_sum)
        return eps_range, kernel_sums
    # Construct a log-linearly spaced range of 100 points between (0.5 * eps_min) and eps_max

    def max_derivative(self, eps_range, kernel_sums): 
        " Compute the maximum discrete derivative of the log of the kernel sum"
        # Compute the discrete derivative of the log of the array kernel_sums
        log_kernel_sums = np.log2(kernel_sums)
        discrete_derivative = np.diff(log_kernel_sums)

        # Find the entry with the maximum discrete derivative
        max_derivative_index = np.argmax(discrete_derivative)
        max_derivative_value = discrete_derivative[max_derivative_index]
        max_eps = eps_range[max_derivative_index]
        return max_eps, max_derivative_index, max_derivative_value, \
            eps_range, discrete_derivative

    def k_sum_test(self, data):
        " Compute the sum of the kernel for a range of epsilon values"
        "input: dmap: diffusion_map object, data: (num_features, num_samples) to compute the kernel sum"
        eps_range, kernel_sums = self.ksums(data)
        max_eps, max_derivative_index, max_derivative_value, \
            eps_range, discrete_derivative = self.max_derivative(eps_range, kernel_sums)
        self.epsilon = max_eps
        return max_eps, max_derivative_index, max_derivative_value, \
            eps_range, discrete_derivative
    
    def semi_group_vals(self, data):
        if self.sq_dists is None:
            self._compute_knn_sq_dists(data)
            print("computed sq_dists!")
        eps_max, eps_min, _ = self.max_min_epsilon()
        eps_range = np.logspace(np.log2(0.25*eps_min), np.log2(4.0*eps_max), num=100, base=2)
        semi_group_vals = []
        for i in tqdm(range(100)):
            self.epsilon = eps_range[i]
            K = -(1/self.epsilon)*self.sq_dists
            K = K.expm1()+1.0
            semi_group_val = np.linalg.norm(K.multiply(K), K**2)
            semi_group_vals.append(semi_group_val)
        return eps_range, semi_group_vals
    
    def semi_group_test(self, data):
        eps_range, semi_group_vals = self.semi_group_vals(data)
        opt_eps = np.argmin(semi_group_vals)

        self.epsilon = opt_eps
        return opt_eps, eps_range, semi_group_vals


    def _construct_diffusion_coords(self, L):
        r""" Description Here

        Parameters
        ----------
        L : array, (num samples, num samples)
            Diffusion map generator matrix 

        Returns
        -------
        dmap : array, (num features, desired number of evecs)
            ith column is the ith `diffusion coordinate'
        evecs : array,  (num features, desired number of evecs)
            ith column is the ith eigenvector of generator
        evals : list, (1, desired number of evecs)
            ith entry is the ith eigval of the generator
        """    
        # Compute eigvals, eigvecs 
        print("computing eigvec matrix") 
        
        evals, evecs = sps.linalg.eigs(L, self.num_evecs + 1, which='LR')
        idx = evals.argsort()[::-1][1:]     # Ignore first eigval / eigfunc
        evals = np.real(evals[idx])
        evecs = np.real(evecs[:, idx])
        dmap = np.dot(evecs, np.diag(np.sqrt(-1./evals)))

        return dmap, evecs, evals

    def get_stationary_dist(self):
        r""" Returns the stationary distribution for the diffusion map markov chain  

        Returns
        -------
        stationary : array, (num samples, 1)
            The stationary distribution for the diffusion map markov chain  
        """    
        
        # Compute left eigvec corresponding to eigval 0
        eval, stationary = sps.linalg.eigs(self.L.T, 1, which='LR')
        stationary = np.real(stationary[:, 0])
        stationary *= np.sign(stationary[0])
        stationary = stationary / np.sum(stationary)        # normalize to 1

        return stationary

    def _compute_knn_sq_dists(self, data):
        r""" Given dataset data, computes matrix of pairwise squared distances and stores sparsely based on k - nearest neighbors

        Parameters
        ----------
        data : array, (num features, num samples)
            data matrix
    
        Returns
        -------
        knn_sq_dists : csr matrix
                       knn-sparse matrix of squared distances
        """

        ##############
        #OLD block of code. this works, but blows up for high dimensional data
        ############## 
        # Construct matrix of pairwise square distances
        #diffs = data.T[np.newaxis, ...] - data.T[:, np.newaxis, ...]
        #if self.pbc_dims is not None:
        #   # Use input pbc_dimensions for distance calculations
        #   diffs = helpers.periodic_restrict(diffs, self.pbc_dims)
        
        ## Construct nearest neighbors graph, sparsify square distances
        #sq_dists = np.sum(diffs**2, axis=-1)

        sq_dists = sp_dist.pdist(data.T, 'sqeuclidean')
        sq_dists = sp_dist.squareform(sq_dists) 

        if self.n_neigh is None:
            self.n_neigh = data.shape[1] # make dense if no param set
            knn_sq_dists = sps.csr_matrix(sq_dists)
        else:
            neigh = NearestNeighbors(n_neighbors=self.n_neigh,
                                 metric='precomputed')
            neigh.fit(sq_dists)
            self.neigh = neigh
            knn_sq_dists = neigh.kneighbors_graph(sq_dists, mode='distance')
        # Compute epsilon from square distance data
        # self.choose_epsilon(sq_dists)
         
        knn_sq_dists.sort_indices()
        self.sq_dists = knn_sq_dists 
        self.flag = True
        return knn_sq_dists
   
    @staticmethod
    def construct_committor(L, B_bool, C_bool):
        r"""Constructs the committor function w.r.t to product set A, reactant set B, C = domain \ (A U B) using the generator L

        Applies boundary conditions and restricts L to solve 
        solve Lq = 0, with q(A) = 0, q(B) = 1

        Parameters
        ----------

        L : sparse array, num data points x num data points
            generator matrix corresponding to a data set, generally the L
                matrix from diffusion maps
        B_bool : boolean vector
            indicates data indices corresponding to reactant B, same length
                as number of data points
        C_bool : boolean vector
            indicates data indices corresponding to transition region domain
                \ (A U B), same length as number of data points

        Returns
        ---------
        q : vector
            Committor function with respect to sets defined by B_bool, C_bool
        """
        Lcb = L[C_bool, :]
        Lcb = Lcb[:, B_bool]
        Lcc = L[C_bool, :]
        Lcc = Lcc[:, C_bool]

        # Assign boundary conditions for q, then solve L(C,C)q(C) = L(C,B)1
        q = np.zeros(L.shape[1])
        q[B_bool] = 1
        row_sum = np.array(np.sum(Lcb, axis=1)).ravel()
        q[C_bool] = sps.linalg.spsolve(Lcc, -row_sum)
        return q

    def get_epsilon(self):
        return self.epsilon

    def get_kernel(self):
        return self.K

    def get_generator(self):
        return self.L

class MahalanobisDiffusionMap(DiffusionMap):
    r""" 
    Class for implementing Mahalonobis diffusion maps, replacing the square distance of usual diffusion maps 
    """
    def __init__(self, alpha=0, epsilon=0.1, num_evecs=1, pbc_dims=None,
                 n_neigh=None, load_covars=None):

        super().__init__(alpha=alpha, epsilon=epsilon,
                         num_evecs=num_evecs, pbc_dims=pbc_dims,
                         n_neigh=n_neigh)
        self.load_covars = load_covars 

    def _compute_knn_sq_dists(self, data, KDE=False):
            r""" Given dataset data, computes matrix of pairwise Mahalanobis squared distances and stores sparsely based on k - nearest neighbors

            Parameters
            ----------
            data : array, (num features, num samples)
                data matrix

            Returns
            -------
            knn_sq_dists : csr matrix
                           knn-sparse matrix of squared distances
            """
            if KDE:
                # Construct matrix of pairwise square distances
                diffs = data.T[np.newaxis, ...] - data.T[:, np.newaxis, ...]
                if self.pbc_dims is not None:
                    # Use input pbc_dimensions for distance calculations
                    diffs = helpers.periodic_restrict(diffs, self.pbc_dims)
        
                sq_dists = np.sum(diffs**2, axis=-1)
            
            else:
                # Compute Mahalanobis distance matrix
                sq_dists = self._compute_mahal_sq_dists(data)

            if self.n_neigh is None:
                self.n_neigh = data.shape[1] # make dense if no param set
                knn_sq_dists = sps.csr_matrix(sq_dists)
            else:
                neigh = NearestNeighbors(n_neighbors=self.n_neigh,
                                     metric='precomputed')
                neigh.fit(sq_dists)
                self.neigh = neigh
                knn_sq_dists = neigh.kneighbors_graph(sq_dists, mode='distance')
            # Compute epsilon from square distance data
            self.choose_epsilon(sq_dists)
            knn_sq_dists.sort_indices()
            self.sq_dists = knn_sq_dists 
            
            return knn_sq_dists

    def _compute_mahal_sq_dists(self, data):
        r""" Computes matrix of pairwise mahalanobis squared distances 

        Parameters
        ----------
        data : array, (num features, num samples)
            data matrix
    
        Returns
        -------
        mahal_sq_dists : csr matrix
                       knn-sparse matrix of squared distances

        """

        dim = data.shape[0]
        N = data.shape[1]
        ################################################################### 
        # Create tensor copies of inverse cholesky matrices:
        ################################################################### 
        #    1) move axis 0,1,2 of inv_chol_covs to axis 1,2,3
        #    2) Copies inv_chol_covs N times along axis 0, then shift to axis 1:
        #          bigL[i, j, k, l] = inv_chol covs(i, k, l) for j=1,...N

        self._compute_inv_chol_covs(N, dim)
        bigL = np.broadcast_to(self.inv_chol_covs, (N, N, dim, dim))
        bigL = np.swapaxes(bigL, 0, 1)
        
        # Create block matrix of pairwise differences
        diffs = data.T[:, np.newaxis, ...] - data.T[np.newaxis, ...]

        if self.pbc_dims is not None: 
            diffs = helpers.periodic_restrict(diffs, self.pbc_dims)

        # Multiply each inverse cholseky matrix by each pairwise difference
        Ldiffs = np.einsum('ijkl,ijl->ijk', bigL, diffs)
        mahal_sq_dists = np.sum(Ldiffs**2, axis=-1)

        # Symmetrize 
        mahal_sq_dists += mahal_sq_dists.T 
        mahal_sq_dists *= 0.5 
        return mahal_sq_dists

    def _compute_inv_chol_covs(self, N, dim, data=None):        
        r""" Compute inverse cholesky factorization of input diffusion matrices

        """
        inv_chol_covs = np.zeros((N, dim, dim))
        if self.load_covars is not None:
            covars = self.load_covars
            
            for n in range(N):
                chol = self.compute_cholesky(covars[n, :, :], n)
                inv_chol_covs[n, :, :] = cla.dtrtri(chol, lower=1)[0]
            self.inv_chol_covs = inv_chol_covs
        else: 
            print("No capacity to compute covariances right now! Please upload some, this is defaulting to regular dmaps")
            if data is not None:
                # Make a list of identity matrices
                self.inv_chol_covs = np.ones((N,1,1)) * np.eye(dim)[np.newaxis, :] 
        return self
        
    @staticmethod
    def compute_cholesky(M, n=-1):
        # Error handling block of code for cholesky decomp
        try:
            chol = np.linalg.cholesky(M)
        except np.linalg.LinAlgError as err:
            if 'positive definite' in str(err):
                print(f"index {n} covar is NOT positive definite, using cholesky hack")
                chol = helpers.cholesky_hack(M)
            else:
                raise
        return chol


class TargetMeasureDiffusionMap(DiffusionMap):
    r""" 
    Class for implementing Target Measure Diffusion Maps, which computes generators with respect to an input measure
    """
    def __init__(self, target_measure, epsilon=0.1, num_evecs=1, pbc_dims=None,
                 n_neigh=None, density=None):
        super().__init__(epsilon=epsilon,
                         num_evecs=num_evecs, pbc_dims=pbc_dims,
                         n_neigh=n_neigh)
        self.target_measure = target_measure
        self.density = density

    def construct_generator(self, data):
        r""" Construct the generator approximation corresponding to input data
        
        Parameters
        ----------
        data: array (num features, num samples)
        
        """  

        K = self._construct_kernel(data)
        N = K.shape[-1]
        
        # Make right normalizing vector, first with kernel density estimate
        if self.density is not None:
            q = self.density
        else:
            q = np.array(K.sum(axis=1)).ravel()
        q_inv = np.power(q, -1) 
        Q_inv = sps.spdiags(q_inv, 0, N, N)

        # Multiply by target measure and right-normalize
        pi = np.power(self.target_measure, 0.5) 
        Pi = sps.spdiags(pi, 0, N, N)
        Q_inv = Q_inv.dot(Pi)
        K_rnorm = K.dot(Q_inv)
        
        # Make left normalizing vector 
        q = np.array(K_rnorm.sum(axis=1)).ravel()
        q_alpha = np.power(q, -1)
        D_alpha = sps.spdiags(q_alpha, 0, N, N)
        P = D_alpha.dot(K_rnorm)

        # Transform Markov Matrix P to get discrete generator L 
        L = (P - sps.eye(N, N))/self.epsilon

        self.L = L

        return self

class NeumannMap(DiffusionMap): 
    def __init__(self, alpha=0.0, epsilon=1.0, num_evecs=3, marked = False, pbc_dims=None,
                 n_neigh=None, density=None, delta=0.5):
        super().__init__(alpha=alpha, epsilon=epsilon,
                         num_evecs=num_evecs, pbc_dims=pbc_dims,
                         n_neigh=n_neigh)
        # marked = boundary, the NMap will always be computed on the unmarked pixels 
        if ~marked: 
            self.marked = marked
        else:
            self.marked = ~marked
        self.delta = delta
    
    def construct_generator(self,data,subgraph): 
        
        # compute map on unmarked points 
        if ~self.marked: 
            subgraph_inds = subgraph
        else: 
            subgraph_inds = ~subgraph
        
        # construct kernel 
        K = self._construct_kernel(data)
        
        # alpha renorm 
        D_alph_inv_vec = (K@np.ones((K.shape[1],1)).flatten())**(-self.alpha)
        D_alph_inv = sps.diags(D_alph_inv_vec)
        K = D_alph_inv@K@D_alph_inv


        # construct graph laplacian
        D = sps.diags(K@np.ones((K.shape[1],1)).flatten())
        L = D - K
        
        # construct Neumann matrix 
        B = K[~subgraph_inds, :][:, subgraph_inds] # boundary matrix 
        delta_TS_vec = B@np.ones((B.shape[1],1)).flatten()
        delta_TS_vec_inv = (1/delta_TS_vec) # deltaT_S matrix 
        delta_TS_inv = sps.diags(delta_TS_vec_inv)
        L_N = self.delta*L[subgraph_inds, :][:, subgraph_inds] - (1-self.delta)*B.T@delta_TS_inv@B # neumann laplacian 
        T_S = D.tocsr()[subgraph_inds,:][:,subgraph_inds] # degree matrix of subgraph 
        K_N = T_S - L_N # Neumann kernel matrix 
        
        # renormalize the kernel matrix
        one_over_T_S_sqrt_vec = 1/(K_N@np.ones((K_N.shape[1],1)).flatten())**(1/2)
        one_over_T_S_sqrt = sps.diags(one_over_T_S_sqrt_vec)
        renormalized_K_N = one_over_T_S_sqrt@K_N@one_over_T_S_sqrt
        T_S_sqrt_vec = (K_N@np.ones((K_N.shape[1],1)).flatten())**(1/2)
        T_S_sqrt = sps.diags(T_S_sqrt_vec)
        P_N = one_over_T_S_sqrt@renormalized_K_N@T_S_sqrt # transition matrix of reflecting random walk 
        
        generator = (P_N - sps.eye(P_N.shape[0]))/self.epsilon # generator of reflecting walk 
        
        self.L = generator
        
        return self 