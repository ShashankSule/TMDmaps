"""
Diffusion map class, closely following ``pydiffmap'' library 
by Erik Thiede, Zofia Trstanova and Ralf Banisch, 
Github: https://github.com/DiffusionMapsAcademics/pyDiffMap/blob/master/docs/usage.rst
"""
import numpy as np 
import scipy.sparse as sps 
from scipy.linalg.lapack import clapack as cla
from sklearn.neighbors import NearestNeighbors

import helpers as helpers


class DiffusionMap(object):
    r"""
    Class for computing the diffusion map of a given data set. 
    """

    def __init__(self, alpha=0, epsilon="MAX_MIN", num_evecs=1, pbc_dims=None,
                 n_neigh=None, density=None):
        r""" Initialize diffusion map object with basic hyperparameters.

        Parameters
        ----------
        param : datatype
            description 

        Returns
        -------
        param : datatype
            description 
        """    
        
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_evecs = num_evecs
        self.pbc_dims = pbc_dims
        self.n_neigh = n_neigh
        self.density = density

    def _construct_kernel(self, data):
        r"""Construct kernel matrix of a given data set

        Takes an input data set with structure num_features x
        num_observations, constructs squared distance
        matrix and gaussian kernel
        
       Parameters
        ----------
        param : datatype
            description 

        Returns
        -------
        param : datatype
            description 
        """   
        
        K = self._compute_knn_sq_dists(data)

        # Construct kernel from data matrix
        K.data = np.exp(-K.data / (2*self.epsilon))
        #print("NOTE: symmetrizing KNN kernel matrix")
        K = 0.5*(K + K.T)
        #print("finished computing kernel matrix")
        self.K = K
        return K

    def choose_epsilon(self, sq_dists, k=1):
        
        if self.epsilon == "MAX_MIN":
            # TODO: this is not an efficient way of organizing this
            # If not using knn, there isn't an nearest neighbors object yet 
            if self.n_neigh is None:
                print("making the nieghbors")
                neigh = NearestNeighbors(n_neighbors=k+1,
                                 metric='precomputed')
                neigh.fit(sq_dists)
                self.neigh = neigh

            # pick epsilon so each kernel weight is approx. non-zero
            [neigh_dist, neigh_ind] = self.neigh.kneighbors(sq_dists)
            self.epsilon = np.max(neigh_dist[:, k])
            print("choosing min_max epsilon with k=%d" % k) 
        # otherwise, keep the epsilon value chosen by the user
        print("epsilon = %f" % self.epsilon) 
        return self

    def _construct_diffusion_coords(self, L):
        r""" Description Here

        Parameters
        ----------
        param : datatype
            description 

        Returns
        -------
        param : datatype
            description 
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
        r""" Description Here

        Parameters
        ----------
        param : datatype
            description 

        Returns
        -------
        param : datatype
            description 
        """    
        
        # Compute left eigvec corresponding to eigval 0
        eval, stationary = sps.linalg.eigs(self.L.T, 1, which='LR')
        stationary = np.real(stationary[:, 0])
        stationary *= np.sign(stationary[0])
        #stationary = stationary / np.sum(stationary)

        return stationary

    def _compute_knn_sq_dists(self, X):
        r""" Given dataset X, computes matrix of pairwise squared distances and stores sparsely based on k - nearest neighbors

        Parameters
        ----------
        X : array, (num features, num samples)
            data matrix
    
        Returns
        -------
        knn_sq_dists : csr matrix
                       knn-sparse matrix of squared distances
        """

        # Construct matrix of pairwise square distances
        diffs = X.T[np.newaxis, ...] - X.T[:, np.newaxis, ...]
        if self.pbc_dims is not None:
           # Use input pbc_dimensions for distance calculations
           diffs = helpers.periodic_restrict(diffs, self.pbc_dims)
        
        # Construct nearest neighbors graph, sparsify square distances
        sq_dists = np.sum(diffs**2, axis=-1)
        if self.n_neigh is None:
            self.n_neigh = X.shape[1] # make dense if no param set
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

    def construct_generator(self, data):
        r""" Construct the generator approximation corresponding to input data

        Parameters
        ----------
        param : datatype
            description 

        Returns
        -------
        param : datatype
            description 
        """  
        K = self._construct_kernel(data)
        N = K.shape[-1]
        print("done with kernel!")

        if self.density is not None:
            q = self.density
            #print("USING THE CUSTOM DENSITY")
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

        return self

    def fit(self, data):
        r""" Computes the generator and diffusion map for input data

        Parameters
        ----------
        param : datatype
            description 

        Returns
        -------
        param : datatype
            description 
        """  
        self.construct_generator(data) 
        dmap, evecs, evals = self._construct_diffusion_coords(self.L)

        self.dmap = dmap
        self.evecs = evecs
        self.evals = evals

        return self
    
    @staticmethod
    def construct_committor(L, B_bool, C_bool):
        """Constructs the committor function w.r.t to product set A, reactant set B, C = domain \ (A U B) using the generator L

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

    def fit_transform(self, data):
        r""" Fits the data as in fit() method, and returns diffusion map 

        Parameters
        ----------
        param : datatype
            description 

        Returns
        -------
        param : datatype
            description 
        """    
        self.fit(data) 

        return self.dmap

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
                 n_neigh=None, load_covars=None, load_drifts=None,
                 SYMMETRIZE=True, local_eps=None):

        super().__init__(alpha=alpha, epsilon=epsilon,
                         num_evecs=num_evecs, pbc_dims=pbc_dims,
                         n_neigh=n_neigh)
        self.load_covars = load_covars 
        self.load_drifts = load_drifts
        self.SYMMETRIZE = SYMMETRIZE
        self.local_eps = local_eps

    def _construct_kernel(self, data, KDE=False):
        r"""Construct kernel matrix of a given data set

        Takes an input data set with structure num_features x
        num_observations, constructs squared distance
        matrix and gaussian kernel

        Parameters
        ----------
        param : datatype
            description 

        Returns
        -------
        param : datatype
            description 
        """   

        # Construct kernel from data matrix
        K = self._compute_knn_sq_dists(data, KDE)

        if self.local_eps is not None and KDE is False:
            K.data = np.exp(-K.data / (2*self.local_eps))
        else:
            K.data = np.exp(-K.data / (2*self.epsilon))

        #print("NOTE: symmetrizing KNN kernel matrix")
        K = 0.5*(K + K.T)

        #print("finished computing kernel matrix")
        self.K = K
        return K
    
    def _compute_knn_sq_dists(self, X, KDE=False):
            r""" Given dataset X, computes matrix of pairwise Mahalanobis squared distances and stores sparsely based on k - nearest neighbors


            Parameters
            ----------
            X : array, (num features, num samples)
                data matrix

            Returns
            -------
            knn_sq_dists : csr matrix
                           knn-sparse matrix of squared distances
            """
            if KDE:
                # Construct matrix of pairwise square distances
                diffs = X.T[np.newaxis, ...] - X.T[:, np.newaxis, ...]
                if self.pbc_dims is not None:
                    # Use input pbc_dimensions for distance calculations
                    diffs = helpers.periodic_restrict(diffs, self.pbc_dims)
        
                sq_dists = np.sum(diffs**2, axis=-1)
            
            else:
                # Compute Mahalanobis distance matrix
                sq_dists = self._compute_mahal_sq_dists(X)

            # Construct nearest neighbors graph, sparsify square distances
            if self.n_neigh is None:
                self.n_neigh = X.shape[1] # make dense if no param set
            
            neigh = NearestNeighbors(n_neighbors=self.n_neigh,
                                     metric='precomputed')
            neigh.fit(sq_dists)
            knn_sq_dists = neigh.kneighbors_graph(sq_dists, mode='distance')
            knn_sq_dists.sort_indices()
            self.neigh = neigh

            # Compute epsilon from square distance data
            self.choose_epsilon(sq_dists)
            self.sq_dists = knn_sq_dists 
            return knn_sq_dists

    def _compute_mahal_sq_dists(self, X):
        """[summary]

        Parameters
        ----------
        X : (num features, num_observations)
            [description]
        """
        dim = X.shape[0]
        N = X.shape[1]
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
        diffs = X.T[:, np.newaxis, ...] - X.T[np.newaxis, ...]

        if self.pbc_dims is not None: 
            diffs = helpers.periodic_restrict(diffs, self.pbc_dims)

        if self.load_drifts is not None:
            # Add in drifts
            drift_factor = self.local_eps*self.load_drifts.T
            diffs += drift_factor.T[:, np.newaxis, ...]
        
        # Multiply each inverse cholseky matrix by each pairwise difference
        Ldiffs = np.einsum('ijkl,ijl->ijk', bigL, diffs)
        mahal_sq_dists = np.sum(Ldiffs**2, axis=-1)

        if self.SYMMETRIZE:
            mahal_sq_dists += mahal_sq_dists.T 
            mahal_sq_dists *= 0.5 
        return mahal_sq_dists

    def _compute_inv_chol_covs(self, N, dim, X=None):        
        """ Description Here

        Parameters
        ----------

        Returns
        -------
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
            if X is not none:
                # Make a list of identity matrices
                self.inv_chol_covs = np.ones((N,1,1)) * np.eye(dim)[np.newaxis, :] 
        return self
        
    def construct_generator(self, data):
        r""" Construct the generator approximation corresponding to input data

        Parameters
        ----------
        param : datatype
            description 

        Returns
        -------
        param : datatype
            description 
        """  
        N = data.shape[-1]
        K_local = self._construct_kernel(data)
        if not self.SYMMETRIZE: 
            K_local = (K_local + K_local.transpose())

        if self.alpha > 0:
            # Make right normalizing vector
            if self.local_eps is not None:
                K = self._construct_kernel(data, KDE=True)
                q = np.array(K.sum(axis=1)).ravel()
            else:
                q = np.array(K_local.sum(axis=1)).ravel()
            q_alpha = np.power(q, -self.alpha)
            Q_alpha = sps.spdiags(q_alpha, 0, N, N)
            K_rnorm = K_local.dot(Q_alpha)
        else:
            K_rnorm = K_local

        # Make left normalizing vector 
        q = np.array(K_rnorm.sum(axis=1)).ravel()
        q_alpha = np.power(q, -1)
        D_alpha = sps.spdiags(q_alpha, 0, N, N)
        P = D_alpha.dot(K_rnorm)

        # Transform Markov Matrix P to get discrete generator L 
        if self.local_eps is not None:
            L = (P - sps.eye(N, N))/self.local_eps
        else:
            L = (P - sps.eye(N, N))/self.epsilon

        self.K_local = K_local 
        self.L = L 

        if not self.SYMMETRIZE:
            print("testing out multipyling by 2, see symmetric kernel stuff from berry paper")
            self.L *= 2
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
        param : datatype
            description 

        Returns
        -------
        param : datatype
            description 
        """  
        K = self._construct_kernel(data)
        N = K.shape[-1]
        #print("done with kernel!")
        
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

