# from google.colab import drive
# drive.mount('/content/drive')

import os
import copy
import sys 
# os.chdir('/content/drive/MyDrive/Colab Notebooks/Shashank')

# os.chdir('/Users/shashanksule/Documents/TMDmaps')
# Regular Modules
import numpy as np
import sklearn as sklearn
import matplotlib.pyplot as plt
import datetime
import scipy.integrate as scint
#from numpy.random import default_rng
import numpy.ma as ma
import matplotlib.tri as tri
import scipy.io
import time 
from mpl_toolkits.mplot3d import axes3d
# parallelization modules 

from math import nan
from joblib import Parallel, delayed
import multiprocessing
import itertools

# # My Modules
import helpers as helpers
import model_systems as model_systems
import diffusion_map as dmap

dataset = sys.argv[1]
print(dataset)

# Load FEM solution 
inData = scipy.io.loadmat("DistmeshTwowell_1.mat")
fem_committor = inData['committor']
fem_grid = inData['pts']

if dataset == 'metadynamics': 
        # Load gibbs density data
        inData = scipy.io.loadmat(os.getcwd() + "/ground_data/Twowell_data_metadynamics_longsample_beta_0.66.mat")
        data = inData['traj']
        data = data.T
        N = data.shape[0]
elif dataset == 'uniform':
        # Load uniform density data
        data = fem_grid.T 
        N = data.shape[0]
else: 
        # Load gibbs density data
        inData = scipy.io.loadmat("Twowell_trajectory_1.5.mat")
        data = inData['traj']
        data = data.T
        N = data.shape[0]

# get eps range 
eps_vals = 2.0**np.arange(-18, 4, 0.5)
delta_vals = np.linspace(1e-6, 1e-1, 10)[:4]
knn_vals = 2**np.arange(10,6,-5)
vbdry_vals = np.arange(1, -50, -60)
num_idx = eps_vals.shape[0]
num_delta = delta_vals.shape[0]
num_knn = knn_vals.shape[0]
num_vbdry = vbdry_vals.shape[0]
error_data_FEM_TMD = np.zeros((num_idx, num_delta, num_knn, num_vbdry))

# use this for computing kernel sums 
kernel_data = np.zeros((num_idx, num_delta, num_knn, num_vbdry))
flag = True 

def onepass(t):
        # define a pass
        i,j,k,l = t # t is a tuple of iterates 
        eps = eps_vals[i] # Current epsilon 
        delta = delta_vals[j] # Current delta net resolution 
        n_neigh = knn_vals[k] # Current knn values 
        vbdry = vbdry_vals[l] # Current vbdry term
        ϵ_net = helpers.epsilon_net(data, delta)[0] # set up \delta net
        N = np.size(ϵ_net) # number of points in net
        if n_neigh > N:
                n_neigh = N-2
        print("Computing for parameters: ", eps, delta, n_neigh, vbdry, end = "...")
        err_boolz = helpers.throwing_pts_twowell(data, vbdry) # set up error points based on vbdry 
        error_bool = err_boolz['error_bool']
        A_bool = err_boolz['A_bool']
        B_bool = err_boolz['B_bool']
        C_bool = err_boolz['C_bool']
        data_current = data[:, ϵ_net] # solve TMD for points in 
        A_bool_current = A_bool[ϵ_net]
        B_bool_current = B_bool[ϵ_net]
        C_bool_current = C_bool[ϵ_net]
        error_bool_current = error_bool[ϵ_net]
        # Compute target measure 
        def V(x): return model_systems.twowell_potential(x)
        target_measure = np.zeros(N)

        #!!!!!! define target density here, by default its to a different temperature of gibbs
        target_beta = 1.0  
        for m in range(N):
                target_measure[m] = np.exp(-target_beta*V(data_current[:, m]))/helpers.twowellbeta

        ##### Compute some sort of diffusion map 
        target_dmap = dmap.TargetMeasureDiffusionMap(epsilon=eps,
                                                n_neigh=n_neigh, target_measure=target_measure)

        # Can use regular diffusion map if the 'target measure' is the same as the gibbs density the data came from
        #reg_dmap = dmap.DiffusionMap(epsilon=eps, alpha=0.5,
        #                                          n_neigh=n_neigh, target_measure=target_measure)

        target_dmap.construct_generator(data_current)
        K = target_dmap.get_kernel()
        L = target_dmap.get_generator()
        # use this bit of code to compute kernel sum value 
        if flag: 
                return scipy.sparse.csr_matrix.mean(K)
        
        try:
                q = target_dmap.construct_committor(L, B_bool_current, C_bool_current);
        except BaseException as e: 
                print(e)
                error_data_FEM_TMD[i,j,k,l] = 1e10
        else:
                q = target_dmap.construct_committor(L, B_bool_current, C_bool_current)
                fem_error_bool = helpers.throwing_pts_twowell(fem_grid.T, vbdry)['error_bool']
                fem_committor_vbdry = fem_committor[fem_error_bool] # comment this out if you don't want error data over thrown points
                rev_interpolant = scipy.interpolate.griddata(data_current.T, q, fem_grid, method='linear')[fem_error_bool]
                rev_interpolant_refined = np.delete(rev_interpolant, np.where(np.isnan(rev_interpolant)))
                fem_interpolant_refined = np.delete(fem_committor_vbdry, np.where(np.isnan(rev_interpolant)))
                if sum(fem_error_bool) < N*0.06:
                        error_data_FEM_TMD[i,j,k,l] = 1e10
                else:
                        try:
                                error_data_FEM_TMD[i,j,k,l] = helpers.RMSerror(rev_interpolant_refined, fem_interpolant_refined)
                        except:
                                error_data_FEM_TMD[i,j,k,l] = 1e10 
                        else:
                                error_data_FEM_TMD[i,j,k,l] = helpers.RMSerror(rev_interpolant_refined, fem_interpolant_refined)
                print(error_data_FEM_TMD[i,j,k,l])

# parallelize and compute 
print("Data checks out. Computing now...")
iters = itertools.product(range(num_idx), range(num_delta), range(num_knn), range(num_vbdry))
# delayeds = [delayed(onepass)(i) for i in iters]
# num_cores = multiprocessing.cpu_count()
# parallels = Parallel(n_jobs = num_cores, require='sharedmem')
# try: 
#     outputs = parallels(delayeds)
# except BaseException as e:
#     print("error: ", e)
for i in iters:
  try:
    kernel_data[i] = onepass(i)
  except BaseException as e: 
    print("Exception: ", e)
    continue

# np.save(os.getcwd() + '/error_data/Error_data_' + dataset + '_beta_1_FEMpts_twowell.npy', error_data_FEM_TMD)
np.save(os.getcwd() + '/error_data/kernel_data_' + dataset + '.npy', kernel_data)