## imports 

import os
import copy
import sys 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
# sys.path.append("..")

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
import argparse
import scipy.sparse as sps

# parallelization modules 
from math import nan
from joblib import Parallel, delayed
import multiprocess
import itertools
import tqdm

# # My Modules
import src.model_systems as model_systems
import src.helpers as helpers
import src.potentials as potentials
import src.diffusion_map as diffusion_map
from src.fem.distmesh import * 
from src.fem.FEM_TPT import *
import src.sampling as sampling
import datetime

# get args from command line 
parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=str, help="type of sampling density", default="uniform")
parser.add_argument("--func", type=str, help="Function whose Lf(0) we shall approx", default="committor")
parser.add_argument("--parallel", type=bool, help="Flag to compute in parallel or not", default=False)
parser.add_argument("--note", type=str, help="Any additional notes", default="")
args = parser.parse_args()

# set regime 
sample = args.sample
func = args.func
parallel = args.parallel
note = args.note

# set up params
beta = 1
d = 1
def potential(x): 
    return (x**2 - 1)**2 

def drift(x): 
    return 4*x*(x**2 - 1)

Z = scint.quad(lambda x: np.exp(-beta*potential(x)), -50, 50)[0]

def mu(x): return (1/Z)*np.exp(-beta*potential(x))

def f(x): return np.sin(2*np.pi*x)
def Lf(x): return -4*(np.pi**2)*np.sin(2*np.pi*x) - beta*4*x*(x**2-1)*(2*np.pi*np.cos(2*np.pi*x))
Z_committor = scint.quad(lambda x: np.exp(beta*potential(x)), -1.9,1.9)[0]

def committor(x):
    if x < -1.9: 
        return 0.0
    elif x > 1.9:
        return 1.0
    else: 
        return (1/Z_committor)*scint.quad(lambda y: np.exp(beta*potential(y)), -0.9,x)[0]

def construct_L(epsilon, target_measure, K, inds):
        r""" Construct the generator approximation corresponding to input data
        
        Parameters
        ----------
        data: array (num features, num samples)
        
        """  

        N = K.shape[-1]
        
        
        q = np.array(K.sum(axis=1)).ravel()
        q_inv = np.power(q, -1) 
        Q_inv = sps.spdiags(q_inv, 0, N, N)

        # Multiply by target measure and right-normalize
        pi = np.power(target_measure, 0.5) 
        Pi = sps.spdiags(pi, 0, N, N)
        Q_inv = Q_inv.dot(Pi)
        K_rnorm = K.dot(Q_inv)
        
        # reindex based on specification
        K_rnorm = K_rnorm[:,inds][inds,:]
        N = K_rnorm.shape[-1]
        
        print("new N: " + str(N))
        
        # Make left normalizing vector 
        q = np.array(K_rnorm.sum(axis=1)).ravel()
        q_alpha = np.power(q, -1)
        D_alpha = sps.spdiags(q_alpha, 0, N, N)
        P = D_alpha.dot(K_rnorm)

        # Transform Markov Matrix P to get discrete generator L 
        L = (P - sps.eye(N, N))/epsilon

        return L

def task(t, regime="uniform", func="committor"):
    # ϵ_uniform = epsilons_uniform[i]
    np.random.seed() 
    ϵ, _  = t
    
    # sample based on regime 
    if regime=="uniform":
        N = int(1.33*(ϵ**(-3)))  # turn on only if you intend to kick data out 
        # N = int(ϵ**(-3))
        data = np.random.uniform(-4.0,4.0,N+1).reshape(N+1,d)
    else:
        N = int(ϵ**(-5))
        data = np.random.randn(int(N+1)).reshape(N+1,d)
    
    data[N,0] = 0.0
    target_measure = np.zeros(N+1)

    for i in range(N+1):
        target_measure[i] = mu(data[i,:])

    target_dmap = diffusion_map.TargetMeasureDiffusionMap(epsilon=ϵ, n_neigh=int(N), \
                                                              target_measure=target_measure)
    target_dmap.construct_generator(data.T)
    # print("Got kernel!")
    K = target_dmap.get_kernel()
    
    # take care of boundary effect for uniform 
    if regime=="uniform":
        inds = np.where(np.abs(data) <= 3)[0]
        # L_uniform = target_dmap_uniform.get_generator() 
        L = construct_L(ϵ, target_measure, K, inds) # get generator only constructed on sample points 
        data = data[inds]
    else: 
        L = target_dmap.get_generator()

    # approx. based on function  
    if func=="committor":
        F = np.apply_along_axis(committor, 1, data) 
        Lf_atzero_TMD = L[-1,:]@F
        ans = Lf_atzero_TMD[0] 
    else: 
        F = np.apply_along_axis(f,1,data)
        Lf_atzero_TMD = L[-1,:]@F
        ans = Lf_atzero_TMD[0] - Lf(0.0)
    
    print("Result = ", ans)
    return ans

if sample=="uniform":
    # set up info 
    epsilons = np.linspace(0.04, 0.06, 10)  # actual sim 
    # epsilons = np.linspace(0.06, 0.07, 2)     # trial params for debug 
else: 
    epsilons = np.linspace(0.14, 0.18, 10) # actual sim
    # epsilons = np.linspace(0.16,0.17,2) # trial params for debug  
    


epsilons_range = len(epsilons)
ntrials = 12 # actual sim 
# ntrials = 1    # trial params for debug 
trial_ids = np.linspace(1,ntrials,ntrials)
Lpointwise_errors_TMD = np.zeros((epsilons_range, ntrials))

# now compute! 
if parallel:
    print("in parallel mode!")
    for i in tqdm.tqdm(range(epsilons_range)):
        print("Starting new epsilon...")
        ϵ = epsilons[i]
        def task_sub(x): return task([ϵ,x], regime=sample,func=func)
        with multiprocess.Pool(2) as pool: 
            result = pool.map(task_sub, list(trial_ids))
        Lpointwise_errors_TMD[i,:] = np.array(result)
else:
    for i in tqdm.tqdm(range(epsilons_range)):
        ϵ = epsilons[i]
        def task_sub(x): return task([ϵ,x], regime=sample, func=func)
        for j in tqdm.tqdm(range(ntrials)):
            Lpointwise_errors_TMD[i,j] = task_sub(trial_ids[j])

# save sim 
stats = {'epsilons': epsilons, 'pointwise_errors': Lpointwise_errors_TMD}
filename = sample + '_' + func + '_' + note + '_' + str(datetime.datetime.now())
np.save(os.getcwd()+'/run/'+filename, stats) 

