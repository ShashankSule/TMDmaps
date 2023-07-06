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
# parser.add_argument("--parallel", type=bool, help="Flag to compute in parallel or not", default=False)
parser.add_argument('--parallel', default=False, action=argparse.BooleanOptionalAction)
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
    return (4*(np.cos(x/2))**2 - 3/2)**2

def drift(x): 
    return -4*np.sin(x/2)*np.cos(x/2)*(8*(np.cos(x/2))**2 - 3)

Z = scint.quad(lambda x: np.exp(-beta*potential(x)), 0, 2*np.pi)[0]

def mu(x): return (1/Z)*np.exp(-beta*potential(x))

# define theta, r

theta_2 = 2*np.arccos(-(np.sqrt(3/2)/2))
theta_1 = 2*np.arccos((np.sqrt(3/2)/2))
r = 0.5

# define f 
def f(x): return np.sin(x)
def Lf(x): return -np.sin(x) - beta*drift(x)*np.cos(x)

# define committor 
Z_committor_12 = scint.quad(lambda x: np.exp(beta*potential(x)), theta_1+r,theta_2-r)[0]
Z_committor_21 = scint.quad(lambda x: np.exp(beta*potential(x)), theta_2+r, 2*np.pi)[0] + \
                 scint.quad(lambda x: np.exp(beta*potential(x)), 0, theta_1-r)[0]

def committor(x):
    # x = 2*np.pi*(x%1.0) # rescale to [0,2pi] interval 
    if 0.0 <= x < theta_1 - r: 
        val = (1/Z_committor_21)*scint.quad(lambda y: np.exp(beta*potential(y)), x, theta_1 - r)[0]
        return val 
    elif theta_1-r <= x < theta_1+r:
        return 0.0
    elif theta_1+r <= x < theta_2-r: 
        val = (1/Z_committor_12)*scint.quad(lambda y: np.exp(beta*potential(y)), theta_1 + r, x)[0]
        return val 
    elif theta_2-r <= x < theta_2+r: 
        return 1.0
    else: 
        val = (1/Z_committor_21)*scint.quad(lambda x: np.exp(beta*potential(x)), 0, theta_1-r)[0] + \
                    (1/Z_committor_21)*scint.quad(lambda y: np.exp(beta*potential(y)), x, 2*np.pi)[0]
        return val 

# set up sweep 

def task(t, regime="uniform", func="committor"):
    # ϵ_uniform = epsilons_uniform[i]
    np.random.seed() 
    ϵ, _  = t
    N = int(ϵ**(-3))
    # print(N)
    
    if regime=="uniform":
        data = np.random.uniform(0.0,2*np.pi,N+1).reshape(N+1,d)        
    else:
        sig = 0.1
        data = 0.5 + sig*np.random.randn(int(N+1)).reshape(N+1,d) # p = biased[N]
        data = 2*np.pi*(np.abs(data) % 1.0) 
        
    data[N,0] = np.pi
    data_embedded = np.array([np.cos(data), np.sin(data)])[:,:,0].T

    target_measure = np.zeros(N+1)

    for i in range(N+1):
        target_measure[i] = mu(data[i,:])

    target_dmap = diffusion_map.TargetMeasureDiffusionMap(epsilon=ϵ, n_neigh=int(N), \
                                                              target_measure=target_measure)
    target_dmap.construct_generator(data_embedded.T)
    # print("Got kernel!")
    K = target_dmap.get_kernel()
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
    
    # print("Result = ", ans)
    return ans

# if sample=="uniform":
#     # set up info 
#     # epsilons = np.linspace(0.04, 0.06, 10)  # actual sim 
#     # epsilons = np.linspace(0.06, 0.07, 2)     # trial params for debug 
#     # epsilons = 2.0**np.linspace(-16,4,40)
# else: 
#     # epsilons = np.linspace(0.14, 0.18, 10) # actual sim
#     # epsilons = np.linspace(0.16,0.17,2) # trial params for debug  
#     # epsilons = 2.0**np.linspace(-16,4,40)

epsilons = np.linspace(0.04, 0.06, 10)  # actual sim 
# epsilons = np.linspace(0.06, 0.07, 2)     # trial params for debug 
# epsilons = 2.0**np.linspace(-16,4,40)


epsilons_range = len(epsilons)
ntrials = 30 # actual sim 
# ntrials = 3    # trial params for debug 
trial_ids = np.linspace(1,ntrials,ntrials)
Lpointwise_errors_TMD = np.zeros((epsilons_range, ntrials))

# now compute! 
if parallel:
    print("in parallel mode!")
    for i in tqdm.tqdm(range(epsilons_range)):
        print("Starting new epsilon...")
        ϵ = epsilons[i]
        def task_sub(x): return task([ϵ,x], regime=sample,func=func)
        with multiprocess.Pool(multiprocess.cpu_count()) as pool: 
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

