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
args = parser.parse_args()

# set regime 
sampling = args.sample
func = args.func

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
Z_committor = scint.quad(lambda x: np.exp(beta*potential(x)), -0.9,0.9)[0]

def committor(x):
    if x < -0.9: 
        return 0.0
    elif x > 0.9:
        return 1.0
    else: 
        return (1/Z_committor)*scint.quad(lambda y: np.exp(beta*potential(y)), -0.9,x)[0]

def task(t, regime="uniform"):
    # ϵ_uniform = epsilons_uniform[i]
    np.random.seed() 
    ϵ, _  = t
    N = int(ϵ**(-3))
    
    if regime=="uniform":
        data = np.random.uniform(-2.0,2.0,N+1).reshape(N+1,d)
    else:
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
    L = target_dmap.get_generator() 
    # check committor on uniform set 
    committor_data = np.apply_along_axis(committor, 1, data)
    # LF_uniform_true = Lf(uniform)
    Lcommittor_atzero = L[-1,:]@committor_data
    print("Got result!")
    return Lcommittor_atzero[0]

if sampling=="uniform":
    # set up info 
    epsilons = np.linspace(0.03, 0.06, 10)  # actual sim 
    # epsilons = np.linspace(0.06, 0.07, 2)     # trial params for debug 
    Ns_uniform = epsilons**(-3)
    epsilons_range = len(epsilons)
    ntrials = 12 # actual sim 
    # ntrials = 3    # trial params for debug 
    trial_ids = np.linspace(1,ntrials,ntrials)
    Lcommittor_uniform_TMD = np.zeros((epsilons_range, ntrials))
    
    for i in tqdm.tqdm(range(epsilons_range)):
        print("Starting new epsilon...")
        ϵ = epsilons[i]
        def task_sub(x): return task([ϵ,x], regime="uniform")
        with multiprocess.Pool(multiprocess.cpu_count()) as pool: 
            result = pool.map(task_sub, list(trial_ids))
        Lcommittor_uniform_TMD[i,:] = np.array(result)

    # save sim 
    filename = sampling + '_' + func + '_' + str(datetime.datetime.now())
    np.save(os.getcwd()+'/run/'+filename, Lcommittor_uniform_TMD) 
else: 
    print("biased mode not written yet!")

