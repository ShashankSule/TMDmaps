#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import copy
import sys 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

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

# get args from command line 
parser = argparse.ArgumentParser()
parser.add_argument("--sys", type=str, help="either muller or twowell", default='')
parser.add_argument("--dset", type=str, help="gibbs/metadynamics/uniform", default='gibbs')
parser.add_argument("--note", type=str, help="add note", default='')
parser.add_argument("--tru", type=str, help="location of ground truth solution", default='')
parser.add_argument("--save", type=str, help="where to save error data", default='')
args = parser.parse_args()


problem = args.sys
dataset = args.dset
note = args.note 
datadir = args.tru
savedir = args.save

# first choose problem 
# problem = "muller"
# datadir = "/Users/shashanksule/Documents/TMDmaps/data/Muller/ground_data/DistmeshMueller_20.mat"
if problem == "muller":
    system = potentials.Muller(1/20, datadir)
    Vbdry = 10 
    system.plant_point = np.array([1.0, 0.0])
elif problem == "twowell":
    system = potentials.Twowell(1, datadir)
    Vbdry = 1
    system.plant_point = np.array([1.0, -0.5])
else:
    print("invalid problem")
# savedir = "/Users/shashanksule/Documents/TMDmaps/data/Muller/error_data/"

# next choose dataset params 

# dataset = "metadynamics"
x0 = np.array([0,0])
dt = 1e-4

# metadynamics params here
Nbumps = int(1e3) 
Ndeposit = int(1e3)
subsample = int(1e2)
height = 5*np.ones(Nbumps)
sig = 0.05 


# compute dataset 

if dataset == "gibbs": 
    data = sampling.euler_maruyama_OLD(system.drift, system.target_beta,dt, x0,int(1e6), int(1e2))
elif dataset == "metadynamics":
    data = sampling.euler_maruyama_metadynamics_OLD(system.drift, system.target_beta,dt, x0, height, sig,Ndeposit, Nbumps, subsample)
elif dataset == "uniform": 
    data = sampling.fem_pts(system, 0.05, Vbdry)

# visualize dataset 

# plt.scatter(data[:,0], data[:,1])


# upload fem soltuon
system.load_fem()

print("System has been set up!")

# # Run error sweep for one parameter combination 

def error_data(t, \
               pw_error=False, count_points = False, kernel_stats = False, \
               verbose = False, error_stats = True): 
    
    ϵ, data_uniformized, vbdry, n_neigh = t # unravel parameters 
    
    if verbose:
         print("Started!")
    
    if pw_error:
        data_uniformized = np.vstack((data_uniformized, system.plant_point))
    
    err_boolz = system.throwing_pts(data_uniformized.T, vbdry) # get points on data for error calculation
    fem_error_boolz = system.throwing_pts(system.qfem['pts'].T, vbdry) # get points on fem mesh for error calc.
    
    N = data_uniformized.shape[0] # get # of data points 
    outputs = []
    
    # collect # of points 
    if count_points: 
        outputs.append(N)
    
    # check knn condition 
    if n_neigh > N: 
        n_neigh = N-1
    
    # compute t.m.
    target_measure = np.zeros(N)
    for i in range(N):
        target_measure[i] = system.density(data_uniformized[i,:])
        
    # get tmdmap 
    target_dmap = diffusion_map.TargetMeasureDiffusionMap(epsilon=ϵ, n_neigh=n_neigh, \
                                                          target_measure=target_measure)
    
    # get kernel and generator
    target_dmap.construct_generator(data_uniformized.T)
    K = target_dmap.get_kernel()
    L = target_dmap.get_generator() 
     
    if pw_error: 
        # interpolate the true solution 
        q_interpolant_fem_to_tmd = scipy.interpolate.griddata(system.qfem['pts'], system.qfem['committor'], \
                                                              data_uniformized, method='linear')
        # compute L_epsilon,mu * q(x)
        inds_bool = np.isnan(q_interpolant_fem_to_tmd)

        # flash error message if the interpolation fails 
        if inds_bool[-1]:
            if verbose:
                print("failed to interpolate to plant point")
            outputs.append(nan)
        else:
            if verbose:
                print(np.sum(inds_bool))
            LL = L[np.where(~inds_bool)[0],:][:,np.where(~inds_bool)[0]]
            qq = q_interpolant_fem_to_tmd[np.where(~inds_bool)]
            Lf = LL@qq
            outputs.append(np.abs(Lf[-1]))
        
    if kernel_stats:
        
        # singer's estimate 
        outputs.append(scipy.sparse.csr_matrix.mean(K))
     
    if error_stats: 
        
        # solve committor problem
        try:
            q_tmd = target_dmap.construct_committor(L, err_boolz['B_bool'], err_boolz['C_bool'])
        except BaseException as e:
            print(e)
            outputs.append(1e10)
        else:
            if verbose:
                 print("hard part--done!")

            # checking interpolation, run this only if you want
            q_interpolant_fem_to_tmd = scipy.interpolate.griddata(system.qfem['pts'], system.qfem['committor'],\
                                                                  data_uniformized, method='linear')
            q_interpolant_tmd_to_fem = scipy.interpolate.griddata(data_uniformized, q_tmd, system.qfem['pts'], \
                                                          method='linear')

            # compute errors on fem points 
            q_fem_error = system.qfem['committor'][fem_error_boolz['error_bool']]
            q_interpolant_tmd_to_fem_error = q_interpolant_tmd_to_fem[fem_error_boolz['error_bool']].reshape(q_fem_error.shape)

            # compute errors on tmd points 
            q_tmd_error = q_tmd[err_boolz['error_bool']]
            q_interpolant_fem_to_tmd_error = q_interpolant_fem_to_tmd[err_boolz['error_bool']].reshape(q_tmd_error.shape)
            
            outputs.append(helpers.RMSerror(q_tmd_error, q_interpolant_fem_to_tmd_error, checknans=False))
        
            if verbose:
                 print(outputs)
    return outputs  


# # Playing with multiprocessing/data collection


# set up sparsification modules 

def deltanet(delta):
    δ_net, _ = helpers.epsilon_net(data.T, delta)
    return data[δ_net, :]
def uniformnet(scaling):
    return sampling.fem_pts(system, scaling, Vbdry)



# set up post-processed datasets 

num = multiprocess.cpu_count()
# deltas = list(np.linspace(1e-6, 1e-1, 10))
deltas = [0.02, 0.04]
if dataset == "uniform":
    print("Special processing for uniform data...")
    deltas = list(np.linspace(0.02, 0.05, 10))
    with multiprocess.Pool(num) as processing_pool:
        datasets = processing_pool.map(uniformnet, deltas)
else:
    with multiprocess.Pool(num) as processing_pool:
        datasets = processing_pool.map(deltanet, deltas)



# set up all the other parameters of the system 
epsilons = [2**(-5), 2**(-6), 2**(-7)]
# epsilons = list(2.0**np.arange(-16, 2, 0.25))
vbdry = [10]
n_neigh = [1024]
args = list(itertools.product(*[epsilons, datasets, vbdry, n_neigh])) # create iterable for multiprocess
params = {"epsilons": epsilons, "deltas": deltas, "vbry": vbdry, "n_neigh": n_neigh}

print("parameters are ready! Beginning error analysis...")


# run error analysis: this order is VERY important! 
count_points = True
pw_error = True
kernel_stats = True
error_stats = True

# stats for algorithm 
verbose = True
parallel = False

def onepass(t): return error_data(t,pw_error,count_points,kernel_stats, verbose, error_stats)

if parallel: 
    with multiprocess.Pool(num) as pool:
        result = pool.map(onepass, args)
else:
    result = []
    for i in tqdm.tqdm(range(len(args))):
        ans = onepass(args[i])
        result.append(ans)

# process data 
stats = [count_points, pw_error, kernel_stats, error_stats]
stat_names = np.array(["N_points", "PW_error", "singer_estimates", "error_tensor"], dtype=str)
stat_names = stat_names[stats]
sim_results = {}
for names in stat_names:
    sim_results[names] = []
for j in range(len(result)):
    for i in range (len(sim_results.items())):
        sim_results[stat_names[i]].append(result[j][i])
for name,_ in sim_results.items():
    sim_results[name] = np.array(sim_results[name]).reshape(len(epsilons), len(deltas), len(vbdry), len(n_neigh))

# write to file 
stats = {"system": problem, "sampling": dataset, "dataset": data, "beta": system.target_beta, "args": params, "sim_results": sim_results}
filename = savedir + "/" + problem + "_" + dataset + "_" + note + ".npy"
np.save(filename, stats)


# to load data:
# stats_loaded = np.load(filename, allow_pickle = True).item()
