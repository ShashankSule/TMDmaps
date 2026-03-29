#!/usr/bin/env python
# coding: utf-8

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
# import multiprocess
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
import simulation_helpers as sim_helpers
# create rarefied datasets
def deltafy(data, delta):
    """
    Create a rarefied dataset by removing points that are too close to each other.
    """
    # Create a mask for points that are too close to each other
    delta_net, _ = helpers.epsilon_net(data.T, delta)
    return data[delta_net, :]

# check 
conda_env = os.environ.get('CONDA_DEFAULT_ENV')
print(f"Conda Environment: {conda_env}")

# get system 
problem = "muller" # "muller", "twowell"
datadir = "/Users/shashanksule/Documents/TMDmaps/data/Muller/ground_data/DistmeshMuller_20.mat"
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
data_dict = np.load('/Users/shashanksule/Documents/TMDmaps/data/Muller/error_data/sim_feb6/muller_gibbs.npy', allow_pickle=True).item()

data = data_dict['dataset']
beta = data_dict['beta']
args = data_dict['args']
epsilons = data_dict['args']['epsilons']
deltas = data_dict['args']['deltas']
vbdry = data_dict['args']['vbry']
n_neigh = data_dict['args']['n_neigh']

datasets = [deltafy(data, delta) for delta in deltas]

args = list(itertools.product(*[epsilons, datasets, vbdry, n_neigh])) # create iterable for multiprocess
params = {"epsilons": epsilons, "deltas": deltas, "vbry": vbdry, "n_neigh": n_neigh}

pw_error, count_points, kernel_stats, verbose, error_stats = True, True, True, True, True
def onepass(t): return sim_helpers.error_data(t, system=system, weighting=weighting, pw_error=pw_error, \
                                  count_points=count_points, kernel_stats=kernel_stats, \
                                  verbose=verbose, error_stats=error_stats)