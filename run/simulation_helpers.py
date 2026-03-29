
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

def error_data(t, system, weighting, \
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
            
            if weighting: 
                weights = target_measure[err_boolz['error_bool']]
                weights = weights/np.mean(weights)
            else: 
                weights = np.ones(q_tmd_error.shape)
            
            outputs.append(helpers.RMSerror(q_tmd_error, q_interpolant_fem_to_tmd_error, \
                                            weights=weights, checknans=False))

            if verbose:
                 print(outputs)
    return outputs

class simulation():
    def __init__(self, system, weighting, pw_error,count_points,kernel_stats, verbose, error_stats, **kwargs):
        """
        Initialize the simulation class with the given parameters.
        """
        self.system = system
        self.weighting = weighting
        self.pw_error = pw_error
        self.count_points = count_points
        self.kernel_stats = kernel_stats
        self.verbose = verbose
        self.error_stats = error_stats
    def onepass(self,t):
        return error_data(t, system=self.system, weighting=self.weighting, \
                          pw_error=self.pw_error, count_points=self.count_points, \
                            kernel_stats=self.kernel_stats, verbose=self.verbose, \
                                error_stats=self.error_stats)

def run_simulation(simulation, args):
    result = []
    for i in tqdm.tqdm(range(len(args))):
        ans = simulation.onepass(args[i])
        result.append(ans)
    return result

