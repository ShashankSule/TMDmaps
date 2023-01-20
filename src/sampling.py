#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os 
import numpy as np 
import scipy.io
import matplotlib.pyplot as plt
import model_systems, helpers, potentials
from fem.distmesh import *
from fem.FEM_TPT import *

def euler_maruyama_OLD(drift, beta, dt, x0, samples=1e4, subsample=1e2):
    # Euler-Maryuama subroutine for simulating SDE X_t = drift*dt + (2β^-1)dWt 
    # inputs: 
    # dt: time step 
    # x0: initialization 
    # samples: total # of iterations 
    # samples/subsample: # of recorded iterations
    
    n = x0.shape[0] # get dims 
    sqh = np.sqrt(2*dt*(1/beta)) # step control 
    traj = np.zeros((int(samples/subsample),n))
    x = x0
    j = 0; # sampling counter
    for i in range(int(samples)):
        x = x + drift(x)*dt + sqh*np.random.randn(n)
        if i%subsample == 0: 
            traj[j,:] = x 
            j = j + 1 
    
    return traj 

def euler_maruyama_metadynamics_OLD(drift, beta, dt, x0, height, Ndeposit = int(1e4), Nbumps = int(1e2), subsample=1e2):
    
    # setup 
    n = x0.shape[0] # get dims of problem
    sqh = np.sqrt(2*dt*(1/beta)) # time step re-normalization for euler-maruyama
    samples = np.zeros((int(np.floor(Ndeposit*Nbumps/subsample)),n)) # number of samples 
    coef = np.zeros((Nbumps,1)) # magnitude of each bump 
    xbump = np.zeros((Nbumps,2)) # locations of gaussian centres 
    i=0 # subsampling counter 
    height = height.reshape(coef.shape)
    
    # iteration: dX_t = grad V(X_t) + Σ_{i=1}^{k}V_i(X_t) dt + (2β^-1) dW_t 
    for k in range(Nbumps): 

        traj =  np.zeros((Ndeposit+1,n))
        traj[0,:] = x0

        for j in range(Ndeposit):
            current_point = traj[j,:]

            # compute modified gradient 
            aux = current_point - xbump # x - x_i 
            mod_grads = aux*(np.exp(-0.5*np.linalg.norm(aux,axis=1)**2).reshape(coef.shape))
            dVbump = np.sum(coef*mod_grads, axis=0) 

            # compute drift gradient 
            dV = drift(current_point)

            # # update
            traj[j+1,:] = current_point + (dV + dVbump).reshape(n)*dt + sqh*np.random.randn(n)

            # subsample trajectory 
            if ((k-1)*Ndeposit + j)%subsample == 0:
                samples[i,:] = current_point
                i=i+1 

        # prepare for the next gaussian bump 
        x0 = traj[-1,:]
        xbump[k,:] = x0
        coef[k,:] = height[k,:]
    
    return samples 

def fem_pts(system, scaling, Vbdry):
    # system -- the relevant system 
    # scaling -- scaling parameter 
    # vbdry -- the level set for 
    
    # extract geometric parameters 
    h0 = scaling 
    xa = system.centerx_A
    ya = system.centery_A
    xb = system.centerx_B
    yb = system.centery_B
    ra = system.rad_A
    rb = system.rad_B
    xmin = system.xmin
    xmax = system.xmax
    ymin = system.ymin
    ymax = system.ymax 
    def potential(x): return np.apply_along_axis(system.potential, 1, x)
    
    # set up problem geometry 
    nx,ny= (100,100) # hardcoded for contour boundary; this is same as controlling scaling. 
    nxy = nx*ny
    x1 = np.linspace(xmin,xmax,nx)
    y1 = np.linspace(ymin,ymax,ny)
    x_grid, y_grid = np.meshgrid(x1,y1)
    x_vec = np.reshape(x_grid, (nxy,1))
    y_vec = np.reshape(y_grid, (nxy,1))
    v = np.zeros(nxy)
    xy = np.concatenate((x_vec,y_vec),axis=1)
    v = potential(xy)
    vmin = np.amin(v)
    v_grid = np.reshape(v,(nx,ny))
    # vbdry = 100 
    
    # set sets A and B and the outer boundary
    Na = int(round(2*math.pi*ra/h0))
    Nb = int(round(2*math.pi*rb/h0))
    ptsA = put_pts_on_circle(xa,ya,ra,Na)
    ptsB = put_pts_on_circle(xb,yb,rb,Nb)

    # outer boundary
    bdrydata = plt.contour(x_grid,y_grid,v_grid,[Vbdry]) # need this for the meshing
    plt.close()
    for item in bdrydata.collections:
        for i in item.get_paths():
            p_outer = i.vertices
    # reparametrize the outer boundary to make the distance 
    # between the nearest neighbor points along it approximately h0
    pts_outer = reparametrization(p_outer,h0);

    Nouter = np.size(pts_outer,axis=0)
    Nfix = Na+Nb+Nouter

    bbox = [xmin,xmax,ymin,ymax]
    pfix = np.zeros((Nfix,2))
    pfix[0:Na,:] = ptsA
    pfix[Na:Na+Nb,:] = ptsB
    pfix[Na+Nb:Nfix,:] = pts_outer

    def dfunc(p):
        d0 = potential(p)
        dA = dcircle(p,xa,ya,ra)
        dB = dcircle(p,xb,yb,rb)
        d = ddiff(d0-Vbdry,dunion(dA,dB))
        return d

    pts,_ = distmesh2D(dfunc,huniform,h0,bbox,pfix)
    
    return pts 
