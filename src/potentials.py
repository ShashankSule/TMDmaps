import os 
import numpy as np 
import scipy.io
import matplotlib.pyplot as plt
import model_systems as model_systems
import helpers as helpers
import sys

# Muller
class Muller:
    def __init__(self, target_beta, datadir):
        if target_beta != 1/20: 
            print("Temperature not equals 20 is still under construction. Setting T = 20...")
        
        self.target_beta = 1/20; # set beta 
        self.Z = 214.03364928462958 # set normalization factor 
        self.datadir = datadir # set directory for data
        self.centerx_A, self.centery_A, self.rad_A = [-0.558, 1.441, 0.1] # set centres of A 
        self.centerx_B, self.centery_B, self.rad_B = [0.623, 0.028, 0.1]  # set centres of B 
    
    def potential(self, x): 
        return model_systems.muller_potential(x) # V
 
    def drift(self, x): 
        return model_systems.muller_drift(x) # grad V
    
    def density(self, x): 
        return np.exp(-self.target_beta*self.potential(x))/self.Z

    
    def throwing_pts_muller(self, data, Vbdry):
        
          # Throws points away from data depending on > Vbdry condition 
          # and potential well condition 
          N = data.shape[1]
          A_bool, B_bool, C_bool = helpers.is_in_ABC(data, self.centerx_A, self.centery_A, self.rad_A, 
                                                  self.centerx_B, self.centery_B, self.rad_B)
          A_test = data[:, A_bool]
          B_test= data[:, B_bool]
          C_test = data[:, C_bool]
          transition_bool = np.logical_not(np.logical_or(A_bool, B_bool))
          #Vbdry = -50     # !! tunable threshold
          outliers = np.zeros(N)
          for n in range(N):
              if self.potential(data[:, n]) > Vbdry:
                  outliers[n] = True
          # Define what points to keep and not for error
          error_bool = np.logical_not(outliers)

          # Throw away A,B points from 'error' set
          error_bool_AB = np.logical_not(np.logical_or(A_bool, B_bool)) 
          error_bool = np.logical_and(error_bool, error_bool_AB)

          throwing_bool = np.logical_not(error_bool)
            
          return dict([('error_bool', error_bool), 
                       ('throwing_bool', throwing_bool), 
                       ('A_bool', A_bool), 
                       ('B_bool', B_bool), 
                       ('C_bool', C_bool)])

    def plot_density(self): 
      # plot density for beta = 1/20 with muller's potential
      # Setting inverse temperature for plotting
      beta = self.target_beta

      # Normalizing Constant for density
      xmin, xmax = -1.75, 1.5
      ymin, ymax = -0.5, 2.25
      nx, ny = 128, 128
      volume = (xmax - xmin)*(ymax - ymin)

      # Plot potential on a grid
      plt.figure()
      plot_params = [nx, ny, xmin, xmax, ymin, ymax]
      [potential_grid, xx, yy] = helpers.gen_plot_data(self.potential, plot_params)
      grid_min = np.min(potential_grid)
      #print("grid minimum is: %d" % grid_min)
      contour_levels = np.linspace(grid_min, 0.4, 50)
      plt.contour(xx, yy, potential_grid, levels=contour_levels)
      plt.colorbar()
      plt.xlim([xmin, xmax])
      plt.ylim([ymin, ymax])
      plt.title("Potential")

      # Gibbs Density
      plt.figure()
      plot_params = [nx, ny, xmin, xmax, ymin, ymax]
      [density_grid, xx, yy] = helpers.gen_plot_data(self.density, plot_params)
      grid_min = np.min(density_grid)
      grid_max = np.max(density_grid)
      #print("grid minimum is: %d" % grid_min)
      print("grid maximum is: %d" % grid_min)
      contour_levels = np.linspace(grid_min, grid_max, 1000)
      #plt.contour(xx, yy, density_grid, levels=contour_levels)
      plt.contour(xx, yy, density_grid, 500)
      plt.colorbar()
      plt.xlim([xmin, xmax])
      plt.ylim([ymin, ymax])
      plt.title("Gibbs Density")

# two well
class Twowell:
    def __init__(self, target_beta, datadir):
        if target_beta != 1: 
            print("Temperature not equals 1 is still under construction. Setting T = 20...")
        
        self.target_beta = 1; # set beta 
        self.Z = 4284.628955358415 # set partition function for beta = 1 
        self.centerx_A, self.centery_A, self.rad_A = [-1.0 , 0., 0.15] # geometry of A 
        self.centerx_B, self.centery_B, self.rad_B = [1.0, 0., 0.15] # geometry of B
        self.datadir = "/Users/shashanksule/Documents/TMDmaps/data/Twowell" # data directory, change this to run 
                                                                  # on your own machine when calling object
    def potential(self,x):
        return model_systems.twowell_potential(x) # V
    
    def drift(self, x):
        return model_systems.twowell_drift(x) # grad V
    
    def density(self, x): 
        return np.exp(-self.target_beta*self.potential(x))/self.Z # density 
    

    
    def throwing_pts_twowell(self, data, Vbdry):
        
      # Throws points away from data depending on > Vbdry condition 
      # and potential well condition 
      N = data.shape[1]
      A_bool, B_bool, C_bool = helpers.is_in_ABC(data, self.centerx_A, self.centery_A, self.rad_A, 
                                              self.centerx_B, self.centery_B, self.rad_B)
      A_test = data[:, A_bool]
      B_test= data[:, B_bool]
      C_test = data[:, C_bool]
      transition_bool = np.logical_not(np.logical_or(A_bool, B_bool))
      #Vbdry = -50     # !! tunable threshold
      outliers = np.zeros(N)
      for n in range(N):
          if self.potential(data[:, n]) > Vbdry:
              outliers[n] = True
      # Define what points to keep and not for error
      error_bool = np.logical_not(outliers)

      # Throw away A,B points from 'error' set
      error_bool_AB = np.logical_not(np.logical_or(A_bool, B_bool)) 
      error_bool = np.logical_and(error_bool, error_bool_AB)

      throwing_bool = np.logical_not(error_bool)
      return dict([('error_bool', error_bool), 
                   ('throwing_bool', throwing_bool), 
                   ('A_bool', A_bool), 
                   ('B_bool', B_bool), 
                   ('C_bool', C_bool)])
    
    def plot_density(self): 
      # plot density for beta = 1/20 with muller's potential
      # Setting inverse temperature for plotting
      beta = self.target_beta

      # Normalizing Constant for density
      xmin, xmax = -1.75, 1.5
      ymin, ymax = -0.5, 2.25
      nx, ny = 128, 128
      volume = (xmax - xmin)*(ymax - ymin)

      # Plot potential on a grid
      plt.figure()
      plot_params = [nx, ny, xmin, xmax, ymin, ymax]
      [potential_grid, xx, yy] = helpers.gen_plot_data(self.potential, plot_params)
      grid_min = np.min(potential_grid)
      #print("grid minimum is: %d" % grid_min)
      contour_levels = np.linspace(grid_min, 0.4, 50)
      plt.contour(xx, yy, potential_grid, levels=contour_levels)
      plt.colorbar()
      plt.xlim([xmin, xmax])
      plt.ylim([ymin, ymax])
      plt.title("Potential")

      # Gibbs Density
      plt.figure()
      plot_params = [nx, ny, xmin, xmax, ymin, ymax]
      [density_grid, xx, yy] = helpers.gen_plot_data(self.density, plot_params)
      grid_min = np.min(density_grid)
      grid_max = np.max(density_grid)
      #print("grid minimum is: %d" % grid_min)
      print("grid maximum is: %d" % grid_min)
      contour_levels = np.linspace(grid_min, grid_max, 1000)
      #plt.contour(xx, yy, density_grid, levels=contour_levels)
      plt.contour(xx, yy, density_grid, 500)
      plt.colorbar()
      plt.xlim([xmin, xmax])
      plt.ylim([ymin, ymax])
      plt.title("Gibbs Density")