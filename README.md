This is a guide to reproducing the figures and data for the paper "Sharp estimates for target measure diffusion maps and applications to the committor problem." by Sule, Evans and Cameron. 

# 1D system

## Reproducing the figures 

Figures 4 & 5 for the 1D test system in Section 5.1 can be reproduced using this [Jupyter notebook](https://github.com/ShashankSule/TMDmaps/blob/pub/data/1D/1D_potential_circular.ipynb). 

Figure 6 with the results of the experiemnts for measuring the bias error prefactor can be reproduced here: [https://github.com/ShashankSule/TMDmaps/blob/pub/data/1D/1d_potential_circular_errordriver.ipynb]. 

## Generating new data and generating the figure

The simulation detailed in Section 5.1 is written in the `run/1d_bias_error.py` python script. To run this script, it is recommended to run the `1d_sim.sh` bash script which iterates over the two choices of sampling density and test function. After running the bash script, move the produced data (i.e four `.npy` files) into the `data/1D/` directory. Then change cell 7 of the `1d_potential_circular_errordriver.ipynb` notebook to account for the file names of the new data. Finally, run the notebook to produce the figure!

# Muller's potential and twowell system

## Reproducing the figures

The landscape of Muller's potential (Figure 7) can be regenerated [here](https://github.com/ShashankSule/TMDmaps/blob/pub/data/Muller/Error_analysis_muller.ipynb).

The landscape of the two well potential (Figure 10) is [here](https://github.com/ShashankSule/TMDmaps/blob/pub/data/Twowell/Error_analysis_twowell.ipynb). 

Figures 8,9 for tracking the committor RMSE over the kernel bandwidth for Muller's potential: [https://github.com/ShashankSule/TMDmaps/blob/pub/data/Muller/error_data/Error%20driver.ipynb]

Figures 8,9 for tracking the committor RMSE over the kernel bandwidth for the two well potential: [https://github.com/ShashankSule/TMDmaps/blob/pub/data/Twowell/Error_analysis_twowell.ipynb]

## Generating new data and generating the figure

The simulation detailed in section 5.2 is written in the `run/error_analysis.py` python script. To run this script, it is recommended to run the `run_sim.sh` bash script which iterates over the gibbs, metadynamics, and FEM choices of sampling density. In the `run_sim.sh` script you can change whether to run the simulation for Muller's potential or two well in the by either commenting out lines 5-7 or 11-13. After running the bash script, move the produced data (i.e three `.npy` files) into the `data/POTENTIAL/error_data/` directory. Then run the file `Error driver.ipynb` in that directory to regenerate the figures. 

# Implementing the diffusion map/sampling/FEM

The diffusion map is implemented in the `src/diffusion_map.py` module. Two sampling algorithms, Euler-Maruyama and Metadynamics are implemented in `src/sampling.py`. Finally, the FEM solutions for validation are generated using the code in the `src/FEM` folder. An example for generating the FEM solution for Muller's potential is provided in `src/FEM/mueller_TPT_driver.ipynb`. 
