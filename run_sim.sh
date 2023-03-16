#!/bin/sh

rootpath=$(pwd)

# Muller
fempath=$rootpath/data/Muller/ground_data/DistmeshMuller_20.mat 
savepath=$rootpath/data/Muller/error_data/

# # Twowell 
# fempath=$rootpath/data/Twowell/ground_data/DistmeshTwowell_1.mat 
# savepath=$rootpath/data/Twowell/error_data 

for var in uniform 
    do python3 src/error_analysis.py --sys muller --dset $var --tru $fempath --save $savepath 
done