import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--sys", type=str, help="either muller or twowell", default='')
parser.add_argument("--dset", type=str, help="gibbs/metadynamics/uniform", default='gibbs')
parser.add_argument("--tru", type=str, help="location of ground truth solution", default='')
parser.add_argument("--save", type=str, help="where to save error data", default='')
args = parser.parse_args()


problem = args.sys
dataset = args.dset
datadir = args.tru
savedir = args.save

print(os.getcwd()+'\n')
print(problem, dataset, datadir, savedir, sep=', ')