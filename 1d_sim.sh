#!/bin/sh
python3 run/1d_bias_error.py --sample uniform --func committor --parallel True --note fixedN 
python3 run/1d_bias_error.py --sample biased --func committor --parallel True --note fixedN
python3 run/1d_bias_error.py --sample uniform --func trigpoly --parallel True --note fixedN 
python3 run/1d_bias_error.py --sample biased --func trigpoly --parallel True --note fixedN