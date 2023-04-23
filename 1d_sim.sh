#!/bin/sh
python3 run/1d_bias_error.py --sample uniform --func committor --note fixedN 
python3 run/1d_bias_error.py --sample biased --func committor --note fixedN
python3 run/1d_bias_error.py --sample uniform --func trigpoly --note fixedN 
python3 run/1d_bias_error.py --sample biased --func trigpoly --note fixedN