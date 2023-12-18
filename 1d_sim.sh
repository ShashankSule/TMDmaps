#!/bin/sh
python3 run/1d_bias_error.py --sample uniform --func committor --note circular 
python3 run/1d_bias_error.py --sample biased --func committor --note circular
python3 run/1d_bias_error.py --sample uniform --func trigpoly --note circular 
python3 run/1d_bias_error.py --sample biased --func trigpoly --note circular