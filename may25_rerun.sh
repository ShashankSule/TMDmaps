#!/bin/sh
cd Muller/
python3 parallel_pass.py uniform
echo "done with muller!"
cd .. 
cd Twowell
python3 parallel_pass_twowell.py uniform
echo "done with twowell uniform"
python3 parallel_pass_twowell.py metadynamics 
echo "done with metadynamics!"
python3 parallel_pass_twowell.py gibbs
echo "done with gibbs!"

