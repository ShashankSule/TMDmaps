#!/bin/sh
cd Muller
python3 parallel_pass.py metadynamics
echo "Done with metadynamics!"
python3 parallel_pass.py uniform
echo "Done with uniform!"
python3 parallel_pass.py gibbs
echo "Done with gibbs!"
