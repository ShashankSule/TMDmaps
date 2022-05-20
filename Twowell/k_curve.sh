#!/bin/sh
python3 parallel_pass_twowell.py metadynamics
echo 'Finished metadynamics! '
python3 parallel_pass_twowell.py uniform
echo 'Finished uniform! '
python3 parallel_pass_twowell.py gibbs
echo 'Finished gibbs! '
