#!/bin/bash

[ $1 = '-h' ] && echo 'Usage: bash evaluation.py time|user' && exit 1
[ $1 != 'time' ] && [ $1 != 'user' ] && echo 'Error: please type time|user for mode!' && exit 1
  
echo Run evaluation on $1 mode ... >&2

total_n=$(ls ./pkl_model/*.pkl | grep -c pkl)
echo  $total_n models are detected under ./pkl_model/ >&2

echo \# 0.2 0.5 0.8 mean train ratio in this result \#
python src/evaluation/evaluation_for_${1}_split.py


