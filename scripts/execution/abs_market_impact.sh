#!/bin/bash

seeds=$(seq 1 50)
random_walks=$(seq 10 20)
num_parallel_runs=90

tao=60

for seed in ${seeds[*]}
  do
  for rw in ${random_walks[*]}
    do
      sem -j${num_parallel_runs} --line-buffer \
      python -u ${PWD}/agent/execution/util/abs_market_impact.py --tao ${tao} --seed ${seed} --rw ${rw} 2>&1 > market_impact.log
    done
  done
sem --wait