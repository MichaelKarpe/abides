#!/bin/bash

seed=123456789

config='execution.marketreplay.execution_marketreplay_rl'
log_dir='execution/marketreplay/marketreplay_rl_state_0_actions_0_buy'

mode='train'
agent_type='rl'

ticker='IBM'

num_simulations=1
num_parallel=20

python -u config/execution/marketreplay/execution_marketreplay_rl_parallel.py \
       --config ${config} \
       --seed ${seed} \
       --num_simulations ${num_simulations} \
       --num_parallel ${num_parallel} \
       --ticker ${ticker} \
       --mode ${mode} \
       --agent_type ${agent_type} \
       --log_dir ${log_dir}