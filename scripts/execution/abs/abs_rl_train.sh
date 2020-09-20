#!/bin/bash

seed=123456789

config='execution.abs.execution_abs_rl'
log_dir='execution/abs/abs_rl_state_0_actions_0_buy'

mode='train'
agent_type='rl'

num_simulations=1
num_parallel=70

python -u config/execution/abs/execution_abs_rl_parallel.py \
       --config ${config} \
       --seed ${seed} \
       --num_simulations ${num_simulations} \
       --num_parallel ${num_parallel} \
       --mode ${mode} \
       --agent_type ${agent_type} \
       --log_dir ${log_dir}