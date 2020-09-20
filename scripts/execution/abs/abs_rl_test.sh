#!/bin/bash

seed=987654321

config='execution.abs.execution_abs_rl'
log_dir='execution/abs/abs_rl_state_0_actions_0_buy'

mode='test'

q_table_path=${PWD}/log/execution/qtables/q_table_abs_rl_state_0_actions_0_buy_train_random_walk_1_2.bz2

# RL Agent
python -u config/execution/abs/execution_abs_rl_parallel.py \
       --config ${config} \
       --seed ${seed} \
       --num_simulations 5 \
       --num_parallel 5 \
       --mode ${mode} \
       --agent_type 'rl' \
       --q_table ${q_table_path} \
       --log_dir ${log_dir}

#Baseline Agent
python -u config/execution/abs/execution_abs_rl_parallel.py \
       --config ${config} \
       --seed ${seed} \
       --num_simulations 5 \
       --num_parallel 5 \
       --mode ${mode} \
       --agent_type 'baseline' \
       --log_dir ${log_dir}