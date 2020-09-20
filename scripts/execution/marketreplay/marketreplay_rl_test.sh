#!/bin/bash

seed=987654321

config='execution.marketreplay.execution_marketreplay_rl'
log_dir='execution/marketreplay/marketreplay_rl_state_0_actions_0_buy'

mode='test'

ticker='IBM'

q_table_path=${PWD}/execution/marketreplay_rl_state_0_actions_0_buy.bz2

# RL Agent
python -u config/execution_marketreplay_rl_parallel.py \
       --config ${config} \
       --seed ${seed} \
       --num_simulations 5 \
       --num_parallel 5 \
       --mode ${mode} \
       --agent_type 'rl' \
       --ticker ${ticker} \
       --q_table ${q_table_path} \
       --log_dir ${log_dir}

#Baseline Agent
python -u config/execution_marketreplay_rl_parallel.py \
       --config ${config} \
       --seed ${seed} \
       --num_simulations 5 \
       --num_parallel 5 \
       --mode ${mode} \
       --agent_type 'baseline' \
       --ticker ${ticker} \
       --log_dir ${log_dir}