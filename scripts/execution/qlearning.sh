#!/bin/bash

log_folder='/efs/_abides/dev/mm/abides-dev/log/execution/abs/'
folder_name='abs_rl_state_0_actions_0_buy'

python -u agent/execution/rl/QLearningAlgorithm.py \
       --num_episodes 100  \
       --log_folder ${log_folder} \
       --folder_name ${folder_name}