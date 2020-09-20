import argparse
import os
from multiprocessing import Pool
import psutil
import datetime as dt
import numpy as np


def run_in_parallel(config, num_simulations, num_parallel, mode, agent_type,
                    q_table, fundamental_path, log_dir, verbose):
    """

    :param config: name of the configuration file
    :param num_simulations: total number of abides simulations
    :param num_parallel: number of simulations in parallel
    :param mode: train or test
    :param agent_type: baseline or rl
    :param q_table: path of the Q-table
    :param fundamental_path: path of the fundamental time series
    :param log_dir: log directory
    :param verbose: Maximum verbosity flag
    :return: None
    """

    processes = None
    global_seeds = np.random.randint(0, 2**32, num_simulations)
    print(f'Global Seeds: {global_seeds}')

    if mode == 'train':
        directions = ['BUY']
        parent_quantities = [6e3, 6e3, 9e3, 12e3]
        start_hours = [10, 11, 12, 13]
        horizon_lengths = [30, 60, 90, 120]
        freq = '1min'
        random_walk = 'random_walk_1'

        processes = [
            f'python -u abides.py -c {config} -s {global_seed} ' \
            f'--direction {direction} --parent_qty {int(parent_qty)} --start_hour {start_hour} ' \
            f'--horizon_length {horizon_length} --freq {freq} '
            f'-f {fundamental_path}/{random_walk}.bz2 ' \
            f'-m {mode} -a {agent_type} {f"-q {q_table}" if q_table else ""} ' \
            f'-l {log_dir}_{mode}_{random_walk}_{global_seed}_{direction}_{start_hour}_{horizon_length}_{int(parent_qty)} ' \
            f'{" -v" if verbose else ""}'
            for direction in directions
            for start_hour in start_hours
            for horizon_length in horizon_lengths
            for parent_qty in parent_quantities
            for global_seed in global_seeds]

    elif mode == 'test':
        direction = 'BUY'
        parent_qty = 2e3
        start_hour = 10
        horizon_length = 60
        freq = '1min'

        processes = [f'python -u abides.py -c {config} -s {global_seed} ' \
                     f'--direction {direction} --parent_qty {int(parent_qty)} --start_hour {start_hour} ' \
                     f'--horizon_length {horizon_length} --freq {freq} '
                     f'-f {fundamental_path}/random_walk_100.bz2 ' \
                     f'-m {mode} -a {agent_type} {f"-q {q_table}" if q_table else ""} ' \
                     f'-l {log_dir}_{mode}_{agent_type}_{global_seed}_q_table_rws_1_2_{direction}_{start_hour}_{horizon_length}_{int(parent_qty)}' \
                     f'{" -v" if verbose else ""}'
                     for global_seed in global_seeds]

    pool = Pool(processes=num_parallel)
    pool.map(run_process, processes)


def run_process(process):
    os.system(process)


if __name__ == "__main__":
    start_time = dt.datetime.now()

    parser = argparse.ArgumentParser(description='Main config to run multiple ABIDES simulations in parallel')

    parser.add_argument('--config', required=True, help='Name of config file to execute')
    parser.add_argument('--seed', type=int, required=False, help='Name of config file to execute')
    parser.add_argument('--num_simulations', type=int, default=1, help='Total number of simulations to run')
    parser.add_argument('--num_parallel', type=int, default=None, help='Number of processes to run in parallel')

    parser.add_argument('--mode', default=None, help='train (collect experiences) or test (use the Q-table)')
    parser.add_argument('--agent_type', default=None, help='baseline or rl agent')

    parser.add_argument('--q_table', default=None, help='Q Table file if running in test mode')

    parser.add_argument('--log_dir', default=None, help='Log directory name (default: unix timestamp at program start)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Maximum verbosity!')

    args, remaining_args = parser.parse_known_args()

    config = args.config
    seed = args.seed
    num_simulations = args.num_simulations
    num_parallel = args.num_parallel if args.num_parallel else psutil.cpu_count()

    mode = args.mode
    agent_type = args.agent_type

    q_table = args.q_table

    log_dir = args.log_dir
    verbose = args.verbose

    fundamental_path = '/efs/data/synthetic'

    np.random.seed(seed)

    print(f'Configuration: {config}')
    print(f'Seed: {seed}')
    print(f'Total number of simulation: {num_simulations}')
    print(f'Number of simulations to run in parallel: {num_parallel}')
    print(f'Mode: {mode}')
    print(f'Agent Type: {agent_type}')

    run_in_parallel(config=config,
                    num_simulations=num_simulations,
                    num_parallel=num_parallel,
                    mode=mode,
                    agent_type=agent_type,
                    q_table=q_table,
                    fundamental_path=fundamental_path,
                    log_dir=log_dir,
                    verbose=verbose)

    end_time = dt.datetime.now()
    print(f'Total time taken to run in parallel: {end_time - start_time}')