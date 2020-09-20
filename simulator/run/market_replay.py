from multiprocessing import Pool
import logging as log
import pandas as pd
from tqdm import tqdm
import datetime as dt
import numpy as np
import sys
import psutil

sys.path.insert(0, "/efs/_abides/dev/mm/abides-rl/abides/")

import simulator.mr_config as mr_config
from simulator.abides import mr_abides
from agent.execution.rl.TabularQLearningAgent import TabularQLearningAgent
from util.model.QTable import QTable


def run_experiments(experiment):
    agent_type, D, H, T, POV = experiment.split("_")

    H = int(H)
    T = int(T)
    POV = float(POV)

    q_table_shape = (100, 100, 20, 200, 5)
    q_table = QTable(
        dims=q_table_shape,
        alpha=0.99,
        alpha_decay=0.999,
        alpha_min=0,
        epsilon=0.5,
        epsilon_decay=0.999,
        epsilon_min=0,
        gamma=0.90,
    )

    episode_num = 1
    for security, date in tqdm(TRAINING_LIST, total=len(TRAINING_LIST), desc=experiment):

        # 1) get pre-comupted trade execution schedule
        schedule, duration, freq, end_time, parent_quantity = get_schedule(security, date, H, T, POV)

        # 2) create the pool of agents using the market replay config
        background_agents, agent_count = mr_config.get_mr_config(SEED, security, date)
        experimental_agent = TabularQLearningAgent(
            id=agent_count,
            name=experiment,
            type="RLAgent",
            security=security,
            starting_cash=0,
            direction=D,
            quantity=parent_quantity,
            duration=duration,
            freq=freq,
            schedule=schedule,
            q_table=q_table,
            log_orders=False,
            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)),
        )
        agents = background_agents + [experimental_agent]

        log.info("Number of Agents: {}".format(agent_count + 1))
        log.info("Number of Background Agents: {}".format(len(background_agents)))
        log.info("Experimental Agent Type: {}".format(agent_type))

        # 3) run abides
        experiment_name = "MR_{}_EPISODE_{}_{}_{}".format(experiment, episode_num, security, date)
        agents_saved_objects = mr_abides(name=experiment_name, agents=agents, date=date, stop_time=end_time)

        q_table = agents_saved_objects[-1]
        q_table.alpha *= q_table.alpha_decay
        q_table.alpha = max(q_table.alpha, q_table.alpha_min)
        q_table.epsilon *= q_table.epsilon_decay
        q_table.epsilon = max(q_table.epsilon, q_table.epsilon_min)

        log.info("MR - Q-Table - Pct populated: {}%".format(np.count_nonzero(q_table.q) / q_table.q.size * 100))

        if episode_num % 10 == 0:
            log.info("saving Q for episode")
            pd.to_pickle(q_table.q, "{}_MR_{}_Q_TABLE_EPISODE_{}.bz2".format(LOG_FOLDER, experiment, episode_num))
        episode_num += 1

    log.info("saving Final Q")
    pd.to_pickle(q_table.q, "{}_MR_{}_Q_TABLE.bz2".format(LOG_FOLDER, experiment))


def get_schedule(security, date, H, T, pov):
    def get_full_day_transacted_volume(*args):
        security, date = args[0], args[1]

        historical_transacted_volume_whole_day_1m_bins = pd.read_pickle(
            f"/efs/data/get_real_data/marketreplay-logs/log/" f"marketreplay_{security}_{date}/transacted_volume.bz2"
        )
        return historical_transacted_volume_whole_day_1m_bins.sum()

    # 1) calculate the order parent quantity based on the historical transacted volume
    full_day_txn_volume = get_full_day_transacted_volume(security, date)
    parent_quantity = round(pov * full_day_txn_volume)

    # 2) create the execution schedule (wakeup times and quantities to trade at each wakeup)
    start_time = pd.Timestamp("{} 10:00:00".format(date))
    end_time = start_time + pd.to_timedelta(H, unit="m")
    freq = pd.Timedelta(minutes=H) / T
    duration = pd.date_range(start=start_time, end=end_time, freq=freq)
    schedule = {}
    child_quantity = int(round(parent_quantity / T))
    if child_quantity == 0:
        parent_quantity = T
        child_quantity = int(round(parent_quantity / T))
    log.info("Child Quantity: {}".format(parent_quantity))
    log.info("Child Quantity: {}".format(child_quantity))
    for t in duration:
        schedule[t] = child_quantity
    return schedule, duration, freq, end_time, parent_quantity


if __name__ == "__main__":
    script_start_time = dt.datetime.now()
    log.basicConfig(level=log.DEBUG)

    SEED = 28
    np.random.seed(SEED)

    NUM_EPISODES = 2

    SECURITIES = ["IBM"]

    DATES = [
        "20190621",
        "20190620",
        "20190619",
        "20190618",
        "20190617",
        "20190614",
        "20190613",
        "20190612",
        "20190611",
        "20190610",
        "20190607",
        "20190606",
        "20190605",
        "20190604",
        "20190603",
    ]

    LOG_FOLDER = "/efs/_abides/dev/mm/abides/log/"

    TRAINING_LIST = np.array([(security, date) for security in SECURITIES for date in DATES])
    random_choices = np.random.choice(TRAINING_LIST.shape[0], NUM_EPISODES)

    TRAINING_LIST = TRAINING_LIST[random_choices, :]
    log.info("Total Number of Training Data Days: {}".format(len(TRAINING_LIST)))

    EXPERIMENTS = ["QLEARNING_BUY_360_360_0.15"]  # Tabular Q-Learning Agent
    # Buy 0.15 * historical transacted volume, duration 360 minutes, trading every 60 seconds

    num_processes = len(EXPERIMENTS) if len(EXPERIMENTS) < psutil.cpu_count() else psutil.cpu_count()
    log.info("Total Number of Experiments: {}".format(len(EXPERIMENTS)))

    pool = Pool(processes=num_processes)

    results = pool.map(run_experiments, EXPERIMENTS)

    script_end_time = dt.datetime.now()
    log.info("Total time taken for the experiment: {}".format(script_end_time - script_start_time))
