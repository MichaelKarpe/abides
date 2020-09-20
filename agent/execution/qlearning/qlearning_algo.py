import argparse
import pandas as pd
import numpy as np
import datetime as dt
import pickle
import sys
import os

sys.path.append(r"/efs/_abides/dev/mm/abides-dev/")

from util.model.QTable import QTable


def create_experience_df(log_folder, folder_name):
    simulation_files = []
    for file in os.listdir(log_folder):
        if folder_name in file:
            simulation_files.append(log_folder + file + "/agent_experience.bz2")
    print(f"Number of simulations: {len(simulation_files)}")

    experience_cols = ["s", "a", "s_prime", "r"]
    experience_df = pd.DataFrame(columns=experience_cols)

    for file in simulation_files:
        experience_df = experience_df.append(pd.read_pickle(file))

    experience_df = experience_df.dropna()
    experience_df = experience_df.reset_index(drop=True)

    print(f"Number of experiences: {len(experience_df)}")
    print(f"state range: {min(experience_df.s), max(experience_df.s)}")
    print(f"action range: {min(experience_df.a), max(experience_df.a)}")
    return experience_df


def q_update(num_episodes, experiences_array, q_table):

    alpha = 0.99
    alpha_decay = 0.999
    alpha_min = 0.3

    gamma = 0.98

    errors_per_episode = {}
    for episode in range(1, num_episodes + 1):

        errors_in_episode = []
        print(f"Episode: {episode}")
        print(f"Alpha: {alpha}")
        for experience in experiences_array:
            s, a, s_prime, r = experience

            q = q_table[s][a]
            td_target = r + (gamma * q_table[s_prime][np.argmax(q_table[s_prime])])

            q_table[s][a] = (1 - alpha) * q + alpha * td_target

            errors_in_episode.append(td_target - q)

        avg_td_error = np.nanmean(errors_in_episode)
        errors_per_episode[episode] = avg_td_error
        alpha *= alpha_decay
        alpha = max(alpha, alpha_min)

    return q_table, errors_per_episode


if __name__ == "__main__":

    start_time = dt.datetime.now()

    parser = argparse.ArgumentParser(description="Process The Q-Learning Agent experience tuples to create the Q-Table")

    parser.add_argument("--num_episodes", type=int, default=1, help="Number of Episodes")

    parser.add_argument(
        "--log_folder",
        default=None,
        help="Parent log folder containing the different ABIDES simulation logs with the experience df",
    )
    parser.add_argument("--folder_name", default=None, help="Child log folder name")

    args, remaining_args = parser.parse_known_args()

    num_episodes = args.num_episodes
    log_folder = args.log_folder
    folder_name = args.folder_name

    print(f"------------------------------------")
    print(f"Applying Q-Learning update on experience tuples in {log_folder}, {folder_name}")
    print(f"Number of Episodes: {num_episodes}")

    print(f"Combining Experience Tuples...")
    experience_df = create_experience_df(log_folder=log_folder, folder_name=folder_name)
    experiences_array = experience_df.values
    print(f"Experience Tuples combined...")

    q_table_shape = (200, 200, 5)
    q_table = np.zeros(shape=q_table_shape)

    q_table = q_update(num_episodes=num_episodes, experience_df=experience_df, q_table=q_table)

    q_table_abides = QTable(
        dims=q_table_shape, random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32))
    )
    q_table_abides.q = q_table

    print(f"Saving the Q-table to {log_folder}/q_table_abs_rl_state_0_actions_0_buy.bz2")
    with open(f"{log_folder}/q_table_abs_state_0_actions_0_buy.bz2", "wb") as fout:
        pickle.dump(q_table_abides, fout, protocol=-1)

    end_time = dt.datetime.now()
    print(f"Total time taken to run in parallel: {end_time - start_time}")
