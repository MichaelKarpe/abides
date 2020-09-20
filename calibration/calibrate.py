import logging as log
import numpy as np
import pandas as pd
import datetime as dt
import psutil
import sys

sys.path.append("../")

import optuna
from optuna.samplers import RandomSampler, TPESampler

from calibration.abides import abides
import calibration.config
from calibration.config import config

from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle
from model.LatencyModel import LatencyModel


SEED = 123456789
np.random.seed(SEED)


def oracle():
    symbols = {
        calibration.config.SECURITY: {
            "r_bar": calibration.config.R_BAR,
            "kappa": calibration.config.KAPPA,
            "fund_vol": calibration.config.THETA,
            "megashock_lambda_a": calibration.config.MEGASHOCK_LAMBDA_A,
            "megashock_mean": calibration.config.MEGASHOCK_MEAN,
            "megashock_var": calibration.config.MEGASHOCK_VAR,
            "random_state": np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)),
        }
    }

    oracle = SparseMeanRevertingOracle(
        pd.to_datetime(f"{calibration.config.DATE} {calibration.config.START_TIME}"),
        pd.to_datetime(f"{calibration.config.DATE} {calibration.config.STOP_TIME}"),
        symbols,
    )
    return oracle


def latency_model(num_agents):
    latency_rstate = np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32))

    model_args = {
        "connected": True,
        "min_latency": np.random.uniform(low=21000, high=100000, size=(num_agents, num_agents)),  # All in NYC
        "jitter": 0.3,
        "jitter_clip": 0.05,
        "jitter_unit": 5,
    }

    latency_model = LatencyModel(latency_model="cubic", random_state=latency_rstate, kwargs=model_args)
    return latency_model


def objective(trial):
    """The objective function to be optimized given parameters of the agent-based model

    :param trial: a single execution of the objective function
    :return: objective function
    """

    params = {"n_value": trial.suggest_int("n_value", 1, 100), "n_noise": trial.suggest_int("n_noise", 1, 5000)}

    # 1) get list of agents using params
    agents = config(SEED, params)

    # 2) run abides
    agents_saved_states = abides(
        name=trial.number,
        agents=agents,
        oracle=oracle(),
        latency_model=latency_model(num_agents=len(agents)),
        default_computation_delay=1000,
    )  # one millisecond

    # 3) run abides with the set of agents and properties and observe the resulting agent states
    exchange, gains = agents_saved_states[0], agents_saved_states[1:]

    return sum(gains)  # simple objective to maximize the gains of all agents


if __name__ == "__main__":

    start_time = dt.datetime.now()
    log.basicConfig(level=log.INFO)

    system_name = "  ABIDES: Calibration"
    log.info("=" * len(system_name))
    log.info(system_name)
    log.info("=" * len(system_name))
    log.info(" ")

    log.info(f"Seed: {SEED}")

    study_name = "abides_study"
    log.info(f"Study : {study_name}")

    n_trials = 100
    n_jobs = psutil.cpu_count()

    log.info(f"Number of Trials : {n_trials}")
    log.info(f"Number of Parallel Jobs : {n_jobs}")

    # sampler = RandomSampler(seed=seed)
    sampler = TPESampler(seed=SEED)  # Make the sampler behave in a deterministic way.

    # study: A study corresponds to an optimization task, i.e., a set of trials.
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(),
        storage=f"sqlite:///{study_name}.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    log.info(f"Best Parameters: {study.best_params}")
    log.info(f"Best Value: {study.best_value}")

    df = study.trials_dataframe()

    df.to_pickle(f"{study_name}_df.bz2")

    end_time = dt.datetime.now()
    log.info(f"Total time taken for the study: {end_time - start_time}")
