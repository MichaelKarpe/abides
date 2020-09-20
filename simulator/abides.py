import pandas as pd
import numpy as np
import datetime as dt
import logging as log

from Kernel import Kernel
import simulator.latency as latency
import simulator.abs_config as abs_config
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle


def mr_abides(name, agents, date, stop_time=None):
    """run an abides simulation in market replay mode using a list of agents and latency model.
    :param name: simulation name
    :param agents: list of agents in the Market Replay simulation
    :return: agent_states (saved states for each agent at the end of the simulation)
    """
    simulation_start_time = dt.datetime.now()

    kernel = Kernel(name, random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))

    start_time = pd.Timestamp(f"{date}")
    stop_time = stop_time + pd.to_timedelta("00:05:00")

    # TODO: computational delay for market replay
    agents_saved_states = kernel.runner(
        agents=agents,
        startTime=start_time,
        stopTime=stop_time,
        agentLatencyModel=latency.latency_model(num_agents=len(agents)),
        defaultComputationDelay=0,
        log_dir=name,
    )

    simulation_end_time = dt.datetime.now()

    log.info(f"Time taken to run simulation {name}: {simulation_end_time - simulation_start_time}")

    return agents_saved_states


def abs_abides(name, agents, date, stop_time=None):
    """run an abides simulation in agent-based simulation using a list of agents and latency model.
    :param name: simulation name
    :param agents: list of agents in the ABM simulation
    :return: agent_states (saved states for each agent at the end of the simulation)
    """
    simulation_start_time = dt.datetime.now()

    def get_oracle():
        symbols = {
            abs_config.SECURITY: {
                "r_bar": abs_config.R_BAR,
                "kappa": abs_config.KAPPA,
                "fund_vol": abs_config.THETA,
                "megashock_lambda_a": abs_config.MEGASHOCK_LAMBDA_A,
                "megashock_mean": abs_config.MEGASHOCK_MEAN,
                "megashock_var": abs_config.MEGASHOCK_VAR,
                "random_state": np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)),
            }
        }

        oracle = SparseMeanRevertingOracle(
            pd.to_datetime(f"{abs_config.DATE} {abs_config.MKT_OPEN_TIME}"),
            pd.to_datetime(f"{abs_config.DATE} {abs_config.MKT_CLOSE_TIME}"),
            symbols,
        )
        return oracle

    kernel = Kernel(name, random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))

    start_time = pd.Timestamp(f"{date}")
    stop_time = stop_time + pd.to_timedelta("00:05:00")

    agents_saved_states = kernel.runner(
        agents=agents,
        startTime=start_time,
        stopTime=stop_time,
        agentLatencyModel=latency.latency_model(num_agents=len(agents)),
        defaultComputationDelay=50,  # nanoseconds
        oracle=get_oracle(),
        log_dir=name,
    )

    simulation_end_time = dt.datetime.now()

    log.info(f"Time taken to run simulation {name}: {simulation_end_time - simulation_start_time}")

    return agents_saved_states
