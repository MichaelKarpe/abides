import logging as log
import datetime as dt
import numpy as np
import pandas as pd

import sys
sys.path.append('../')

from Kernel import Kernel
import calibration.config


def abides(name, agents, oracle, latency_model, default_computation_delay):
    """ run an abides simulation using a list of agents and latency model.

    :param name: simulation name
    :param agents: list of agents in the ABM simulation
    :param props: simulation-specific properties
    :param oracle: the data oracle for the simulation

    :return: agent_states (saved states for each agent at the end of the simulation)
    """
    simulation_start_time = dt.datetime.now()

    kernel = Kernel(name,
                    random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))

    agents_saved_states = kernel.runner(agents=agents,
                                        startTime=pd.Timestamp(f'{calibration.config.DATE} {calibration.config.START_TIME}') - pd.to_timedelta('00:01:00'),
                                        stopTime=pd.Timestamp(f'{calibration.config.DATE} {calibration.config.STOP_TIME}') + pd.to_timedelta('00:01:00'),
                                        agentLatencyModel=latency_model,
                                        defaultComputationDelay=default_computation_delay,
                                        oracle=oracle,
                                        log_dir=f'calibration_log_{name}')

    simulation_end_time = dt.datetime.now()

    log.info(f"Time taken to run simulation {name}: {simulation_end_time - simulation_start_time}")

    return agents_saved_states
