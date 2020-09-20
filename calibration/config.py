# RMSC-3 (Reference Market Simulation Configuration):
# - 1     Exchange Agent
# - 1     POV Market Maker Agent
# - 100   Value Agents
# - 25    Momentum Agents
# - 5000  Noise Agents

import numpy as np
import pandas as pd

from agent.ExchangeAgent import ExchangeAgent
from agent.NoiseAgent import NoiseAgent
from agent.ValueAgent import ValueAgent
from agent.market_makers.POVMarketMakerAgent import POVMarketMakerAgent
from agent.examples.MomentumAgent import MomentumAgent

from util import util
from util.order import LimitOrder

util.silent_mode = True
LimitOrder.silent_mode = True

SECURITY = "ABS"
DATE = "20190628"
START_TIME = "09:30:00"
STOP_TIME = "16:00:00"

N_NOISE = 5000
CASH_NOISE = 1e7

N_VALUE = 100
CASH_VALUE = 1e7

R_BAR = 1e5  # true mean fundamental value
SIGMA_N = R_BAR / 10  # observation noise variance
KAPPA = 1.67e-12  # mean reversion parameter
LAMBDA_A = 1e-12  # 7e-11      # mean arrival rate of value agents

# OU process
MU = R_BAR
GAMMA = KAPPA
THETA = 1e-4

MEGASHOCK_MEAN = 1e3
MEGASHOCK_VAR = 5e4
MEGASHOCK_LAMBDA_A = 2.77778e-13

# 3) Market Maker Agents:
N_MARKET_MAKERS = 1
CASH_MARKET_MAKERS = 1e7
MARKET_MAKERS_MIN_SIZE = 100
MARKET_MAKERS_MAX_SIZE = 101

MM_WAKE_UP_FREQ = "10S"  # How often the market maker wakes up
MM_POV = 0.05  # Percentage of transacted volume seen in previous `mm_wake_up_freq` that
# the market maker places at each level
MM_MIN_ORDER_SIZE = 25  # Minimum size of of order placed in `transacted_volume * mm_pov` is smaller
MM_WINDOW_SIZE = 5  # Spread of market maker (in ticks) around the mid price
MM_NUM_TICKS = 50  # Number of levels to place orders in around the spread

# 4) Momentum Agents:
N_MOMENTUM = 25
CASH_MOMENTUM = 1e7
MOMENTUM_MIN_SIZE = 1
MOMENTUM_MAX_SIZE = 10
MOMENTUM_WAKE_UP_FREQ = "20S"


def config(seed, params):
    """create the list of agents for the simulation

    :param params: abides config parameters
    :return: list of agents given a set of parameters
    """

    np.random.seed(seed)

    agent_count, agents, agent_types = 0, [], []

    # 1) Exchange Agent
    agents.extend(
        [
            ExchangeAgent(
                id=0,
                name="EXCHANGE_AGENT",
                type="ExchangeAgent",
                mkt_open=pd.to_datetime(f"{DATE} {START_TIME}"),
                mkt_close=pd.to_datetime(f"{DATE} {STOP_TIME}"),
                symbols=[SECURITY],
                log_orders=True,
                pipeline_delay=0,
                computation_delay=0,
                stream_history=int(1e5),
                book_freq=0,
                wide_book=True,
                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)),
            )
        ]
    )
    agent_count += 1

    # 2) Noise Agents
    agents.extend(
        [
            NoiseAgent(
                id=j,
                name="NoiseAgent {}".format(j),
                type="NoiseAgent",
                symbol=SECURITY,
                starting_cash=CASH_NOISE,
                wakeup_time=util.get_wake_time(pd.Timestamp(f"{DATE} 09:00:00"), pd.Timestamp(f"{DATE} 16:00:00")),
                log_orders=False,
                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)),
            )
            for j in range(agent_count, agent_count + params["n_noise"])
        ]
    )
    agent_count += params["n_noise"]

    # 3) Value Agents
    agents.extend(
        [
            ValueAgent(
                id=j,
                name="Value Agent {}".format(j),
                type="ValueAgent",
                symbol=SECURITY,
                starting_cash=CASH_VALUE,
                sigma_n=SIGMA_N,
                r_bar=R_BAR,
                kappa=KAPPA,
                lambda_a=LAMBDA_A,
                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)),
            )
            for j in range(agent_count, agent_count + params["n_value"])
        ]
    )
    agent_count += params["n_value"]

    # 4) Market Maker Agent
    agents.extend(
        [
            POVMarketMakerAgent(
                id=j,
                name="POV_MARKET_MAKER_AGENT_{}".format(j),
                type="POVMarketMakerAgent",
                symbol=SECURITY,
                starting_cash=CASH_MARKET_MAKERS,
                pov=MM_POV,
                min_order_size=MM_MIN_ORDER_SIZE,
                window_size=MM_WINDOW_SIZE,
                num_ticks=MM_NUM_TICKS,
                wake_up_freq=MM_WAKE_UP_FREQ,
                log_orders=False,
                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)),
            )
            for j in range(agent_count, agent_count + N_MARKET_MAKERS)
        ]
    )
    agent_count += N_MARKET_MAKERS

    # 5) Momentum Agents
    agents.extend(
        [
            MomentumAgent(
                id=j,
                name="MOMENTUM_AGENT_{}".format(j),
                type="MomentumAgent",
                symbol=SECURITY,
                starting_cash=CASH_MARKET_MAKERS,
                min_size=MOMENTUM_MIN_SIZE,
                max_size=MOMENTUM_MAX_SIZE,
                wake_up_freq=MOMENTUM_WAKE_UP_FREQ,
                log_orders=False,
                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)),
            )
            for j in range(agent_count, agent_count + N_MOMENTUM)
        ]
    )
    agent_count += N_MOMENTUM

    return agents
