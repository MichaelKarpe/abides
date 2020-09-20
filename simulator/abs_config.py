import numpy as np
import pandas as pd

from agent.ExchangeAgent import ExchangeAgent

# ABS Agents:
from agent.NoiseAgent import NoiseAgent
from agent.ValueAgent import ValueAgent
from agent.market_makers.POVMarketMakerAgent import POVMarketMakerAgent
from agent.examples.MomentumAgent import MomentumAgent

from util import util
from util.order import LimitOrder

util.silent_mode = True
LimitOrder.silent_mode = True

SECURITY = "ABS"
DATE = "20200101"

MKT_OPEN_TIME = "09:30:00"
MKT_CLOSE_TIME = "16:00:00"

# (1) Noise Agents
N_NOISE = 5000
CASH_NOISE = 1e7

# (2) Value Agents
N_VALUE = 5
CASH_VALUE = 1e7

R_BAR = 1e5  # true mean fundamental value
SIGMA_N = R_BAR / 10  # observation noise variance
KAPPA = 1.67e-15  # mean reversion parameter
LAMBDA_A = 7e-11  # mean arrival rate of value agents

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

# 4) Momentum Agents:
N_MOMENTUM = 25
CASH_MOMENTUM = 1e7
MOMENTUM_MIN_SIZE = 1
MOMENTUM_MAX_SIZE = 10
MOMENTUM_WAKE_UP_FREQ = "20S"


def get_abs_config(global_seed, test=False):
    """create the list of agents for the simulation
    :param global_seed: global seed for the simulation
    """

    np.random.seed(global_seed)

    agent_count, agents, agent_types = 0, [], []

    # 1) Exchange Agent
    agents.extend(
        [
            ExchangeAgent(
                id=0,
                name="EXCHANGE_AGENT",
                type="ExchangeAgent",
                mkt_open=pd.to_datetime(f"{DATE} {MKT_OPEN_TIME}"),
                mkt_close=pd.to_datetime(f"{DATE} {MKT_CLOSE_TIME}"),
                symbols=[SECURITY],
                log_orders=test,
                pipeline_delay=0,
                computation_delay=0,
                stream_history=int(1e2),
                book_freq=0 if test else None,
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
                name="NOISE_AGENT_{}".format(j),
                type="NoiseAgent",
                symbol=SECURITY,
                starting_cash=CASH_NOISE,
                wakeup_time=util.get_wake_time(pd.Timestamp(f"{DATE} 09:00:00"), pd.Timestamp(f"{DATE} 16:00:00")),
                log_orders=test,
                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)),
            )
            for j in range(agent_count, agent_count + N_NOISE)
        ]
    )
    agent_count += N_NOISE

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
                log_orders=test,
                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)),
            )
            for j in range(agent_count, agent_count + N_VALUE)
        ]
    )
    agent_count += N_VALUE

    # 4) Market Maker Agents

    """
    window_size ==  Spread of market maker (in ticks) around the mid price
    pov == Percentage of transacted volume seen in previous `mm_wake_up_freq` that
           the market maker places at each level
    num_ticks == Number of levels to place orders in around the spread
    wake_up_freq == How often the market maker wakes up
    """
    # each elem of mm_params is tuple (window_size, pov, num_ticks, wake_up_freq)
    # mm_params = [(2, 0.02, 1, '1S'), (4, 0.01, 4, '10S'), (10, 0.005, 100, '2min')]
    # mm_params = [(8, 0.02, 4, '10S'), (12, 0.005, 100, '2min')]
    mm_params = [(5, 0.10, 5, "10S")]
    N_MARKET_MAKERS = len(mm_params)

    mm_min_order_size = 25  # Minimum size of of order placed in `transacted_volume * mm_pov` is smaller

    agents.extend(
        [
            POVMarketMakerAgent(
                id=j,
                name="POV_MARKET_MAKER_AGENT_{}".format(j),
                type="POVMarketMakerAgent",
                symbol=SECURITY,
                starting_cash=CASH_MARKET_MAKERS,
                pov=mm_params[idx][1],
                min_order_size=mm_min_order_size,
                window_size=mm_params[idx][0],
                num_ticks=mm_params[idx][2],
                wake_up_freq=mm_params[idx][3],
                log_orders=test,
                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)),
            )
            for idx, j in enumerate(range(agent_count, agent_count + N_MARKET_MAKERS))
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
                starting_cash=CASH_MOMENTUM,
                min_size=MOMENTUM_MIN_SIZE,
                max_size=MOMENTUM_MAX_SIZE,
                wake_up_freq=MOMENTUM_WAKE_UP_FREQ,
                log_orders=test,
                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)),
            )
            for j in range(agent_count, agent_count + N_MOMENTUM)
        ]
    )
    agent_count += N_MOMENTUM

    return agents, agent_count
