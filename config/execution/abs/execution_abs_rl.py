import argparse
import numpy as np
import pandas as pd
import sys
import datetime as dt
import pickle

from agent.ExchangeAgent import ExchangeAgent

from agent.NoiseAgent import NoiseAgent
from agent.ValueAgent import ValueAgent
from agent.market_makers.MarketMakerAgent import MarketMakerAgent
from agent.examples.MomentumAgent import MomentumAgent

from agent.execution.baselines.twap_agent import TWAPExecutionAgent
from agent.execution.qlearning.qlearning_twap_agent import QLearningTWAPAgent

from Kernel import Kernel
from util import util
from util.order import LimitOrder
from util.oracle.ExternalFileOracle import ExternalFileOracle

sys.path.append(r"..")

########################################################################################################################
############################################### GENERAL CONFIG #########################################################

simulation_start_time = dt.datetime.now()

parser = argparse.ArgumentParser(description="Detailed options for market replay config.")

parser.add_argument("-c", "--config", required=True, help="Name of config file to execute")
parser.add_argument("-s", "--seed", type=int, default=None, help="numpy.random.seed() for simulation")

parser.add_argument("--direction", default=None, help="BUY or SELL")
parser.add_argument("--parent_qty", type=int, default=None, help="Total Size of the parent order")
parser.add_argument("--start_hour", type=int, default=None, help="Time (in hours) to start executing the parent order")
parser.add_argument(
    "--horizon_length", type=int, default=None, help="Length of the execution time horizon (in minutes)"
)
parser.add_argument("--freq", default=None, help="Frequency of Order Placement")

parser.add_argument("-f", "--fundamental-file-path", required=True, help="Path to external fundamental file.")

parser.add_argument("-m", "--mode", default=None, help="train (collect experiences) or test (use the Q-table)")
parser.add_argument("-a", "--agent_type", default=None, help="baseline or rl agent")

parser.add_argument("-q", "--q_table", default=None, help="Q Table file")


parser.add_argument(
    "-l", "--log_dir", default=None, help="Log directory name (default: unix timestamp at program start)"
)
parser.add_argument("-v", "--verbose", action="store_true", help="Maximum verbosity!")
parser.add_argument("--config_help", action="store_true", help="Print argument options for this config file")

args, remaining_args = parser.parse_known_args()

if args.config_help:
    parser.print_help()
    sys.exit()

seed = args.seed  # Random seed specification on the command line.
if not seed:
    seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
np.random.seed(seed)

util.silent_mode = not args.verbose
LimitOrder.silent_mode = not args.verbose

print("Mode: {}".format(args.mode))
print("Simulation Start Time: {}".format(simulation_start_time))
print("Configuration seed: {}".format(seed))
print("Fundamental File Path: {}".format(args.fundamental_file_path))

######################## Agents Config #########################################################################


def add_background_agents(agent_count, agents, agent_types, agent_seeds, oracle, symbol):
    # Hyperparameters
    starting_cash = 10000000  # Cash in this simulator is always in CENTS.

    r_bar = 1e5
    sigma_n = r_bar / 10
    kappa = 1.67e-15
    lambda_a = 7e-11

    symbols = {
        symbol: {
            "r_bar": r_bar,
            "kappa": 1.67e-12,
            "agent_kappa": kappa,
            "sigma_s": 0,
            "fund_vol": 1e-4,
            "megashock_lambda_a": 2.77778e-13,
            "megashock_mean": 1e3,
            "megashock_var": 5e4,
            "random_state": np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)),
        }
    }

    # Noise Agents
    num_noise = 5000
    agents.extend(
        [
            NoiseAgent(
                id=j,
                name="NOISE_AGENT_{}".format(j),
                type="NoiseAgent",
                symbol=symbol,
                starting_cash=starting_cash,
                wakeup_time=util.get_wake_time(mkt_open, mkt_close),
                random_state=util.get_rand_obj(agent_seeds[j]),
            )
            for j in range(agent_count, agent_count + num_noise)
        ]
    )
    agent_count += num_noise
    agent_types.extend(["NoiseAgent"])

    # Value Agents
    num_value = 100
    agents.extend(
        [
            ValueAgent(
                id=j,
                name="VALUE_AGENT_{}".format(j),
                type="ValueAgent",
                symbol=symbol,
                starting_cash=starting_cash,
                sigma_n=sigma_n,
                r_bar=r_bar,
                kappa=kappa,
                lambda_a=lambda_a,
                random_state=util.get_rand_obj(agent_seeds[j]),
            )
            for j in range(agent_count, agent_count + num_value)
        ]
    )
    agent_count += num_value
    agent_types.extend(["ValueAgent"])

    # Market Maker Agent
    num_mm_agents = 1
    agents.extend(
        [
            MarketMakerAgent(
                id=j,
                name="MARKET_MAKER_AGENT_{}".format(j),
                type="MarketMakerAgent",
                symbol=symbol,
                starting_cash=starting_cash,
                min_size=500,
                max_size=1000,
                wake_up_freq="1min",
                random_state=util.get_rand_obj(agent_seeds[j]),
            )
            for j in range(agent_count, agent_count + num_mm_agents)
        ]
    )
    agent_count += num_mm_agents
    agent_types.extend("MarketMakerAgent")

    # Momentum Agents
    num_momentum_agents = 25
    agents.extend(
        [
            MomentumAgent(
                id=j,
                name="MOMENTUM_AGENT_{}".format(j),
                type="MomentumAgent",
                symbol=symbol,
                starting_cash=starting_cash,
                min_size=1,
                max_size=10,
                random_state=util.get_rand_obj(agent_seeds[j]),
            )
            for j in range(agent_count, agent_count + num_momentum_agents)
        ]
    )
    agent_count += num_momentum_agents
    agent_types.extend("MomentumAgent")
    return agent_count, agents, agent_types


# symbol
symbol = "ABS"

# Historical date to simulate.
date = "20200101"
historical_date_pd = pd.to_datetime(date)
mkt_open = historical_date_pd + pd.to_timedelta("09:00:00")
mkt_close = historical_date_pd + pd.to_timedelta("16:00:00")

agent_count, agents, agent_types = 0, [], []

num_agents = 5128
agent_seeds = [np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32))] * num_agents

# 1) Exchange Agent
agents.extend(
    [
        ExchangeAgent(
            id=0,
            name="EXCHANGE_AGENT",
            type="ExchangeAgent",
            mkt_open=mkt_open,
            mkt_close=mkt_close,
            symbols=[symbol],
            pipeline_delay=0,
            computation_delay=0,
            stream_history=10,
            book_freq=None,
            log_orders=False,
            random_state=util.get_rand_obj(agent_seeds[0]),
        )
    ]
)
agent_types.extend("ExchangeAgent")
agent_count += 1

# 2) Background Agents:

# Oracle
symbols = {
    symbol: {
        "fundamental_file_path": args.fundamental_file_path,
        "random_state": np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")),
    }
}
oracle = ExternalFileOracle(symbols)

agent_count, agents, agent_types = add_background_agents(agent_count, agents, agent_types, agent_seeds, oracle, symbol)

random_state = util.get_rand_obj(agent_seeds[agent_count])

direction = args.direction
parent_qty = args.parent_qty

start_time = pd.Timestamp(f"{date} {args.start_hour}:00:00")
end_time = pd.Timestamp(f"{date} {args.start_hour}:00:00") + pd.Timedelta(minutes=args.horizon_length)
freq = args.freq

execution_time_horizon = pd.date_range(start=start_time, end=end_time, freq=freq)

if args.agent_type == "rl":
    # RL Execution Agent Config
    def unpickle_q_table(file_name):
        with open(file_name, "rb") as fin:
            q_table = pickle.load(fin)
        return q_table

    q_table = unpickle_q_table(args.q_table) if args.mode == "test" else None

    q_learning_agent = QLearningTWAPAgent(
        id=agent_count,
        name="TWAPExecutionAgent",
        type="TWAPExecutionAgent",
        symbol=symbol,
        starting_cash=0,
        execution_time_horizon=execution_time_horizon,
        freq=freq,
        direction=direction,
        quantity=parent_qty,
        mode=args.mode,
        q_table=q_table,
        log_orders=False if args.mode == "train" else True,
        trade=True,
        random_state=random_state,
    )
    execution_agents = [q_learning_agent]
    agents.extend(execution_agents)
    agent_types.extend("RLExecutionAgent")
    agent_count += 1

elif args.agent_type == "baseline":

    twap_agent = TWAPExecutionAgent(
        id=agent_count,
        name="TWAPExecutionAgent",
        type="TWAPExecutionAgent",
        symbol=symbol,
        starting_cash=0,
        execution_time_horizon=execution_time_horizon,
        freq=freq,
        direction=direction,
        quantity=parent_qty,
        log_orders=False if args.mode == "train" else True,
        trade=True,
        random_state=random_state,
    )

    execution_agents = [twap_agent]
    agents.extend(execution_agents)
    agent_types.extend("RLTWAPExecutionAgent")
    agent_count += 1

########################################################################################################################
########################################### KERNEL AND OTHER CONFIG ####################################################

kernel = Kernel(
    "Market Replay Kernel",
    random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype="uint64")),
)

kernelStartTime = historical_date_pd
kernelStopTime = end_time + pd.Timedelta("10min")

defaultComputationDelay = 0
latency = np.zeros((agent_count, agent_count))
noise = [0.0]

kernel.runner(
    agents=agents,
    startTime=kernelStartTime,
    stopTime=kernelStopTime,
    agentLatency=latency,
    latencyNoise=noise,
    defaultComputationDelay=defaultComputationDelay,
    defaultLatency=0,
    oracle=oracle,
    log_dir=args.log_dir,
)

simulation_end_time = dt.datetime.now()
print("Simulation End Time: {}".format(simulation_end_time))
print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))
