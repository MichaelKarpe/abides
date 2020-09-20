import numpy as np
import pandas as pd

from agent.ExchangeAgent import ExchangeAgent
from agent.examples.MarketReplayAgent import MarketReplayAgent


SECURITIES = [
    "MMM",
    "AXP",
    "BA",
    "CAT",
    "CVX",
    "CSCO",
    "KO",
    "DWDP",
    "XOM",
    "HD",
    "IBM",
    "INTC",
    "JNJ",
    "MCD",
    "MRK",
    "MSFT",
    "NKE",
    "PFE",
    "PG",
    "TRV",
    "UNH",
    "UTX",
    "VZ",
    "V",
    "WMT",
    "WBA",
    "DIS",
    "AAPL",
    "JPM",
    "DIA",
    "GS",
]

DATES = [
    "20190628",
    "20190627",
    "20190626",
    "20190625",
    "20190624",
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

MKT_OPEN_TIME = "09:30:00"
MKT_CLOSE_TIME = "16:00:00"


def get_mr_config(global_seed, security, date, test=False):
    """create the list of agents for the market replay simulation
    :param global_seed: abides config parameters
    :param security: abides config parameters
    :param date: abides config parameters
    :param test: abides config parameters
    :return: list of agents given a set of parameters
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
                mkt_open=pd.to_datetime("{} {}".format(date, MKT_OPEN_TIME)),
                mkt_close=pd.to_datetime("{} {}".format(date, MKT_CLOSE_TIME)),
                symbols=[security],
                log_orders=test,
                book_freq=0 if test else None,
                wide_book=True,
                pipeline_delay=0,
                computation_delay=0,
                stream_history=int(1e2),
                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)),
            )
        ]
    )
    agent_count += 1

    # L3 data path
    file_name = f"DOW30/{security}/{security}.{date}"
    orders_file_path = f"/efs/data/{file_name}"
    processed_orders_folder_path = "/efs/data/marketreplay/"

    agents.extend(
        [
            MarketReplayAgent(
                id=1,
                name="MARKET_REPLAY_AGENT",
                type="MarketReplayAgent",
                symbol=security,
                log_orders=test,
                date=pd.to_datetime(date),
                start_time=pd.to_datetime("{} {}".format(date, MKT_OPEN_TIME)),
                end_time=pd.to_datetime("{} {}".format(date, MKT_CLOSE_TIME)),
                orders_file_path=orders_file_path,
                processed_orders_folder_path=processed_orders_folder_path,
                starting_cash=0,
                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)),
            )
        ]
    )
    agent_count += 1

    return agents, agent_count
