import datetime
import pandas as pd
import numpy as np
import jsons as js

from agent.TradingAgent import TradingAgent
from agent.execution.signals import mid_price, spread, volume_order_imbalance
from agent.execution.util import create_uniform_grid, discretize
from util.util import log_print


class QLearningTWAPAgent(TradingAgent):
    SIZE_ALLOCATION = {1: [1, 0, 0, 0],
                       2: [.5, .5, 0, 0],
                       3: [.34, .33, .33, 0],
                       4: [.25, .25, .25, .25]}

    def __init__(self, id, name, type, symbol, starting_cash,
                 direction, quantity, execution_time_horizon, freq,
                 mode='train', q_table=None, trade=True, log_orders=False, random_state=None):

        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)

        # client order information
        self.symbol = symbol
        self.direction = direction  # 'BUY' or 'SELL'
        self.quantity = quantity  # parent order size

        # order scheduler information
        self.execution_time_horizon = execution_time_horizon  # fixed frequency DatetimeIndex for the agent trade times
        self.start_time = self.execution_time_horizon[0]
        self.end_time = self.execution_time_horizon[-1]
        self.freq = freq
        self.schedule = self.generate_schedule()

        self.remaining_time = len(self.execution_time_horizon) - 1  # remaining_time is in the units of execution time
        self.remaining_qty = quantity

        self.s = None
        self.a = None
        self.t = 0

        self.q_table = q_table
        self.experience = dict()  # Tuples of (s,a,s',r).

        self.discrete_state_grid = create_uniform_grid(low=[0, 0], high=[1.0, 1.0], bins=(200, 200))
        self.action_space_size = 5
        self.mode = mode
        self.state = 'AWAITING_WAKEUP'
        self.log_orders = log_orders
        self.trade = trade

        self.arrival_price = None
        self.accepted_orders = []

    def wakeup(self, current_time):
        can_trade = super().wakeup(current_time)
        if not can_trade: return
        if self.trade:
            try:
                self.setWakeup([time for time in self.execution_time_horizon if time > current_time][0])
            except IndexError:
                log_print(f"[---- {self.name}  t={self.t} -- {current_time} ----]: RL Agent wakeups complete")
                self.trade = False

        self.getCurrentSpread(self.symbol, depth=500)
        self.state = 'AWAITING_SPREAD'

    def receiveMessage(self, current_time, msg):
        super().receiveMessage(current_time, msg)
        if msg.body['msg'] == 'ORDER_ACCEPTED':
            self.handle_order_acceptance(current_time, msg)
        elif msg.body['msg'] == 'ORDER_EXECUTED':
            self.handle_order_execution(current_time, msg)
        elif current_time in self.execution_time_horizon[:-1] and self.remaining_qty > 0 and \
                self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD':
            self.cancel_orders(current_time)
            self.place_order(current_time)
            self.t += 1

    def kernelStopping(self):
        super().kernelStopping()

        slippage = self.get_average_transaction_price() - self.arrival_price if self.direction == 'BUY' else \
            self.arrival_price - self.get_average_transaction_price()

        self.logEvent('DIRECTION', self.direction, True)
        self.logEvent('TOTAL_QTY', self.quantity, True)
        self.logEvent('REM_QTY', self.remaining_qty, True)
        self.logEvent('ARRIVAL_MID', self.arrival_price, True)
        self.logEvent('AVG_TXN_PRICE', self.get_average_transaction_price(), True)
        self.logEvent('SLIPPAGE', slippage, True)

        agent_experience_df = pd.DataFrame(self.experience).T
        agent_experience_df.columns = ['s', 'a', 's_prime', 'r']

        self.writeLog(dfLog=agent_experience_df, filename='agent_experience')

    def place_order(self, current_time):

        bids, asks = self.getKnownBidAsk(symbol=self.symbol, best=False)
        if current_time == self.start_time:
            self.arrival_price = mid_price(bids[0][0], asks[0][0])
            _, self.s = self.get_observation(current_time, bids, asks)
            log_print(f"[---- {self.name} t={self.t} -- {current_time} ----]: Arrival Mid Price {self.arrival_price}")
            log_print(f"[---- {self.name} t={self.t} -- {current_time} ----]: Parent Order Details: {self.direction} {self.quantity} "
                  f"between {self.start_time} and {self.end_time}, {self.freq} frequency \n")

        _, s_prime = self.get_observation(current_time, bids, asks)
        a = self.take_action(current_time, bids, asks, s_prime)

        self.experience[self.t] = (self.s, a, s_prime, None)

        self.s = s_prime
        self.a = a

    def get_observation(self, current_time, bids, asks):

        def get_remaining_time(current_time):
            current_time = current_time.floor(self.freq)
            for i, t in enumerate(list(self.execution_time_horizon)):
                if t == current_time:
                    current_ts_index = i
                    return len(self.execution_time_horizon) - 1 - current_ts_index
            return len(self.execution_time_horizon)

        # Private Variables (1) Time Remaining, (2) Quantity Remaining
        self.remaining_time = get_remaining_time(current_time)

        time_remaining = self.remaining_time / (len(self.execution_time_horizon) - 1)
        qty_remaining = self.remaining_qty / self.quantity

        # Market Variables:
        # (1) Spread, (2) Volume Imbalance, (3) Volatility
        # sprd = int(spread(best_bid_price=bids[0][0], best_ask_price=asks[0][0]))
        # vol_imbalance = round(volume_order_imbalance(best_bid_size=bids[0][1], best_ask_size=asks[0][1]), 2)

        observation = [time_remaining, qty_remaining]
        discrete_observation = discretize(observation, self.discrete_state_grid)
        return observation, tuple(discrete_observation)

    def take_action(self, current_time, bids, asks, s):

        a = None
        if self.mode == 'train':
            a = self.random_state.randint(0, self.action_space_size)
        elif self.mode == 'test':
            a = np.argmax(self.q_table.q[s])

        qty = self.schedule[pd.Interval(current_time, current_time + datetime.timedelta(minutes=1))]

        if a == 0:
            # Action 1: Place a MARKET order
            log_print(f'[---- {self.name} t={self.t} -- {current_time} ----]: Placing a MARKET order for qty = {qty}')
            self.placeMarketOrder(symbol=self.symbol, direction=self.direction, quantity=qty)
        elif a in range(1, self.action_space_size):
            # Action X: Place a LIMIT buy (sell) order at different bid (ask) price levels
            # price = bids[a][0] if self.direction == 'BUY' else asks[a][0]

            size_allocations = QLearningTWAPAgent.SIZE_ALLOCATION.get(a)
            for price_level, allocation in enumerate(size_allocations):
                size = round(allocation * qty)
                price = bids[price_level][0] if self.direction == 'BUY' else asks[price_level][0]
                if size != 0:
                    log_print(f'[---- {self.name} t={self.t} -- {current_time} ----]: '
                          f'Placing a LIMIT order for qty = {size} @ {price}, a={a}')
                    self.placeLimitOrder(symbol=self.symbol, quantity=size,
                                         is_buy_order=self.direction == 'BUY', limit_price=price)
        return a

    def compute_reward(self, executed_order, current_time, s):
        r = None
        # 1) Reward based on slippage
        if self.direction == 'BUY':
            r = 1 - ((executed_order.fill_price - self.arrival_price) / self.arrival_price)
        elif self.direction == 'SELL':
            r = 1 - ((self.arrival_price - executed_order.fill_price) / self.arrival_price)

        # 2) Reward modification based on s_prime
        time_remaining = s[0]
        qty_remaining = s[1]
        r = r * time_remaining * qty_remaining

        log_print(f'[---- {self.name} t={self.t} -- {current_time} ----]: '
              f'EXECUTED ORDER FILL PRICE: {executed_order.fill_price}, ARRIVAL PRICE: {self.arrival_price}, REWARD: {r}')
        return r

    def getWakeFrequency(self):
        return self.start_time - self.mkt_open

    def generate_schedule(self):
        bins = pd.interval_range(start=self.start_time, end=self.end_time, freq=self.freq)
        schedule = {}
        child_quantity = int(self.quantity / (len(self.execution_time_horizon) - 1))
        for b in bins:
            schedule[b] = child_quantity
        log_print(f'[---- {self.name}  - Schedule ----]:')
        log_print(f'[---- {self.name}  - Total Number of Orders ----]: {len(schedule)}')
        for t, q in schedule.items():
            log_print(f"From: {t.left.time()}, To: {t.right.time()}, Quantity: {q}")
        return schedule

    def handle_order_execution(self, current_time, msg):
        executed_order = msg.body['order']
        self.executed_orders.append(executed_order)
        total_executed_quantity = sum(executed_order.quantity for executed_order in self.executed_orders)
        self.remaining_qty = self.quantity - total_executed_quantity
        log_print(f'[---- {self.name} t={self.t} -- {current_time} ----]: LIMIT ORDER EXECUTED - {executed_order.quantity} @ {executed_order.fill_price}')
        log_print(f'[---- {self.name} t={self.t} -- {current_time} ----]: TOTAL EXECUTED QUANTITY: {total_executed_quantity}, '
              f'REMAINING QUANTITY (NOT EXECUTED): {self.remaining_qty}, % EXECUTED: {round((1 - self.remaining_qty / self.quantity) * 100, 2)}')

        if current_time.floor(self.freq) in self.execution_time_horizon:
            experience = list(self.experience[self.t - 1])
            bids, asks = self.getKnownBidAsk(symbol=self.symbol, best=False)
            s_prime_vals, experience[2] = self.get_observation(current_time, bids, asks)
            self.s = experience[2]

            r = self.compute_reward(executed_order, current_time, s_prime_vals)
            experience[3] = r
            self.experience[self.t-1] = tuple(experience)
            log_print(f'[---- {self.name} t={self.t} -- {current_time} ----]: '
                  f't={self.t - 1} -> s={self.experience[self.t - 1][0]}, a={self.experience[self.t - 1][1]}, '
                  f's_prime={self.experience[self.t - 1][2]}, r={self.experience[self.t - 1][3]} \n')

    def handle_order_acceptance(self, current_time, msg):
        accepted_order = msg.body['order']
        self.accepted_orders.append(accepted_order)
        accepted_qty = sum(accepted_order.quantity for accepted_order in self.accepted_orders)
        log_print(f'[---- {self.name} t={self.t} -- {current_time} ----]: ACCEPTED QUANTITY : {accepted_qty}')

        if current_time.floor(self.freq) in self.execution_time_horizon:
            experience = list(self.experience[self.t - 1])
            bids, asks = self.getKnownBidAsk(symbol=self.symbol, best=False)
            _, experience[2] = self.get_observation(current_time, bids, asks)
            self.s = experience[2]
            experience[3] = - 10
            self.experience[self.t - 1] = tuple(experience)
            log_print(f'[---- {self.name} t={self.t} - {current_time} ----]: '
                  f't={self.t - 1} -> s={self.experience[self.t - 1][0]}, a={self.experience[self.t - 1][1]}, '
                  f's_prime={self.experience[self.t - 1][2]}, r={self.experience[self.t - 1][3]} \n')

    def cancel_orders(self, current_time=None):
        """ used by the trading agent to cancel all of its orders.
        """
        for _, order in self.orders.items():
            log_print(f'[---- {self.name} t={self.t} - {current_time} ----]: CANCELLED QUANTITY : {order.quantity}')
            if self.log_orders: self.logEvent('CANCEL_SUBMITTED', js.dump(order, strip_privates=True))
            self.cancelOrder(order)
            self.accepted_orders = []
