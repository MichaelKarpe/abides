import pandas as pd
import numpy as np
import logging as log

from agent.TradingAgent import TradingAgent

# (time remaining, quantity remaining, spread, volume imbalance)
DISCRETE_SPACE_GRID = {'low': [0.1, 0.1, 1, 0.1],
                       'high': [1.0, 1.0, 20, 1.0],
                       'bins': (5, 5, 5, 5)}


class TabularQLearningAgent(TradingAgent):
    """ This class implements a simple Q-learning trading agent in a discrete state
    and action space following a gym-like code interface
    """

    def __init__(self, id, name, type, security, starting_cash, direction, quantity, duration, freq, schedule,
                 q_table=None, log_orders=False, random_state=None):
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.security = security  # name of security traded
        self.direction = direction  # buy or sell
        self.quantity = quantity  # parent order quantity (total quantity that needs to be executed)
        self.duration = duration  # intra-day time horizon to executed the trades
        self.start_time = self.duration[0]  # start time of execution
        self.end_time = self.duration[-1]  # end time of execution
        self.freq = freq  # frequency of order placement
        self.schedule = schedule  # pre-computed execution time and quantity schedule as a dictionary of timestamps
        # to child order quantities
        self.wakeup_times = pd.date_range(start=self.start_time,
                                          end=self.end_time + self.duration.freq, # extra wakeup call to log the results
                                          freq=self.duration.freq)

        self.remaining_parent_quantity = quantity
        self.arrival_price = None # mid price at the start of the order execution
        self.executed_orders_in_step = []
        self.last_trade_price = None

        self.discrete_state_grid = create_uniform_grid(low=DISCRETE_SPACE_GRID['low'],
                                                       high=DISCRETE_SPACE_GRID['high'],
                                                       bins=DISCRETE_SPACE_GRID['bins'])
        self.action_space_size = 5

        self.t = 0
        self.T = len(self.duration) - 1

        self.s = None
        self.a = None

        self.episode_reward = 0 # an episode reward is the total cumulative step rewards
        self.step_rewards = [] # list of the step rewards obtained during the episode
        self.experience = []  # Tuples of (s,a,s',r).

        self.q_table = q_table
        self.alpha = self.q_table.alpha
        self.epsilon = self.q_table.epsilon
        self.gamma = self.q_table.gamma
        self.td_errors = []

        self.state = 'AWAITING_WAKEUP'

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)
        self.log_model_details()

    def kernelStopping(self):
        super().kernelStopping()
        self.saveState(self.q_table)
        self.log_model_details()
        self.log_results()

    def wakeup(self, current_time):
        can_trade = super().wakeup(current_time)
        if not can_trade: return
        try:
            self.setWakeup([time for time in self.wakeup_times if time > current_time][0])
        except IndexError:
            pass
        self.getCurrentSpread(self.security, depth=500)
        self.state = 'AWAITING_SPREAD'

    def getWakeFrequency(self):
        return self.start_time - self.mkt_open # used to determine the first wakeup timestamp

    def receiveMessage(self, current_time, msg):
        super().receiveMessage(current_time, msg)
        if msg.body['msg'] == 'ORDER_EXECUTED':
            self.handle_order_execution(msg)
        elif self.start_time <= current_time <= self.wakeup_times[-1].floor(self.freq) and \
                self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD':
            self.agent_log(msg='Remaining Quantity: {}'.format(self.remaining_parent_quantity), type='info')

            # 1) step in the environment to get the next state and compute reward after taking action at timestep before
            action = self.a
            next_state, reward, done = self.step(action)

            # 2) do model (e.g. Q-Learning) update
            self.learn(reward, next_state)

            # 3) Take the action and place the order
            next_action = self.act(next_state) if self.t != self.T and self.remaining_parent_quantity > 0 else None

            # 3) Update s and a
            self.s = next_state
            self.a = next_action

            # 4) Advance time to next step
            self.t += 1

    def step(self, action):
        """
        returns next_state, r, done after querying the state at t+1 for action at t
        :param action: action taken at time t
        :return: next_state, r, done
        """
        # 1) Query the orderbook to get the agent and market variables for next_state
        bids, asks = self.getKnownBidAsk(symbol=self.security, best=False)
        if self.currentTime.floor(self.freq) == self.start_time:
            self.arrival_price = (bids[0][0] + asks[0][0]) / 2
            self.starting_cash = self.arrival_price * self.quantity
            self.agent_log(msg='Parent Order Details: {} {} between {} and {}, '
                               'every {} minutes'.format(self.direction, self.quantity,
                                                         self.start_time, self.end_time,
                                                         self.freq.seconds / 60), type='info')
            self.agent_log(msg='Child Order size: {}'.format(self.schedule[self.start_time]), type='info')
            self.agent_log(msg='Arrival Mid Price: {}'.format(self.arrival_price), type='info')
            self.log_model_details()

        # 2) get next_state
        next_state = self.observe(bids, asks)

        # 3) compute the step reward
        reward = self.step_reward()

        # 5) Store our experience tuple
        self.experience.append((self.s, action, next_state, reward))
        self.agent_log(msg='s = {}, a = {}, next_state = {}, r = {}'.format(self.s, action, next_state, reward),
                       type='info')

        done = self.t == self.T

        if done:
            self.agent_log(msg='Finito Done!!', type='info')
        return next_state, reward, done

    def learn(self, reward, next_state):
        """
        Q-Learning Update
        :param r:       reward
        :param next_state: next state
        """

        if self.t != 0:
            best_next_action = np.argmax(self.q_table.q[next_state])
            td_target = reward + (self.gamma * self.q_table.q[next_state][best_next_action])
            td_error = td_target - self.q_table.q[self.s][self.a]
            self.q_table.q[self.s][self.a] = self.q_table.q[self.s][self.a] + self.alpha * td_error
            self.td_errors.append(td_error)
            self.agent_log(msg='Q-Table updated, td target: {}, td error {}'.format(td_target, td_error))

    def act(self, next_state):
        def epsilon_greedy():
            if self.random_state.rand() < self.epsilon:
                a = self.random_state.randint(0, self.action_space_size)
                self.agent_log(msg='Exploring Randomly ... a = {}'.format(a), type='info')
            else:
                # Expected best action.
                all_zeros = not np.any(self.q_table.q[next_state])
                if all_zeros:
                    a = self.random_state.randint(0, self.action_space_size)
                    self.agent_log(msg='all actions = 0, exploring randomly ... a = {}'.format(a), type='info')
                else:
                    a = np.argmax(self.q_table.q[next_state])
                self.agent_log(msg='Exploiting ... a = {}'.format(a), type='info')
            return a

        def catchup_market_order():
            if self.remaining_parent_quantity > 0 and \
                    self.currentTime.floor(self.freq) == self.duration[-2]:
                self.placeMarketOrder(symbol=self.security, direction=self.direction,
                                      quantity=self.remaining_parent_quantity)
                self.agent_log(msg='Market order submitted - q={}'.format(self.remaining_parent_quantity))
            return 0

        #a = epsilon_greedy() if self.currentTime.floor(self.freq) != self.duration[-2] \
        #    else catchup_market_order()
        a = epsilon_greedy()

        # cancels any existing order first
        self.cancel_orders()
        # get the latest order book information
        bids, asks = self.getKnownBidAsk(symbol=self.security, best=False)

        # get the child quantity from the schedule
        child_qty = self.schedule[self.currentTime.floor(self.freq)]

        # discrete actions
        # (0) market order, (1) limit order best bid (buy order) or best ask (sell order),
        # (2) limit order at second best bid (buy order) or best ask (sell order), ...
        if a == 0:
            self.placeMarketOrder(symbol=self.security, quantity=child_qty, direction=self.direction)
            self.agent_log(msg='Market order submitted - q={}'.format(child_qty))
        else:
            try:
                price = bids[a - 1][0] if self.direction == 'BUY' else asks[a - 1][0]
                self.last_trade_price = price
                self.placeLimitOrder(symbol=self.security, quantity=child_qty,
                                     is_buy_order=self.direction == 'BUY', limit_price=price)
            except IndexError:
                self.placeLimitOrder(symbol=self.security, quantity=child_qty,
                                     is_buy_order=self.direction == 'BUY', limit_price=self.last_trade_price)
                self.agent_log('ORDERBOOK SIDE EMPTY!!!!!!. placing limit order of price equal to last trade price!',
                               type='error')
                self.agent_log('bids: {}, asks: {}'.format(bids[:10], asks[:10]), type='error')
            self.agent_log(msg='Limit order submitted - a={} ''price level: {}, q={}, p={}'.format(a, a - 1, child_qty,
                                                                                                   self.last_trade_price))
        return a

    def observe(self, bids, asks):
        """
        gets the current observations (private and market variables)
        :param bids:  list of bids
        :param asks:  list of asks
        :return observation: tuple of n continuous or discrete observations
        """

        def spread(best_bid_price, best_ask_price):
            return best_ask_price - best_bid_price

        def volume_order_imbalance(best_bid_size, best_ask_size):
            imb = best_ask_size / (best_bid_size + best_ask_size) if self.direction == 'BUY'  \
                                                                  else best_bid_size / (best_bid_size + best_ask_size)
            return imb

        elapsed_time = self.t / self.T
        remaining_qty = self.remaining_parent_quantity / self.quantity
        sprd = spread(bids[0][0], asks[0][0])
        vol_imb = volume_order_imbalance(bids[0][1], asks[0][1])
        observation = (elapsed_time, remaining_qty, sprd, vol_imb)

        discrete_observation = discretize(observation, self.discrete_state_grid)
        return discrete_observation

    def step_reward(self):
        """
        computes and returns the step reward
        :return step rewards -> list
        """
        step_reward = 0
        if self.executed_orders_in_step:
            for filled_qty, filled_price in self.executed_orders_in_step:
                r = 0
                if self.direction == 'BUY':
                    r = filled_qty * (self.arrival_price - filled_price)
                elif self.direction == 'SELL':
                    r = filled_qty * (filled_price - self.arrival_price)
                step_reward += r

        self.agent_log(msg='Reward for t={}: {} '.format(self.t - 1, step_reward), type='info')
        self.executed_orders_in_step = []
        self.step_rewards.append(step_reward)
        self.episode_reward += step_reward
        return step_reward

    def handle_order_execution(self, msg):
        self.executed_orders.append(msg.body['order'])
        executed_qty = msg.body['order'].quantity
        executed_price = msg.body['order'].fill_price
        self.executed_orders_in_step.append((executed_qty, executed_price))
        self.agent_log(msg='Limit Order Executed p={}, q={}'.format(executed_price, executed_qty))
        self.remaining_parent_quantity -= executed_qty

    def cancel_orders(self):
        """ used by the trading agent to cancel all of its orders.
        """
        for _, order in self.orders.items():
            self.cancelOrder(order)

    def log_model_details(self):
        msg = 'Tabular Q-Learning - Alpha: {}, Epsilon: {}, Gamma: {}'.format(self.alpha, self.epsilon, self.gamma)
        self.agent_log(msg=msg, type='info')

    def log_results(self):

        self.writeLog(dfLog=pd.DataFrame(self.experience).T, filename='transitions')
        self.writeLog(dfLog=pd.DataFrame(self.step_rewards).T, filename='step_rewards')

        total_executed_volume = sum(executed_order.quantity for executed_order in self.executed_orders)
        total_pnl = self.arrival_price * total_executed_volume - sum(executed_order.fill_price * executed_order.quantity
                                                                     for executed_order in self.executed_orders)
        msg = 'Final Results: PnL: {} Reward {}, ' \
              'Arrival Price {}, Avg Txn Price {}, Difference: {}, ' \
              'Remaining Quantity {}'.format(total_pnl, self.episode_reward, self.arrival_price,
                                             self.get_average_transaction_price(),
                                             round(self.arrival_price - self.get_average_transaction_price(), 2),
                                             self.remaining_parent_quantity)
        self.agent_log(msg, type='info')

        # client order details
        self.logEvent('direction', self.direction, True)
        self.logEvent('quantity', self.quantity, True)
        self.logEvent('security', self.security, True)
        self.logEvent('date', str(self.start_time.date()), True)
        self.logEvent('start_time', self.start_time, True)
        self.logEvent('end_time', self.end_time, True)

        # execution results
        self.logEvent('quantity_executed', self.quantity - self.remaining_parent_quantity, True)
        self.logEvent('quantity_remaining', self.remaining_parent_quantity, True)
        self.logEvent('arrival_price', self.arrival_price, True)
        self.logEvent('average_transaction_price', self.get_average_transaction_price(), True)

        difference = self.arrival_price - self.get_average_transaction_price() if self.direction == 'BUY' else \
            self.get_average_transaction_price() - self.arrival_price
        self.logEvent('difference', difference, True)
        self.logEvent('pnl', total_pnl, True)

        # rl algorithm results
        self.logEvent('step_rewards', self.step_rewards, True)
        self.logEvent('episode_reward', self.episode_reward, True)
        self.logEvent('avg_reward_per_step', sum(self.step_rewards) / len(self.step_rewards), True)

        self.logEvent('alpha', self.alpha, True)
        self.logEvent('epsilon', self.epsilon, True)
        self.logEvent('gamma', self.gamma, True)

    def agent_log(self, msg, type='info'):
        if type == 'debug':
            log.debug("[-- {} -- t={} -- {} --]: {}".format(self.kernel.name, self.t, self.currentTime, msg))
        elif type == 'info':
            log.info("[-- {} -- t={} -- {} --]: {}".format(self.kernel.name, self.t, self.currentTime, msg))
        elif type == 'error':
            log.error("[-- {} -- t={} -- {} --]: {}".format(self.kernel.name, self.t, self.currentTime, msg))


def create_uniform_grid(low, high, bins=(10, 10)):
    """Define a uniformly-spaced grid that can be used to discretize a space.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins along each corresponding dimension.

    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    source: https://stackoverflow.com/questions/49766071/create-uniformly-spaced-grid-in-python
    """
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
    return grid


def discretize(sample, grid):
    """Discretize a sample as per given grid.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    grid : list of array_like
        A list of arrays containing split points for each dimension.

    Returns
    -------
    discretized_transition : array_like
        A sequence of integers with the same number of dimensions as sample.
    source: https://stackoverflow.com/questions/49766071/create-uniformly-spaced-grid-in-python
    """
    return tuple(int(np.digitize(s, g)) for s, g in zip(sample, grid))  # apply along each dimension