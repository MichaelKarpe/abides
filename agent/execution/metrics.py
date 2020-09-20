import pandas as pd
import numpy as np


class Metrics:
    """ Utility class to calculate metrics associated with the order book orders and snapshots.
    Attributes:
        orderbook_df: A pandas Dataframe describing the order book snapshots in time
        orders_df: A pandas Dataframe describing the orders stream
    """

    def __init__(self, symbol, date, orderbook_df, orders_df, bps=False):
        self.symbol = symbol
        self.date = date
        self.orderbook_df = orderbook_df
        self.orders_df = orders_df
        self.bps = bps

    def mid_price(self):
        """Returns the mid price in the form of a pandas series (Orderbook specific)"""
        mid_price = (self.orderbook_df.ask_price_1 + self.orderbook_df.bid_price_1) / 2
        return mid_price * 10000 if self.bps else mid_price

    def spread(self, type='naive'):
        """Returns the spread in the form of a pandas series (Orderbook specific)"""
        spread = None
        if type == 'naive':
            spread = self.orderbook_df.ask_price_1 - self.orderbook_df.bid_price_1
        elif type == 'effective':
            num_price_levels = 5
            volume_weighted_ask = sum([self.orderbook_df[f'ask_price_{level}'] * self.orderbook_df[f'ask_size_{level}']
                                       for level in range(1, num_price_levels + 1)]) / \
                                  sum(self.orderbook_df[f'ask_size_{level}'] for level in range(1, num_price_levels + 1))
            volume_weighted_bid = sum([self.orderbook_df[f'bid_price_{level}'] * self.orderbook_df[f'bid_size_{level}']
                                       for level in range(1, num_price_levels + 1)]) / \
                                  sum(self.orderbook_df[f'bid_size_{level}'] for level in range(1, num_price_levels + 1))
            spread = volume_weighted_ask.ffill() - volume_weighted_bid.ffill()
        return spread * 10000 if self.bps else spread

    def volume_order_imbalance(self):
        """Returns the volume order imbalance in the form of a pandas series (Orderbook specific)"""
        return self.orderbook_df.bid_size_1 / (self.orderbook_df.ask_size_1 + self.orderbook_df.bid_size_1)

    def order_flow_imbalance(self, tick_size=0.01, sampling_freq='1ms'):
        """Returns the order flow imbalance in the form of a pandas series (Orderbook specific)"""
        ofi_df = self.orderbook_df[['ask_price_1', 'ask_size_1', 'bid_price_1', 'bid_size_1']].copy().reset_index()
        ofi_df['mid_price_change'] = ((ofi_df['bid_price_1'] + ofi_df['ask_price_1']) / 2).diff().div(tick_size)
        ofi_df['PB_prev'] = ofi_df['bid_price_1'].shift()
        ofi_df['SB_prev'] = ofi_df['bid_size_1'].shift()
        ofi_df['PA_prev'] = ofi_df['ask_price_1'].shift()
        ofi_df['SA_prev'] = ofi_df['ask_size_1'].shift()
        ofi_df = ofi_df.dropna()
        bid_geq = ofi_df['bid_price_1'] >= ofi_df['PB_prev']
        bid_leq = ofi_df['bid_price_1'] <= ofi_df['PB_prev']
        ask_geq = ofi_df['ask_price_1'] >= ofi_df['PA_prev']
        ask_leq = ofi_df['ask_price_1'] <= ofi_df['PA_prev']
        ofi_df['OFI'] = pd.Series(np.zeros(len(ofi_df)))
        ofi_df['OFI'].loc[bid_geq] += ofi_df['bid_size_1'][bid_geq]
        ofi_df['OFI'].loc[bid_leq] -= ofi_df['SB_prev'][bid_leq]
        ofi_df['OFI'].loc[ask_geq] += ofi_df['SA_prev'][ask_geq]
        ofi_df['OFI'].loc[ask_leq] -= ofi_df['ask_size_1'][ask_leq]
        ofi_df = ofi_df.set_index('index')
        ofi_df = ofi_df[['mid_price_change', 'OFI']].resample(sampling_freq).sum().dropna()
        return ofi_df.OFI

    @staticmethod
    def twap(executed_trades_df):
        return executed_trades_df['fill_price'].cumsum() / pd.Series(np.arange(1, len(executed_trades_df) + 1),
                                                                     executed_trades_df.index)
    @staticmethod
    def vwap(executed_trades_df):
        return (executed_trades_df['fill_price'] * executed_trades_df['quantity']).cumsum() / executed_trades_df[
            'quantity'].cumsum()

    @staticmethod
    def slippage(exchange_executed_trades_df, agent_executed_trades_df, trade_direction, benchmark='VWAP', bps=True):
        direction = 1 if trade_direction == 'BUY' else -1
        if benchmark == 'VWAP':
            slippage = direction * (Metrics.vwap(exchange_executed_trades_df) - Metrics.vwap(agent_executed_trades_df)) / \
                       Metrics.vwap(exchange_executed_trades_df)
        elif benchmark == 'TWAP':
            slippage = direction * (Metrics.twap(exchange_executed_trades_df) - Metrics.twap(agent_executed_trades_df)) / \
                       Metrics.twap(exchange_executed_trades_df)
        slippage = slippage[~slippage.isnull()]
        return slippage * 10000 if bps else slippage