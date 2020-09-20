import matplotlib.pyplot as plt
from agent.execution.metrics import Metrics


class Plots:
    """Utility class to plot metrics associated with the order book orders and snapshots.
    Attributes:
        orderbook_df: A pandas Dataframe describing the order book snapshots in time
        orders_df: A pandas Dataframe describing the orders stream
    """

    def __init__(self, symbol, date, orderbook_df, orders_df, log_folder, bps=False):
        self.log_folder = log_folder
        self.metrics = Metrics(symbol, date, orderbook_df, orders_df, bps)

    def plot_mid_price(self, bps=False):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(30, 10)
        ax.set_title(f"Mid Price ({'bps' if bps else '$'})", size=24, fontweight="bold")
        ax.set_xlabel("Time", size=20)
        ax.set_ylabel("Mid Price", size=20)
        ax.set_facecolor("white")
        ax.plot(self.metrics.mid_price())
        plt.savefig(self.log_folder + "/" + "Mid_Price.jpg", bbox_inches="tight")

    def plot_spread(self, bps=True):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(30, 10)
        ax.set_title(f"Spread ({'bps' if bps else '$'})", size=24, fontweight="bold")
        ax.set_xlabel("Time", size=20)
        ax.set_ylabel("Spread", size=20)
        ax.set_facecolor("white")
        ax.plot(self.metrics.spread(type="naive"), label="Naive Spread")
        ax.plot(self.metrics.spread(type="effective"), label="Effective Spread")
        ax.legend()
        plt.savefig(self.log_folder + "/" + "Spread.jpg", bbox_inches="tight")

    def plot_volume_order_imbalance(self):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(30, 10)
        ax.set_title("Volume Order Imbalance", size=24, fontweight="bold")
        ax.set_xlabel("Time", size=20)
        ax.set_ylabel("Volume Order Imbalance", size=20)
        ax.set_facecolor("white")
        ax.plot(self.metrics.volume_order_imbalance())
        plt.savefig(self.log_folder + "/" + "Volume_Order_Imbalance.jpg", bbox_inches="tight")

    def plot_order_flow_imbalance(self):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(30, 10)
        ax.set_title("Order Flow Imbalance", size=24, fontweight="bold")
        ax.set_xlabel("Time", size=20)
        ax.set_ylabel("Order Flow Imbalance", size=20)
        ax.set_facecolor("white")
        ax.plot(self.metrics.order_flow_imbalance())
        plt.savefig(self.log_folder + "/" + "Order_Flow_Imbalance.jpg", bbox_inches="tight")
