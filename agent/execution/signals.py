

def mid_price(best_bid_price, best_ask_price):
    return (best_bid_price + best_ask_price) / 2


def spread(best_bid_price, best_ask_price):
    return best_ask_price - best_bid_price


def volume_order_imbalance(best_bid_size, best_ask_size):
    return (best_ask_size - best_bid_size) / (best_ask_size + best_bid_size)