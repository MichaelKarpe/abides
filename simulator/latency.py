import numpy as np
from scipy.spatial.distance import pdist
from model.LatencyModel import LatencyModel


def latency_model(num_agents):

    latency_rstate = np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32))
    pairwise = (num_agents, num_agents)

    # All agents sit on line from Seattle to NYC
    nyc_to_seattle_meters = 3866660
    pairwise_distances = generate_uniform_random_pairwise_dist_on_line(
        0.0, nyc_to_seattle_meters, num_agents, random_state=latency_rstate
    )
    pairwise_latencies = meters_to_light_ns(pairwise_distances)

    model_args = {"connected": True, "min_latency": pairwise_latencies}

    latency_model = LatencyModel(latency_model="deterministic", random_state=latency_rstate, kwargs=model_args)

    return latency_model


def generate_uniform_random_pairwise_dist_on_line(left, right, num_points, random_state=None):
    """Uniformly generate points on an interval, and return numpy array of pairwise distances between points.
    :param left: left endpoint of interval
    :param right: right endpoint of interval
    :param num_points: number of points to use
    :param random_state: np.RandomState object
    :return:
    """

    x_coords = random_state.uniform(low=left, high=right, size=num_points)
    x_coords = x_coords.reshape((x_coords.size, 1))
    out = pdist(x_coords, "euclidean")
    return out


def meters_to_light_ns(x):
    """Converts x in units of meters to light nanoseconds
    :param x:
    :return:
    """
    x_lns = x / 299792458e-9
    x_lns = x_lns.astype(int)
    return x_lns
