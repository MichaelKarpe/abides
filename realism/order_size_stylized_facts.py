import argparse
import os
import sys
from pathlib import Path
p = str(Path(__file__).resolve().parents[1])  # directory one level up from this file
sys.path.append(p)
from util.formatting.convert_order_stream import dir_path
from order_flow_stylized_facts import unpickle_stream_dfs_to_stream_list, YEAR_OFFSET
import pickle
from realism_utils import get_plot_colors
import matplotlib.pyplot as plt
import numpy as np
from pomegranate import LogNormalDistribution, NormalDistribution, GeneralMixtureModel


class Constants:
    """ Stores constants for use in plotting code. """

    # Plot params -- Generic
    fig_height = 10
    fig_width = 15
    tick_label_size = 20
    legend_font_size = 20
    axes_label_font_size = 20
    title_font_size = 22
    scatter_marker_styles_sizes = [('x', 60), ('+', 60), ('o', 14), (',', 60)]

    # Generalised Mixture model fit params
    norm_spike_sigma = 0.15
    log_norm_initial_weight = 0.5
    EM_inertia = 0.5
    num_norm_spikes = 10
    log_mu = 4
    log_sigma = 1.38

    # Plot params -- Limit Order size
    limit_order_sizes_xlabel = "Limit Order Size"
    limit_order_sizes_ylabel = "Empirical density"
    limit_order_sizes_filename = "limit_order_sizes"
    limit_order_size_fit_lower_bound = 0
    limit_order_size_fit_upper_bound = 10
    limit_order_size_hist_linewidth = 3
    limit_order_size_fit_linewidth = 2
    num_points_x_axis = 500


def bundled_stream_limit_order_sizes(bundled_streams):
    """ From bundled streams return dict with limit order sizes collated by symbol. """

    limit_order_sizes_dict = dict()

    for idx, elem in enumerate(bundled_streams):
        print(f"Processing elem {idx + 1} of {len(bundled_streams)}")
        orders_df = elem["orders_df"]
        symbol = elem["symbol"]
        limit_orders = orders_df[orders_df['TYPE'] == "LIMIT_ORDER"]["SIZE"]

        if symbol not in limit_order_sizes_dict.keys():
            limit_order_sizes_dict[symbol] = limit_orders
        else:
            limit_order_sizes_dict[symbol] = limit_order_sizes_dict[symbol].append(limit_orders)

    return limit_order_sizes_dict


def fit_pomegranate_model(x, num_spikes, log_mu, log_sigma, model_file):
    """ Fits a generalised mixture model using the pomegranate library to 1D data.

    Mixture model is comprised of the following components:
      - A lognormal distribution
      - `num_spikes` very narrow Gaussians, centred on [100 * i for  i in range(1, num_spikes)]

    :param x: data to fit order-size distribution to, 1D numpy.ndarray
    :param num_spikes: Number of spikes in distribution, where the first spike is at x=100, with the n-th spike being at x=n*100
    :param log_mu: mu parameter of lognormal
    :param log_sigma: sigma parameter of lognormal
    :param model_file: path to file where model json is stored
    :return:
    """

    mixture_components = [LogNormalDistribution(log_mu, log_sigma)]
    for n in range(1, num_spikes + 1):
        dist = NormalDistribution(100 * n, Constants.norm_spike_sigma, frozen=True)
        mixture_components.append(dist)

    log_norm_weight = Constants.log_norm_initial_weight
    model_weights = np.array([log_norm_weight] + [log_norm_weight / num_spikes for _ in range(num_spikes)])
    model = GeneralMixtureModel(mixture_components, weights=model_weights)
    X = np.array(x).reshape((x.size, 1))

    print("Fitting Generalised Mixture Model...")
    model.fit(X,
              inertia=Constants.EM_inertia,
              verbose=True,
              n_jobs=-1)

    print("Generalised Mixture Model (GMM) fit params.")
    print(model.to_json())

    with open(model_file, 'w', encoding='utf-8') as f:
        f.write(model.to_json())

    return model


def normalise_pdf(p, x):
    """
        For plotting, takes pdf p evaluated at points x and divides by binwidth for visual aid against histograms

    """
    bins = np.diff(x)
    normalising_constant = np.sum(np.multiply(p[1:], bins))
    return np.true_divide(p, normalising_constant)


def make_plot_x_axis(xlim):
    xx = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), Constants.num_points_x_axis)
    hundreds = np.array([n * 100 for n in range(1, Constants.num_norm_spikes)])
    xx = np.concatenate((xx, hundreds))
    return np.sort(xx)


def plot_limit_order_sizes(limit_order_sizes_dict, output_dir, model_file, scale='log'):
    """ Plots histogram of the limit order sizes for symbols. """

    fig, ax = plt.subplots(figsize=(Constants.fig_width, Constants.fig_height))

    if scale == 'log':
        ax.set(xscale="log", yscale="log")

    ax.set_ylabel(Constants.limit_order_sizes_ylabel)
    ax.set_xlabel(Constants.limit_order_sizes_xlabel)

    symbols = list(limit_order_sizes_dict.keys())
    symbols.sort()
    colors = get_plot_colors(symbols)
    alphas = [1] * len(symbols)

    x_s = []

    for symbol, color, alpha in zip(symbols, colors, alphas):
        limit_order_sizes_series = limit_order_sizes_dict[symbol]
        x = limit_order_sizes_series.sort_values(ascending=True)
        x_s.append(x)
        plt.hist(x, bins="sqrt", density=True, label=symbol, color=color, alpha=alpha, histtype="step",
                 linewidth=Constants.limit_order_size_hist_linewidth)

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    xx = make_plot_x_axis(xlim)

    # # Plot fitted curves, leave out zeroes for better fit
    for x, symbol, color in zip(x_s, symbols, colors):
        model = fit_pomegranate_model(x, Constants.num_norm_spikes, Constants.log_mu, Constants.log_sigma, model_file)
        normed = normalise_pdf(model.probability(xx), xx)
        plt.plot(xx, normed, linestyle="-.", color=color,
                 label=f"{symbol} GMM fit", linewidth=Constants.limit_order_size_fit_linewidth)

    plt.legend(fontsize=Constants.legend_font_size)
    ax.set_ylim(ylim)

    plt.show()
    fig.savefig(f'{output_dir}/{Constants.limit_order_sizes_filename}.png', format='png', dpi=300,
                transparent=False, bbox_inches='tight', pad_inches=0.03)

    return


def set_up_plotting():
    """ Sets matplotlib variables for plotting. """
    plt.rc('xtick', labelsize=Constants.tick_label_size)
    plt.rc('ytick', labelsize=Constants.tick_label_size)
    plt.rc('legend', fontsize=Constants.legend_font_size)
    plt.rc('axes', labelsize=Constants.axes_label_font_size)


if __name__ == "__main__":

    # Create cache and visualizations folders if they do not exist
    try: os.mkdir("cache")
    except: pass
    try: os.mkdir("visualizations")
    except: pass
    try: os.mkdir("order_size_models")
    except: pass

    parser = argparse.ArgumentParser(description='Process order stream files and produce plots of order size (limit and executed).')
    parser.add_argument('targetdir', type=dir_path, help='Path of directory containing order stream files. Note that they must have been preprocessed'
                                                         ' by formatting scripts into format orders_{symbol}_{date_str}.pkl')
    parser.add_argument('-o', '--output-dir', default='visualizations', help='Path to plot output directory', type=dir_path)
    parser.add_argument('-f', '--model-file', default='limit_order_size_model.json', help='Path to output order size model file', type=Path)
    parser.add_argument('-z', '--recompute', action="store_true", help="Rerun computations without caching.")
    args, remaining_args = parser.parse_known_args()

    bundled_orders_dict = unpickle_stream_dfs_to_stream_list(args.targetdir)

    print("### Order size stylized facts plots ###")

    ## limit order sizes
    pickled_bundled_limit_order_sizes_dict = "cache/bundled_limit_order_sizes_dict.pkl"
    if (not os.path.exists(pickled_bundled_limit_order_sizes_dict)) or args.recompute:
        print("Computing limit order sizes...")
        bundled_limit_order_sizes_dict = bundled_stream_limit_order_sizes(bundled_orders_dict)
        pickle.dump(bundled_limit_order_sizes_dict, open(pickled_bundled_limit_order_sizes_dict, "wb"))
    else:
        bundled_limit_order_sizes_dict = pickle.load(open(pickled_bundled_limit_order_sizes_dict, "rb"))

    print("Plotting limit order sizes...")
    plot_limit_order_sizes(bundled_limit_order_sizes_dict, args.output_dir, args.model_file)
