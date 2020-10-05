import numpy
from matplotlib import pyplot as plt


def plot_linear_fit_results(fit_results, plot_params=None, same_axes=False,
                            bar_col=[0.3, 0.8, 0.3], var_col="blue"):
    if plot_params is None:
        plot_params = fit_results.keys()
    
    plot_r = [fit_results[k]["slope"] for k in plot_params]
    plot_v = [fit_results[k]["var_expl"] for k in plot_params]
    adjusted_threshs = numpy.array([1E-2, 1E-3, 1E-4]) / len(plot_params)
    plot_is_sig = [numpy.sum(fit_results[k]["pvalue"] < adjusted_threshs)
                   for k in plot_params]

    if same_axes:
        fig = plt.figure(figsize=(len(plot_params) * 0.3 + 0.2, 2.2));
        ax1 = fig.gca()
        plt.box(False)
        ax2 = fig.add_axes(ax1.get_position(), facecolor="None")
        plt.box(False)
    else:
        fig = plt.figure(figsize=(len(plot_params) * 0.6 + 0.4, 2.2))
        ax1 = fig.add_axes((0.1, 0.1, 0.375, 0.8))
        plt.box(False)
        ax2 = fig.add_axes((0.55, 0.1, 0.375, 0.8))
        plt.box(False)
    ax2.yaxis.set_ticks_position("right")
    ax2.tick_params(axis='y', colors=var_col)
    ax2.yaxis.label.set_color(var_col)
    ax2.yaxis.set_label_position("right")
    ax1.bar(numpy.arange(len(plot_params)),
            plot_r, color=bar_col)
    if same_axes:
        ax2.plot(numpy.arange(len(plot_params)), plot_v, ls="None", marker='d', color=var_col)
    else:
        ax2.bar(numpy.arange(len(plot_params)), plot_v, color=var_col)
    txt_y = numpy.nanmax(plot_r) * 1.05
    for i, is_sig in enumerate(plot_is_sig):
        if is_sig > 0:
            ax1.text(i, txt_y, "*" * is_sig, color="black", horizontalalignment="center", rotation=45)
    ax1.set_xticks(range(len(plot_params)))
    _ = ax1.set_xticklabels(plot_params, rotation='vertical')
    ax2.set_xticks([])
    ax1.set_ylabel("Fit slope")
    ax2.set_ylabel("Additional var expl.")
    ax2.set_xlim(ax1.get_xlim())
