import numpy
import pandas

from scipy.stats import pearsonr

from toposample import Config, TopoData
from toposample.data import read_h5_dataset
from toposample.db import get_column_from_database

from parameters_for_tribes import top_n_weighted_average, tribal_spectrum


def read_inputs(cfg):
    db_fn = cfg._cfg["analyzed"]["database"]
    acc_fn = cfg._cfg["analyzed"]["classifier_components_results"]
    tribes_fn = cfg._cfg["analyzed"]["tribes"]
    stage_cfg = cfg.stage("struc_tribe_analysis")["config"]
    db = pandas.read_pickle(db_fn)
    acc_data = TopoData(acc_fn, follow_link_functions={"data_fn": (read_h5_dataset("scores"), False)})
    acc_data = acc_data["data_fn"].filter(sampling="Radius").map(numpy.nanmean)
    gids = TopoData(tribes_fn)["gids"].filter(sampling="Radius")
    return stage_cfg, db, acc_data, gids


def write_back(cfg, stage_cfg):
    import json
    struc_analysis_fn = cfg._cfg["config"]["struc_tribe_analysis"]
    with open(struc_analysis_fn, "w") as fid:
        json.dump(stage_cfg, fid, indent=2)


def specified_parameter_spec(stage_cfg, param_names):
    ret = []
    for param_spec in stage_cfg["Parameters"]:
        if param_spec["name"] in param_names:
            print("Will optimize: {0}".format(param_spec["name"]))
            ret.append(param_spec)
        else:
            print("Not optimizing: {0}".format(param_spec["name"]))
    print("\toptimizing {0} parameters...".format(len(ret)))
    return ret


def evaluate_fit(input_data):
    input_data = numpy.vstack(input_data)
    return pearsonr(input_data[:, 0], input_data[:, 1])[0]


def find_best_parameter(db, acc_data, gids, param_specs):
    from scipy.optimize import minimize_scalar
    print("Calculating size of tribal overlaps...")
    overlap_sizes = gids.map(lambda g: tribal_spectrum(db, g)[1])

    res_dict = {}
    for param_spec in param_specs:
        if param_spec["prediction_strategy"]["strategy"] != "weighted_mean_by_overlap":
            continue

        print("Trying to optimize {0}".format(param_spec["name"]))
        v = numpy.array(get_column_from_database(db, param_spec["value"]["column"],
                                                 index=param_spec["value"].get("index", None),
                                                 function=param_spec["value"].get("function", None)))

        def func_to_minimize(N):
            prediction = overlap_sizes.map(lambda w: top_n_weighted_average(w, v, number_sampled=N))
            data_to_evaluate = acc_data.extended_map(lambda a, b: [a] + b, [prediction]).get()
            return -numpy.abs(evaluate_fit(data_to_evaluate))

        print("Calling minimizer")
        res = minimize_scalar(func_to_minimize, bounds=[1, len(v)], method='bounded',
                              options={"maxiter": 200, "disp": True})
        res_dict[param_spec["name"]] = res
    return res_dict


def plot_parameter_fit_quality_curve(db, acc_data, gids, param_specs, res_dict, plot_x=None):
    from matplotlib import pyplot as plt
    from matplotlib import cm
    if plot_x is None:
        plot_x = numpy.linspace(1, len(db), 51).astype(int)
        #  numpy.logspace(1, numpy.log10(len(db)), 21).astype(int)
    print("Calculating size of tribal overlaps...")
    overlap_sizes = gids.map(lambda g: tribal_spectrum(db, g)[1])

    fig = plt.figure()
    ax = fig.gca()
    cols = cm.hsv(numpy.linspace(0.0, 1.0, len(param_specs)))
    for param_spec, col in zip(param_specs, cols):
        if param_spec["prediction_strategy"]["strategy"] != "weighted_mean_by_overlap":
            continue

        print("Plotting curve for {0}".format(param_spec["name"]))
        v = numpy.array(get_column_from_database(db, param_spec["value"]["column"],
                                                 index=param_spec["value"].get("index", None),
                                                 function=param_spec["value"].get("function", None)))
        plot_y = []
        x = plot_x.tolist()
        print("Getting data points")

        opt_x = int(res_dict[param_spec["name"]].x)
        opt_y = None
        insert_place = numpy.nonzero(plot_x > opt_x)[0]
        if len(insert_place) > 0:
            x.insert(insert_place[0], opt_x)
        else:
            x.append(opt_x)

        for N in x:
            prediction = overlap_sizes.map(lambda w: top_n_weighted_average(w, v, number_sampled=N))
            data_to_evaluate = acc_data.extended_map(lambda a, b: [a] + b, [prediction]).get()
            plot_y.append(evaluate_fit(data_to_evaluate))
            if N == opt_x:
                opt_y = plot_y[-1]

        print(plot_y)
        ax.plot(x, plot_y, label=param_spec["name"], color=col, lw=0.75)
        ax.plot(opt_x, opt_y, 'o', color=col)
    ax.legend()
    ax.set_xlabel("Number of tribes used")
    ax.set_ylabel("Abs. correlation with classifier accuracy (Pearsonr)")

    return fig


def update_stage_cfg(stage_cfg, res_dict):
    for param in stage_cfg["Parameters"]:
        if param["name"] in res_dict:
            entry = res_dict[param["name"]]
            if entry.success:
                param["prediction_strategy"]["kwargs"] = {
                    "number_sampled": int(entry.x)
                }


def main(cfg_fn, param_names=None, write_optimal_parameters=True, curve_plot_fn=None):
    cfg = Config(cfg_fn)
    stage_cfg, db, acc_data, gids = read_inputs(cfg)
    if (param_names is None) or (len(param_names) == 0):
        param_specs = stage_cfg["Parameters"]
    else:
        param_specs = specified_parameter_spec(stage_cfg, param_names)
    res_dict = find_best_parameter(db, acc_data, gids, param_specs)
    if write_optimal_parameters:
        update_stage_cfg(stage_cfg, res_dict)
        write_back(cfg, stage_cfg)
    if curve_plot_fn is not None:
        fig = plot_parameter_fit_quality_curve(db, acc_data, gids, param_specs, res_dict)
        fig.savefig(curve_plot_fn)


if __name__ == "__main__":
    import sys
    import os
    import getopt

    opts, args = getopt.getopt(sys.argv[1:], "o:w:")
    opts = dict(opts)
    if len(args) == 0:
        print("""Usage:
        {0} (-o plot_output_filename) (-w 0) path_to_common_config param1 param2 param3...""".format(sys.argv[0]))
        sys.exit(2)
    if not os.path.isfile(args[0]):
        print("Common config not found at {0}".format(args[0]))
    main(args[0], param_names=args[1:], write_optimal_parameters=bool(int(opts.get("-w", True))),
         curve_plot_fn=opts.get("-o", None))
