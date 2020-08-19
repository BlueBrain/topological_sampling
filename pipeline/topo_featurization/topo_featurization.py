#!/usr/bin/env python
import json
import os
import importlib
import pandas
import numpy as np
import progressbar

from scipy import sparse

from toposample import config, TopoData
from toposample.data.data_structures import ConditionCollection
from toposample.indexing import GidConverter


def read_input(input_config):
    spiketrains = np.load(input_config["split_spikes"], allow_pickle=True)
    tribes = TopoData(input_config["tribes"])
    adj_matrix = sparse.load_npz(input_config["adjacency_matrix"])
    neuron_info = pandas.read_pickle(input_config["neuron_info"])
    return spiketrains, tribes, adj_matrix, neuron_info


def write_output(data, output_config):
    fn_out = output_config["features"]
    if os.path.exists(fn_out):
        with open(fn_out, "r") as fid:
            existing = json.load(fid)
        for k, v in data.items():
            existing.setdefault(k, {}).update(v)
        with open(fn_out, "w") as fid:
            json.dump(existing, fid, indent=2)
    else:
        with open(fn_out, 'w') as fid:
            json.dump(data, fid, indent=2)


def split_into_t_bins(spike_data, t_bins, n_t_bins):
    """
    Splits spiking data into regular time bins of specified size.
    :param spike_data: n x 2 array. First column: spike time; second column spiking neuron gid
    :param t_bins: The edges of time bins to use.
    :param n_t_bins: Number of time bins (len(t_bins) - 1).
    :return: List of m x 2 numpy arrays for the individual time bins (in order). sum(m_i) <= n
    """
    bin_ids = np.digitize(spike_data[:, 0], t_bins)
    return [spike_data[bin_ids == i, 1].astype(int)
            for i in range(1, n_t_bins + 1)]


def generate_spikers(spiketrains, t_bins, n_t_bins):
    """
    generate_spikers: Which gids fire a spike in each time bin of each trial?
    :param spiketrains: Spiking data, split first by stimulus id, then by trial. Each entry a numpy.array with
    shape n_spikes x 2. First column: time of spike; second column identifier of spiking neuron
    :param t_bins: edges of time bins to use
    :param n_t_bins: len(t_bins) - 1
    :return: list of lists of lists of numpy.arrays. First index: Stimulus id; second index: trial;
    third index: time bin; entries: array of identifiers of spiking neurons
    """
    res = []
    for spikes_for_stim in spiketrains:
        res.append([])
        for spikes_for_trial in spikes_for_stim:
            res[-1].append(split_into_t_bins(spikes_for_trial, t_bins, n_t_bins))
    return res


def translate_spiketrains_to_local_id(spiketrains, converter):
    for per_stimulus in spiketrains:
        for train in per_stimulus:
            train[:, 1] = converter.indices(train[:, 1])


def make_topo_features_for_tribes(spiketrains, t_bins, parameter, adj_matrix, converter):
    """
    :param spiketrains: Spiking data, split first by stimulus id, then by trial. Each entry a numpy.array with
    shape n_spikes x 2. First column: time of spike; second column gid of spiking neuron
    :param t_bins: edges of time bins to use
    :param parameter: No idea
    :param adj_matrix: adjacency matrix of the entire population
    :param converter: toposample.indexing.GidConverter object
    :return: a function object that can be used in an unpool operation to generate feature arrays for each stimulus.
    Adds the "stimulus" condition. individual data is an array of shape n_t_bins x 1 x n_trials
    """
    translate_spiketrains_to_local_id(spiketrains, converter)  # Using indices into adj_matrix instead of gids.
    spikers = generate_spikers(spiketrains, t_bins, len(t_bins) - 1)

    # Dynamic import of specified analyzis module
    import_root = os.path.split(__file__)[0]
    sys.path.insert(0, import_root)
    module = importlib.import_module(parameter["source"])

    def topo_features_for_tribes(tribal_gids):  # target shape: t_bins x 1 x trials
        tribal_ids = converter.indices(tribal_gids.res)  # Using indices into adj_matrix instead of gids.
        print("Featurizing for tribe with {0} gids".format(len(tribal_ids)))
        for stim_id, stim_spikers in enumerate(spikers):
            for_stim = []
            print("\t{0} repetitions of stimulus {0}".format(len(stim_spikers), stim_id))
            pbar = progressbar.ProgressBar()
            for trial_spikers in pbar(stim_spikers):
                tribe_per_t_bin = [np.intersect1d(tribal_ids, _spikers)
                                   for _spikers in trial_spikers]
                mat_per_t_bin = [adj_matrix[np.ix_(_tribe, _tribe)]
                                 for _tribe in tribe_per_t_bin]
                t_series = [module.compute(active_mat, **parameter["kwargs"])
                            for active_mat in mat_per_t_bin]
                for_stim.append([t_series])
            for_stim = np.array(for_stim).transpose()
            yield for_stim, {"stimulus": stim_id}

    return topo_features_for_tribes


def make_write_h5(output_root):
    """
    :param output_root: Root directory to write the data into
    :return: a function object that writes analysis results into an .h5 file. Location of that file is under
    output_root and determined by specified "sampling", "specifier" and "index" conditions. Returns that path.
    """
    import h5py

    def write_h5(*args):
        for sampling, specifier, index, feature_data in zip(*args):
            out_path = os.path.join(output_root, sampling, specifier, index)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out_fn = os.path.join(out_path, "results.h5")
            with h5py.File(out_fn, "w") as h5:
                grp = h5.create_group("per_stimulus")
                for stim_id, features in enumerate(feature_data):
                    grp.create_dataset("stim{0}".format(stim_id), data=features)
            yield {"data_fn": out_fn}, {"sampling": sampling, "specifier": specifier, "index": index}
    return write_h5


def get_idv_label(tribal_data, label_str="idv_label"):
    """
    :param tribal_data: TopoData object holding data about the sampled tribes
    :return: a ConditionCollection object holding further data about the sampled tribes (such as tribal chiefs) that
    is supposed to be inherited by the output of this stage.
    """
    def make_idv_label_dict(some_data):
        return {label_str: some_data}
    data_points = []
    for lbl in ['chief', 'parent', 'center_offset']:
        data_points.extend(tribal_data[lbl].map(make_idv_label_dict).contents)
    return ConditionCollection(data_points)


def ordered_list(idxx, data):
    return [data[i] for i in np.argsort(idxx)]


def main(path_to_config, **kwargs):
    assert "index" not in kwargs, "Splitting by index not supported for topo_featurization!"
    # 1. Evaluate configuration.
    # Read the meta-config file
    cfg = config.Config(path_to_config)
    # Get configuration related to the current pipeline stage
    stage = cfg.stage("topological_featurization")
    topo_featurization_cfg = stage["config"]
    timebin = topo_featurization_cfg["time_bin"]
    parameter = topo_featurization_cfg[topo_featurization_cfg["topo_method"]]
    # number of time steps per trial
    stim_dur = cfg.stage('split_spikes')['config']['stim_duration_ms']
    n_t_bins = int(stim_dur / timebin)
    t_bins = np.arange(n_t_bins + 1) * timebin

    # 2. Read input data
    spiketrains, tribal_data, adj_matrix, neuron_info = read_input(stage["inputs"])
    tribes = tribal_data["gids"]
    tribes = tribes.filter(**kwargs)

    # 3. Analyze.
    # Create analysis function, given the spikes and time bins
    featurization_func = make_topo_features_for_tribes(spiketrains, t_bins, parameter,
                                                       adj_matrix, GidConverter(neuron_info))
    # unpool with this function adds the additional condition of "stimulus" (stimulus identifier)
    tribes.unpool(func=featurization_func)  # shape of data: t_bins x 1 x trials
    # Now put the data into the expected format. First pooling along tribes (index).
    features_data = tribes.pool(["index"], func=np.hstack)  # shape of data: t_bins x tribe_index x trials
    # Then pooling along different stimuli
    features_data = features_data.pool(["stimulus"], func=ordered_list, xy=True)
    # Features that have been removed in the filter step need to be added back for the expected format.
    for k, v in kwargs.items():
        features_data.add_label(k, v)
    # The format also needs an "index" condition. We pooled that away, so we just add to what remains index=0
    features_data.add_label("index", "0")
    # transform writes data into individual hdf5 files and returns their paths.
    fn_data = features_data.transform(["sampling", "specifier", "index"],  # data: str (path to .h5 file)
                                      func=make_write_h5(stage['other']), xy=True)
    # There is some additional info about the neuron samples that we want to inherit from the "tribes" structure.
    # So we add that info to fn_data
    fn_data.extended_map(lambda x, y: x.update(y[0]), [get_idv_label(tribal_data)])
    write_output(TopoData.condition_collection_to_dict(fn_data), stage["outputs"])


def parse_filter_arguments(*args):
    fltr_dict = {}
    for arg in args:
        if "=" in arg:
            splt = arg.split("=")
            fltr_dict[splt[0]] = splt[1]
    return fltr_dict


if __name__ == "__main__":
    import sys
    main(sys.argv[1], **parse_filter_arguments(*sys.argv[2:]))
