import json
import os
import numpy as np

from toposample import config, TopoData


def Flagsering(*args):
    """ Place holder for testing"""
    return np.random.rand()


def read_input(input_config):
    spiketrains = np.load(input_config["split_spikes"], allow_pickle=True)
    tribes = TopoData(input_config["tribes"])
    tribal_chiefs = tribes["chief"]
    tribal_gids = tribes["gids"]
    return spiketrains, tribal_gids


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
    bin_ids = np.digitize(spike_data[:, 0], t_bins)
    return [spike_data[bin_ids == i, 1].astype(int)
            for i in range(1, n_t_bins + 1)]


def generate_spikers(spiketrains, t_bins, n_t_bins):
    """
    generate_spikers: Which gids fire a spike in each time bin of each trial?
    :param spiketrains: Spiking data, split first by stimulus id, then by trial. Each entry a numpy.array with
    shape n_spikes x 2. First column: time of spike; second column gid of spiking neuron
    :param t_bins: edges of time bins to use
    :param n_t_bins: len(t_bins) - 1
    :return: list of lists of lists of numpy.arrays. First index: Stimulus id; second index: trial;
    third index: time bin; entries: array of spiking gids
    """
    res = []
    for spikes_for_stim in spiketrains:
        res.append([])
        for spikes_for_trial in spikes_for_stim:
            res[-1].append(split_into_t_bins(spikes_for_trial, t_bins, n_t_bins))
    return res


def make_topo_features_for_tribes(spiketrains, t_bins, parameter):
    """
    :param spiketrains: Spiking data, split first by stimulus id, then by trial. Each entry a numpy.array with
    shape n_spikes x 2. First column: time of spike; second column gid of spiking neuron
    :param t_bins: edges of time bins to use
    :param parameter: No idea
    :return: a function object that can be used in an unpool operation to generate feature arrays for each stimulus.
    Adds the "stimulus" condition. individual data is an array of shape n_t_bins x 1 x n_trials
    """
    spikers = generate_spikers(spiketrains, t_bins, len(t_bins) - 1)

    def topo_features_for_tribes(tribal_gids):  # target shape: t_bins x 1 x trials
        tribal_gids = tribal_gids.res
        print("Featurizing for tribe with {0} gids".format(len(tribal_gids)))
        for stim_id, stim_spikers in enumerate(spikers):
            for_stim = []
            for trial_spikers in stim_spikers:
                tribe_per_t_bin = [np.intersect1d(tribal_gids, _spikers)
                                   for _spikers in trial_spikers]
                t_series = [Flagsering(parameter, active_tribe)
                            for active_tribe in tribe_per_t_bin]
                for_stim.append([t_series])
            for_stim = np.array(for_stim).transpose()
            yield for_stim, {"stimulus": stim_id}

    return topo_features_for_tribes


def make_write_h5(output_root):
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
            yield out_fn, {"sampling": sampling, "specifier": specifier, "index": index}
    return write_h5


def ordered_list(idxx, data):
    return [data[i] for i in np.argsort(idxx)]


def main(path_to_config, **kwargs):
    # Read the meta-config file
    cfg = config.Config(path_to_config)

    # Get configuration related to the current pipeline stage
    stage = cfg.stage("topological_featurization")

    # Fetch spiketrains
    spiketrains, tribes = read_input(stage["inputs"])
    tribes = tribes.filter(**kwargs)

    topo_featurization_cfg = stage["config"]

    timebin = topo_featurization_cfg["time_bin"]
    parameter = topo_featurization_cfg["topo_method"]  # Is this what parameter is supposed to be?

    # number of time steps per trial
    stim_dur = cfg.stage('split_spikes')['config']['stim_duration_ms']
    n_t_bins = int(stim_dur / timebin)
    t_bins = np.arange(n_t_bins + 1) * timebin

    # this function generates features for trials and time bins
    featurization_func = make_topo_features_for_tribes(spiketrains, t_bins, parameter)
    # unpool adds the additional condition of "stimulus" (stimulus identifier)
    tribes.unpool(func=featurization_func)  # shape of data: t_bins x 1 x trials
    features_data = tribes.pool(["index"], func=np.hstack)  # shape of data: t_bins x tribe_index x trials
    features_data = features_data.pool(["stimulus"], func=ordered_list, xy=True)
    for k, v in kwargs.items():
        features_data.add_label(k, v)
    # adds back the "index" condition, though with only a single value.
    # This is required to fulfill specs of the TopoData format.
    features_data.add_label("index", "0")
    # transform writes data into individual hdf5 files and returns their paths.
    fn_data = features_data.transform(["sampling", "specifier", "index"],
                                      func=make_write_h5(stage['other']), xy=True)

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
