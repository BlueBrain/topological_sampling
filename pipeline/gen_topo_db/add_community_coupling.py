import numpy
import pandas
import progressbar
from scipy import sparse
from toposample import Config
from toposample.indexing import GidConverter


def read_inputs(input_cfg):
    neuron_info = pandas.read_pickle(input_cfg["neuron_info"])
    conv = GidConverter(neuron_info)
    spikes = numpy.load(input_cfg["raw_spikes"])
    return conv, spikes


def calculate_community_coupling(spikes, conv, stage_config):
    t_bins = numpy.arange(0, spikes[:, 0].max() + stage_config["coupling_bin_size"], stage_config["coupling_bin_size"])
    t_idxx = numpy.digitize(spikes[:, 0], bins=t_bins) - 1
    n_idxx = conv.indices(spikes[:, 1].astype(int))

    M = sparse.coo_matrix((numpy.ones_like(t_idxx, dtype=bool), (n_idxx, t_idxx)),
                          shape=(len(conv._index), len(t_bins) - 1))
    M = M.asformat("csr")

    fr = numpy.array(M.mean(axis=0))[0]
    coeff = []
    pbar = progressbar.ProgressBar()
    for idxx in pbar(range(M.shape[0])):
        firing_idv = M[idxx].toarray().astype(float)[0]
        firing_pop = fr - firing_idv / M.shape[0]
        coeff.append(numpy.corrcoef(firing_pop, firing_idv)[1, 0])
    return numpy.array(coeff)


def write_output(DB, output_fn):
    DB.to_pickle(output_fn)


def main(path_to_config):
    # Read the meta-config file
    cfg = Config(path_to_config)

    # Get configuration related to the current pipeline stage
    stage = cfg.stage("gen_topo_db")

    conv, spikes = read_inputs(stage["inputs"])
    topo_db_cfg = stage["config"]

    coupling = calculate_community_coupling(spikes, conv, topo_db_cfg)

    existing_db = pandas.read_pickle(stage["outputs"]["database"])
    existing_db["comm_coupling"] = coupling
    existing_db["tribe_comm_coupling"] = [coupling[conv.indices(existing_db['tribe'].iloc[i])]
                                          for i in range(len(existing_db))]

    write_output(existing_db, stage["outputs"]["database"])


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
