import numpy as np
import pandas as pd
import os
import importlib
import sys

from scipy import sparse

from toposample import config


def read_input(input_config):
    adj_matrix = sparse.load_npz(input_config["adjacency_matrix"])
    neuron_info = pd.read_pickle(input_config["neuron_info"])
    return adj_matrix, neuron_info


def write_output(DB, output_config):
    DB.to_pickle(output_config["database"])


def add_tribes(adj_matrix):  # For now: indices, not neuron gids
    tribes = []

    # Add transpose to get both in- and out-neighbors
    M = adj_matrix.transpose() + adj_matrix
    # Since we iterate rows, we want csr
    if M.format != 'csr':
        M = M.asformat('csr')

    for m in M:
        tribes.append(m.indices)
    return tribes


def add_neuron_info(DB, neuron_info):
    for col in neuron_info.columns:
        if col in DB:
            raise Exception("Duplicate column!")
        DB[col] = neuron_info[col]


def main(path_to_config):
    # Initialize empty database
    DB = pd.DataFrame()

    # Read the meta-config file
    cfg = config.Config(path_to_config)

    # Get configuration related to the current pipeline stage
    stage = cfg.stage("gen_topo_db")

    # Fetch adjacency matrix and neuron infos according to configured locations/files
    adj_matrix, neuron_info = read_input(stage["inputs"])
    assert adj_matrix.shape[0] == len(neuron_info), "Neuron info and adjacency matrix have incompatible sizes!"
    topo_db_cfg = stage["config"]
    precision = topo_db_cfg["precision"]

    # Compute all the tribes
    DB["tribe"] = add_tribes(adj_matrix)

    # Use gids as index
    # Note: This assumes that the order in neuron_info and the adj_matrix are the same!
    DB.index = neuron_info.index

    # Add columns of neuron_info into the DB
    add_neuron_info(DB, neuron_info)

    # Loop over the tribe parameters as listed in topo_db_config, each loop makes a call to the associated
    # function computing the parameter values and injects into DB.
    import_root = os.path.split(__file__)[0]
    sys.path.insert(0, import_root)
    for parameter in topo_db_cfg["parameters"]:
        module = importlib.import_module(topo_db_cfg[parameter]["source"])
        DB[topo_db_cfg[parameter]["column_name"]] = module.compute(DB["tribe"], adj_matrix, precision)

    write_output(DB, stage["outputs"])


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
