#!/usr/bin/env python
"""
NEUTRINO - NEUral TRIbe and Network Observer
Copyright (C) 2020 Blue Brain Project / EPFL

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy
import pandas as pd
import os
import importlib

from scipy import sparse

from toposample import config
from toposample.indexing import GidConverter


def read_input(input_config):
    adj_matrix = sparse.load_npz(input_config["adjacency_matrix"])
    neuron_info = pd.read_pickle(input_config["neuron_info"])
    return adj_matrix, neuron_info


# noinspection PyPep8Naming
def write_output(DB, output_fn):
    DB.to_pickle(output_fn)


def calculate_tribes(adj_matrix, neuron_info):
    """
    Calculate tribes, i.e. all neurons adjacent toa given neuron
    :param adj_matrix: scipy.sparse matrix - adjacency matrix of the neuron population
    :param neuron_info: pandas.DataFrame - additional neuron info; its _index_ will be used to identify tribal neurons
    :return: pandas.DataFrame with a single column "tribe" that has the identifiers of tribal neurons
    """
    converter = GidConverter(neuron_info)
    tribes = []
    tribal_db = pd.DataFrame(numpy.empty((len(neuron_info), 0)),
                             index=neuron_info.index)

    # Add transpose to get both in- and out-neighbors
    M = adj_matrix.transpose() + adj_matrix
    # Add diagonal such the each chief is part of its tribe
    M[numpy.diag_indices_from(M)] = True
    # Since we iterate rows, we want csr
    if M.format != 'csr':
        M = M.asformat('csr')

    for m in M:
        tribes.append(converter.gids(m.indices))
    tribal_db["tribe"] = tribes
    return tribal_db


# noinspection PyPep8Naming
def add_neuron_info(DB, neuron_info):
    """
    Copies columns from one DataFrame into another. Checks for duplicate columns
    :param DB: pandas.DataFrame to paste the columns of neuron_info into
    :param neuron_info: pandas.Dataframe to copy columns from
    :return:
    """
    for col in neuron_info.columns:
        if col in DB:
            raise Exception("Duplicate column!")
        DB[col] = neuron_info[col]


# noinspection PyPep8Naming,PyUnresolvedReferences
def add_parameter_column(DB, tribes, parameter, topo_db_cfg, conv, adj_matrix):
    """
    Loop over the tribe parameters as specified, each loop makes a call to the associated
    function computing the parameter values and injects into DB.
    :param DB: pandas.DataFrame - to put the results into
    :param tribes: pandas.DataFrame - with a column "tribe" that holds gids of associated tribes. Can be the same as DB
    :param parameter: str - parameter name
    :param topo_db_cfg: dict - configuration of the gen_topo_db step
    :param conv: GidConverter
    :param adj_matrix: scipy.sparse.csr_matrix - adjacency matrix of circuit
    :return: None - puts results into DB
    """
    precision = topo_db_cfg["precision"]

    assert parameter in topo_db_cfg["parameters"], "Parameter {0} not in config!".format(parameter)
    print("Calculating {0} for all tribes...".format(parameter))
    try:
        module = importlib.import_module(topo_db_cfg[parameter]["source"])
        DB[topo_db_cfg[parameter]["column_name"]] = module.compute(tribes["tribe"], adj_matrix, conv, precision)
    except ImportError as e:
        print(e)
        print("Unable to load module for {0}".format(parameter))


def create_db_with_specified_columns(lst_columns, tribes, neuron_info, topo_db_cfg, adj_matrix):
    """
    Create a topological database with specified parameters
    :param lst_columns: list - list of parameters to populate the DB with
    :param tribes: pandas.DataFrame - with a column "tribe" that has gids of neurons of the associated tribe
    :param neuron_info: pandas.DataFrame - with additional info about the neurons
    :param topo_db_cfg: dict - configuration of the gen_topo_db step
    :param adj_matrix: scipy.sparse.csr_matrix - adjacency matrix of circuit
    :return: pandas.DataFrame holding all specified parameters in the columns
    """
    import_root = os.path.split(__file__)[0]
    sys.path.insert(0, import_root)

    # Initialize empty database of required length
    DB = pd.DataFrame(numpy.empty((len(neuron_info), 0)))
    # Use gids also as index of the DB
    # Note: This assumes that the order in neuron_info and the adj_matrix are the same!
    DB.index = neuron_info.index
    conv = GidConverter(neuron_info)

    for column_name in lst_columns:
        if column_name == "tribe":
            DB["tribe"] = tribes["tribe"]
        elif column_name == "neuron_info":
            add_neuron_info(DB, neuron_info)
        else:
            add_parameter_column(DB, tribes, column_name, topo_db_cfg, conv, adj_matrix)
    return DB


def main(path_to_config, parameter_name=None):
    # Read the meta-config file
    cfg = config.Config(path_to_config)

    # Get configuration related to the current pipeline stage
    stage = cfg.stage("gen_topo_db")

    # Fetch adjacency matrix and neuron infos according to configured locations/files
    adj_matrix, neuron_info = read_input(stage["inputs"])
    assert adj_matrix.shape[0] == len(neuron_info), "Neuron info and adjacency matrix have incompatible sizes!"
    topo_db_cfg = stage["config"]
    # Calculate tribes, i.e. gids of adjacent neurons
    tribes = calculate_tribes(adj_matrix, neuron_info)

    # Populate DB
    if parameter_name is None:  # Case 1: generate all columns at once
        DB = create_db_with_specified_columns(["tribe", "neuron_info"] + topo_db_cfg["parameters"],
                                              tribes, neuron_info, topo_db_cfg, adj_matrix)
        # Write output to where it's meant to go
        write_output(DB, stage["outputs"]["database"])
    else:  # Case 2: Generate one single column at a time
        DB = create_db_with_specified_columns([parameter_name], tribes, neuron_info, topo_db_cfg, adj_matrix)
        suffix = "." + parameter_name.lower().replace(" ", "_")  # Use parameter name as suffix for output file
        # Write output to the 'other' directory for later merging
        if not os.path.exists(stage["other"]):
            os.makedirs(stage["other"])
        out_fn = os.path.join(stage["other"], os.path.split(stage["outputs"]["database"])[1]) + suffix
        write_output(DB, out_fn)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        main(sys.argv[1], parameter_name=sys.argv[2])
    else:
        main(sys.argv[1])
