#!/usr/bin/env python
"""
toposampling - Topology-assisted sampling and analysis of activity data
Copyright (C) 2020 Blue Brain Project / EPFL & University of Aberdeen

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy
import pandas as pd
import os
import importlib
import json

from scipy import sparse

from toposample import config, TopoData
from toposample.db import get_column_from_database
from toposample.indexing import GidConverter


def read_input(input_config):
    adj_matrix = sparse.load_npz(input_config["adjacency_matrix"])
    neuron_info = pd.read_pickle(input_config["neuron_info"])
    tribes = TopoData(input_config["tribes"])
    return adj_matrix, neuron_info, tribes


# noinspection PyPep8Naming
def write_output(param_dict, output_fn):
    assert not os.path.exists(output_fn)
    with open(output_fn, "w") as fid:
        json.dump(param_dict, fid, indent=2)


# noinspection PyPep8Naming,PyUnresolvedReferences
def add_single_parameter_column(DB, gids, parameter, topo_db_cfg, conv, adj_matrix):
    """
    Loop over the tribe parameters as specified, each loop makes a call to the associated
    function computing the parameter values and injects into DB.
    :param gids: pandas.DataFrame - with a column "gids" that holds gids of associated tribes. Can be the same as DB
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
        col = module.compute(gids["gids"], adj_matrix, conv, precision)
        DB[topo_db_cfg[parameter]["column_name"]] = col
    except ImportError as e:
        print(e)
        print("Unable to load module for {0}".format(parameter))


def generage_parameter_dict_from_db(DB, tribe_conditions, struc_analysis_cfg):
    out_dict = {"Radius": {}}
    set_dict = out_dict["Radius"]
    for param in struc_analysis_cfg["Parameters"]:
        pval = param["value"]
        col = get_column_from_database(DB, pval["column"], index=pval.get("index", None),
                                       function=pval.get("function", None))
        for cond, value in zip(tribe_conditions, col):
            set_dict.setdefault(cond["specifier"], {})[cond["index"]] = value
    return out_dict


def get_parameter_db_for_samples(tribes, neuron_info, topo_db_cfg,
                                 struc_analysis_cfg, adj_matrix):
    """
    Create a topological database with specified parameters
    :param tribes: TopoData - contains the specifications of volumetric samples in tribes["gids"]
    :param neuron_info: pandas.DataFrame - with additional info about the neurons
    :param topo_db_cfg: dict - configuration of the gen_topo_db step
    :param struc_analysis_cfg: dict - configuration of the struc_tribe_analysis step
    :param adj_matrix: scipy.sparse.csr_matrix - adjacency matrix of circuit
    :return: pandas.DataFrame holding all specified parameters in the columns
    """
    import_root = os.path.join(os.path.split(__file__)[0], "..", "gen_topo_db")
    sys.path.insert(0, import_root)

    topo_lookup = dict([(topo_db_cfg[_param]["column_name"], _param)
                        for _param in topo_db_cfg['parameters']])

    conv = GidConverter(neuron_info)
    gids_struc = tribes["gids"].filter(sampling="Radius")
    gids_hack = [_struc.res for _struc in gids_struc.contents]
    tribe_conditions = [_struc.cond for _struc in gids_struc.contents]
    gids_dframe = pd.DataFrame(numpy.empty(shape=(len(gids_hack), 0)))
    gids_dframe["gids"] = gids_hack
    DB = pd.DataFrame(numpy.empty((len(gids_hack), 0)))

    out_dict = {"Radius": {}}
    set_dict = out_dict["Radius"]
    for param in struc_analysis_cfg["Parameters"]:
        pval = param["value"]
        if param["name"] in struc_analysis_cfg["Exclude for volumetric"]:
            continue
        if pval["column"] not in DB:
            parameter = topo_lookup[pval["column"]]
            add_single_parameter_column(DB, gids_dframe, parameter, topo_db_cfg, conv, adj_matrix)
        col = get_column_from_database(DB, pval["column"], index=pval.get("index", None),
                                       function=pval.get("function", None))
        for cond, value in zip(tribe_conditions, col):
            set_dict.setdefault(cond["specifier"], {}).setdefault(cond["index"], {})[param["name"]] = float(value)
    return out_dict


def main(path_to_config):
    # Read the meta-config file
    cfg = config.Config(path_to_config)

    # Get configuration related to the current pipeline stage
    stage = cfg.stage("struc_tribe_analysis")
    topo_db_cfg = cfg.stage("gen_topo_db")["config"]
    stage_cfg = stage["config"]

    # Fetch adjacency matrix and neuron infos according to configured locations/files
    adj_matrix, neuron_info, tribes = read_input(stage["inputs"])
    assert adj_matrix.shape[0] == len(neuron_info), "Neuron info and adjacency matrix have incompatible sizes!"

    param_dict = get_parameter_db_for_samples(tribes,
                                              neuron_info, topo_db_cfg, stage_cfg,
                                              adj_matrix)
    # Write output to where it's meant to go
    write_output(param_dict, stage["outputs"]["struc_parameters_volumetric"])


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
