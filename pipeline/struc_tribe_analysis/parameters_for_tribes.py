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
import pandas
import json

from toposample import config
from toposample import TopoData
from toposample.db import get_entry_from_row, get_column_from_database
from toposample.indexing import GidConverter


def read_input(input_config):
    db = pandas.read_pickle(input_config["database"])
    tribes = TopoData(input_config["tribes"])
    tribal_chiefs = tribes["chief"]
    tribal_gids = tribes["gids"]
    # offsets = tribes["center_offset"]
    return db, tribal_chiefs, tribal_gids


def lookup_parameter_from_db_by_chief(db, list_of_parameters, chief):
    row = db.loc[chief]
    out_dict = {}
    for param_spec in list_of_parameters:
        v = get_entry_from_row(row, param_spec["value"]["column"],
                               index=param_spec["value"].get("index", None),
                               function=param_spec["value"].get("function", None))
        out_dict[param_spec["name"]] = float(v)
    return out_dict


# noinspection PyPep8Naming
def tribal_spectrum(db_metrics, gids):
    L = []
    rel_L = []
    for tribe in db_metrics['tribe']:
        L.append(len(numpy.intersect1d(gids, tribe)))
        rel_L.append(2 * L[-1] / (len(gids) + len(tribe)))
    return L, rel_L


def top_n_weighted_average(w, v, number_sampled=10):
    w, v = numpy.array(w), numpy.array(v)
    idxx = numpy.argsort(w)[-int(number_sampled):]
    v = v[idxx]
    w = w[idxx]
    return numpy.nansum(w * v) / numpy.nansum(w)


def get_relevant_columns_from_db(db, list_of_parameters):
    out_dict = {}
    print("Looking up relevant db entries...")
    for param_spec in list_of_parameters:
        print("...{0}".format(param_spec["name"]))
        v = get_column_from_database(db, param_spec["value"]["column"],
                                     index=param_spec["value"].get("index", None),
                                     function=param_spec["value"].get("function", None))
        out_dict[param_spec["name"]] = numpy.array(v)
    return out_dict


def predict_parameter_from_db_by_gids(db, dict_of_columns, list_of_parameters, gids):
    _, relative_overlap = tribal_spectrum(db, gids)
    conv = GidConverter(db)
    out_dict = {}
    for param_spec in list_of_parameters:
        v = dict_of_columns[param_spec["name"]]
        use_kwargs = param_spec["prediction_strategy"].get("kwargs", {})
        if param_spec["prediction_strategy"]["strategy"] == "weighted_mean_by_overlap":
            out_dict[param_spec["name"]] = top_n_weighted_average(relative_overlap, v, **use_kwargs)
        elif param_spec["prediction_strategy"]["strategy"] == "mean_of_members":
            out_dict[param_spec["name"]] = numpy.nanmean(v[conv.indices(gids)])
    return out_dict


def make_lookup_functions(db, list_of_parameters):
    dict_of_columns = get_relevant_columns_from_db(db, list_of_parameters)

    def lookup_if_non_volumetric(sampling_strats, tribe_spec):  # to be used with tribal chiefs
        for smpl, trb in zip(sampling_strats, tribe_spec):
            if smpl != 'Radius':
                yield lookup_parameter_from_db_by_chief(db, list_of_parameters, trb), {'sampling': smpl}

    def predict_if_volumetric(sampling_strats, tribe_spec):  # to be used with tribal gids
        for smpl, trb in zip(sampling_strats, tribe_spec):
            if smpl == 'Radius':
                print("Predicting parameter value for volumetric sample of {0} gids".format(len(trb)))
                yield predict_parameter_from_db_by_gids(db, dict_of_columns, list_of_parameters, trb),\
                      {'sampling': smpl}
    return lookup_if_non_volumetric, predict_if_volumetric


def lookup_parameters(db, tribal_chiefs, tribal_gids, stage_cfg):
    func_lookup, func_predict = make_lookup_functions(db, stage_cfg["Parameters"])
    out_collection = tribal_chiefs.transform(["sampling"], func=func_lookup, xy=True)
    # The next line will take some minutes
    out_collection.merge(tribal_gids.transform(["sampling"], func=func_predict, xy=True))
    return out_collection


def write_output(data, output_config):
    fn_out = output_config["struc_parameters"]
    final_dict = TopoData.condition_collection_to_dict(data)

    with open(fn_out, 'w') as fid:
        json.dump(final_dict, fid, indent=2)


def main(path_to_config):
    # Read the meta-config file
    cfg = config.Config(path_to_config)
    # Get configuration related to the current pipeline stage
    stage = cfg.stage("struc_tribe_analysis")
    db, tribal_chiefs, tribal_gids = read_input(stage["inputs"])
    tribal_values = lookup_parameters(db, tribal_chiefs, tribal_gids, stage["config"])
    write_output(tribal_values, stage["outputs"])


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
