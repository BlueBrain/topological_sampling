import os

import numpy
import pandas
import json
import progressbar

from toposample import config
from toposample import TopoData


def read_input(input_config):
    db = pandas.read_pickle(input_config["database"])
    tribes = TopoData(input_config["tribes"])
    tribal_chiefs = tribes["chief"]
    tribal_gids = tribes["gids"]
    return db, tribal_chiefs, tribal_gids


def lookup_parameter_from_db_by_chief(db, list_of_parameters, chief):
    row = db.loc[chief]
    out_dict = {}
    for param_spec in list_of_parameters:
        v = row[param_spec["value"]["column"]]
        idx = param_spec["value"].get("index", None)
        if idx is not None:
            v = v[idx]
        out_dict[param_spec["name"]] = v
    return out_dict


# noinspection PyPep8Naming
def tribal_spectrum(db_metrics, gids):
    L = []
    rel_L = []
    for tribe in db_metrics['neighbours']:
        L.append(len(numpy.intersect1d(gids, tribe)))
        rel_L.append(2 * L[-1] / (len(gids) + len(tribe)))
    return L, rel_L


def top_n_weighted_average(w, v, n=10):
    w, v = numpy.array(w), numpy.array(v)
    idxx = numpy.argsort(w)[-n:]
    v = v[idxx]
    w = w[idxx]

    return numpy.nansum(w * v) / numpy.nansum(w)


def predict_parameter_from_db_by_gids(db, list_of_parameters, gids):
    _, relative_overlap = tribal_spectrum(db, gids)
    out_dict = {}
    for param_spec in list_of_parameters:
        v = db[param_spec["value"]["column"]].values
        idx = param_spec["value"].get("index", None)
        if idx is not None:
            v = [_v[idx] for _v in v]
        N = param_spec["prediction_strategy"]["number_sampled"]
        out_dict[param_spec["name"]] = top_n_weighted_average(relative_overlap, v, n=N)
    return out_dict


def make_lookup_functions(db, list_of_parameters):
    def lookup_if_non_volumetric(sampling_strats, tribe_spec): # to be used with tribal chiefs
        for smpl, trb in zip(sampling_strats, tribe_spec):
            if smpl != 'Volumetric':
                yield lookup_parameter_from_db_by_chief(db, list_of_parameters, trb), {'sampling': smpl}

    def predict_if_volumetric(sampling_strats, tribe_spec): # to be used with tribal gids
        for smpl, trb in zip(sampling_strats, tribe_spec):
            if smpl == 'Volumetric':
                yield predict_parameter_from_db_by_gids(db, list_of_parameters, trb), {'sampling': smpl}
    return lookup_if_non_volumetric, predict_if_volumetric


def lookup_parameters(db, tribal_chiefs, tribal_gids, stage_cfg):
    func_lookup, func_predict = make_lookup_functions(db, stage_cfg["Parameters"])
    out_collection = tribal_chiefs.transform(["sampling"], func=func_lookup, xy=True)
    # The next line will take some minutes
    out_collection.merge(tribal_gids.transform(["sampling"], func=func_predict, xy=True))
    return out_collection


def write_output(data, output_config):
    fn_out = output_config["struc_parameters"]
    final_dict = {}
    for sampling in data.labels_of("sampling"):
        smpl_lvl = final_dict.setdefault(sampling, {})
        for specifier in data.labels_of("specifier"):
            spec_lvl = smpl_lvl.setdefault(specifier, {})
            for index in data.labels_of("index"):
                spec_lvl[index] = data.get2(sampling=sampling, specifier=specifier, index=index)

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
