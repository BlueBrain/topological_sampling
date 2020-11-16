"""
toposampling - Topology-assisted sampling and analysis of activity data
Copyright (C) 2020 Blue Brain Project / EPFL & University of Aberdeen

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os

import numpy
import pandas
import json

from toposample import config
from toposample.db import get_column_from_database


def read_input(input_config):
    db = pandas.read_pickle(input_config["database"])
    return db


# noinspection PyPep8Naming
def pick_champs_from_column(db, column, N, index=None, function=None):
    decider = get_column_from_database(db, column, index=index, function=function)
    picked = numpy.argsort(decider)[-N:]
    return db.index[picked]


def random_subsample(base_samples, ss_specs, specifier):
    m = ss_specs["number_samples"]
    n = ss_specs["number_subsamples"]
    chosen_samples = numpy.random.choice(list(base_samples.keys()),
                                         numpy.minimum(m, len(base_samples)),
                                         replace=False)
    out_dict = {}
    for lvl in ss_specs["levels"]:
        ss_dict = out_dict.setdefault(specifier + '@' + str(lvl), {})
        idxx = 0
        for _ in range(n):
            for smpl in chosen_samples:
                gids = base_samples[smpl]["gids"]
                gids = numpy.random.choice(gids, int(len(gids) * lvl / 100), replace=False)
                ss_dict[str(idxx)] = {"gids": gids.tolist(),
                                      "parent": specifier + "/" + str(smpl),
                                      "level": lvl
                                      }
                idxx += 1
    return out_dict


# noinspection PyPep8Naming
def make_sample(db, specifications):
    N = specifications["number"]
    spec_val = specifications["value"]
    chiefs = pick_champs_from_column(db, spec_val["column"], N, index=spec_val.get("index", None),
                                     function=spec_val.get("function", None))

    out_dict = {specifications["name"]: {}}
    for i, chief in enumerate(chiefs):
        gids = db["tribe"].loc[chief].astype(int)
        out_dict[specifications["name"]][i] = {
            "gids": gids.tolist(),
            "chief": int(chief)
        }
    return out_dict


def filter_by_minimum_tribe_size(db, min_size):
    tribal_size = numpy.array(list(map(len, db["tribe"])))
    return db.iloc[tribal_size > min_size]


def make_all_samples(db, full_specification):
    db = filter_by_minimum_tribe_size(db, full_specification.get("Minimum size", 0))
    spec_lbl = full_specification["Specifier_label"]
    out_dict = dict([(spec_lbl, {})])
    for spec in full_specification["Specifiers"]:
        out_dict[spec_lbl].update(make_sample(db, spec))
        if "subsampling" in spec:
            ss_dict = random_subsample(out_dict[spec_lbl][spec["name"]], spec["subsampling"], spec["name"])
            out_dict.setdefault("subsampled", {}).update(ss_dict)
    return out_dict


def write_output(data, output_config):
    if os.path.exists(output_config["tribes"]):
        with open(output_config["tribes"], "r") as fid:
            existing_config = json.load(fid)
            existing_config.update(data)
        with open(output_config["tribes"], "w") as fid:
            json.dump(existing_config, fid, indent=2)
    else:
        with open(output_config["tribes"], "w") as fid:
            json.dump(data, fid, indent=2)


def main(path_to_config):
    # Read the meta-config file
    cfg = config.Config(path_to_config)
    # Get configuration related to the current pipeline stage
    stage = cfg.stage("sample_tribes")
    db = read_input(stage["inputs"])
    tribes = make_all_samples(db, stage["config"]["Champions"])
    write_output(tribes, stage["outputs"])


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
