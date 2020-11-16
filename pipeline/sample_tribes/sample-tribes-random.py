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

from scipy import spatial
from toposample import config


def read_input(input_config):
    db = pandas.read_pickle(input_config["database"])
    return db


def pick_random_where_column_has_certain_value(db, column, value, N):
    valid = db[column] == value  # numpy.bytes_(value)
    valid_index = db.index[valid]
    return numpy.random.choice(valid_index, numpy.minimum(N, len(valid_index)), replace=False)


def make_sample(db, specifications):
    assert "subsampling" not in specifications, "Subsampling not supported for random samples!"
    N = specifications["number"]
    spec_val = specifications["value"]
    random_chiefs = pick_random_where_column_has_certain_value(db, spec_val["column"], spec_val["value"], N)

    out_dict = {specifications["name"]: {}}
    for i, chief in enumerate(random_chiefs):
        gids = db["tribe"].loc[chief].astype(int)
        out_dict[specifications["name"]][str(i)] = {
            "gids": gids.tolist(),
            "chief": int(chief)
        }
    return out_dict


def make_all_samples(db, full_specification):
    spec_lbl = full_specification["Specifier_label"]
    numpy.random.seed(full_specification.get("seed", 9001))
    out_dict = dict([(spec_lbl, {})])
    for spec in full_specification["Specifiers"]:
        out_dict[spec_lbl].update(make_sample(db, spec))
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
    tribes = make_all_samples(db, stage["config"]["Random"])
    write_output(tribes, stage["outputs"])


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
