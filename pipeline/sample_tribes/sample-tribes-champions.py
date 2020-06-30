import os

import numpy
import pandas
import json

from scipy import spatial
from toposample import config


def read_input(input_config):
    db = pandas.read_pickle(input_config["database"])
    return db


def pick_champs_from_column(db, column, N, index=None):
    decider = db[column]
    if index is not None:
        print(column, index)
        decider = [_x[index] if len(_x) > index else 0 for _x in decider]
    picked = numpy.argsort(decider)[-N:]
    return db.index[picked]


def make_sample(db, specifications):
    N = specifications["number"]
    spec_val = specifications["value"]
    chiefs = pick_champs_from_column(db, spec_val["column"], N, index=spec_val.get("index", None))

    out_dict = {specifications["name"]: {}}
    for i, chief in enumerate(chiefs):
        gids = db["neighbours"].loc[chief].astype(int)
        out_dict[specifications["name"]][i] = {
            "gids": gids.tolist(),
            "chief": int(chief)
        }
    return out_dict


def make_all_samples(db, full_specification):
    spec_lbl = full_specification["Specifier_label"]
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
    tribes = make_all_samples(db, stage["config"]["Champions"])
    write_output(tribes, stage["outputs"])


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
