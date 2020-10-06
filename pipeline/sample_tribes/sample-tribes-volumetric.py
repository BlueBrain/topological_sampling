"""
Topological sampling
Copyright (C) 2020 Blue Brain Project / EPFL

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
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


def pick_random_neurons_volumetric(db, radius, offset, M):
    locations = db[['x', 'y', 'z']]
    center = locations.values.mean(axis=0) + offset
    D = spatial.distance.cdist(numpy.array([center]), locations)[0]
    gids = locations.index.values[D <= radius]
    return numpy.random.choice(gids, numpy.minimum(M, len(gids)), replace=False)


def make_sample(db, specifications, offset_amplitudes):
    assert "subsampling" not in specifications, "Subsampling not supported for volumetric samples!"
    N = specifications["number"]
    M = specifications["neuron_count"]
    radius = specifications["value"]
    offset_amplitudes = numpy.array(offset_amplitudes)
    offsets = numpy.random.rand(N, 3) * offset_amplitudes - offset_amplitudes / 2

    out_dict = {specifications["name"]: {}}
    for i, offset in enumerate(offsets):
        gids = pick_random_neurons_volumetric(db, radius, offset, M)
        out_dict[specifications["name"]][i] = {
            "gids": gids.tolist(),
            "center_offset": offset.tolist()
        }
    return out_dict


def make_all_samples(db, full_specification):
    offset_amplitude = full_specification["Arguments"]["offset_amplitudes"]
    spec_lbl = full_specification["Specifier_label"]
    numpy.random.seed(full_specification.get("seed", 1337))
    out_dict = dict([(spec_lbl, {})])
    for spec in full_specification["Specifiers"]:
        out_dict[spec_lbl].update(make_sample(db, spec, offset_amplitude))
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
    tribes = make_all_samples(db, stage["config"]["Volumetric"])
    write_output(tribes, stage["outputs"])


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
