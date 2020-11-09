"""
NEUTRINO - NEUral TRIbe and Network Observer
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


def find_subtribes(db, base_samples, st_specs, specifier):
    db_tribes = db["tribe"]
    num_samples = st_specs["number_samples"]
    num_tribes = st_specs["number_tribes"]
    chosen_samples = sorted(base_samples.keys(), key=int)[:num_samples]
    
    out_dict = {}
    for smpl in chosen_samples:
        st_dict = out_dict.setdefault(specifier + '@' + str(smpl), {})
        gids = base_samples[smpl]["gids"]
        
        # Find subtribes in volumetric sample
        db_subtribes = db_tribes.loc[gids]
        for gid in gids:
            db_subtribes.loc[gid] = numpy.intersect1d(db_subtribes.loc[gid], gids, assume_unique=True).tolist()
        
        # Select N largest tribes
        db_subtribes = db_subtribes.to_frame()
        db_subtribes['size'] = 0
        for gid in gids:
            db_subtribes.loc[gid, 'size'] = len(db_subtribes.loc[gid, 'tribe'])
        
        db_subtribes.sort_values('size', ascending=False, inplace=True)
        subtribes_gids = db_subtribes['tribe'].iloc[:num_tribes].to_list()
        subtribes_chief = db_subtribes.index[:num_tribes].to_list()
        
        # Add to dict
        for idxx in range(num_tribes):
            st_dict[str(idxx)] = {"gids": subtribes_gids[idxx],
                                  "chief": subtribes_chief[idxx]
                                 }
    return out_dict


def add_random_subtribes(base_samples, st_dict, st_specs, rng):
    out_dict = {}
    if "number_random" in st_specs and st_specs["number_random"] > 0: # Add random control samples for all subtribes
        num_rnd = st_specs["number_random"]
        
        for subtr in st_dict.keys():
            spec, smpl = subtr.split("@")
            gids = base_samples[spec][smpl]["gids"]
            
            for n_rnd in range(num_rnd):
                rnd_str = "" if num_rnd == 1 else f"-{n_rnd}" # Add string extension if more than one random samples are generated
                rnd_dict = out_dict.setdefault(f"{spec}@{smpl}{rnd_str}", {})
                for idx in st_dict[subtr].keys():
                    subtr_size = len(st_dict[subtr][idx]["gids"])
                    rnd_sample = rng.choice(gids, subtr_size, replace=False)
                    rnd_dict[idx] = {"gids": rnd_sample.tolist(),
                                     "chief": None
                                    }
    return out_dict


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
        out_dict[specifications["name"]][str(i)] = {
            "gids": gids.tolist(),
            "center_offset": offset.tolist()
        }
    return out_dict


def make_all_samples(db, full_specification):
    offset_amplitude = full_specification["Arguments"]["offset_amplitudes"]
    spec_lbl = full_specification["Specifier_label"]
    numpy.random.seed(full_specification.get("seed", 1337))
    rng_subtr = numpy.random.default_rng(seed=full_specification.get("seed_subtribes", 2559)) # Separate RNG so that random control sampling of subtribes (if selected) not interfering with volumetric sampling
    out_dict = dict([(spec_lbl, {})])
    for spec in full_specification["Specifiers"]:
        out_dict[spec_lbl].update(make_sample(db, spec, offset_amplitude))
        if "subtribes" in spec:
            st_dict = find_subtribes(db, out_dict[spec_lbl][spec["name"]], spec["subtribes"], spec["name"])          
            out_dict.setdefault("subtribes", {}).update(st_dict)
            
            rnd_dict = add_random_subtribes(out_dict[spec_lbl], st_dict, spec["subtribes"], rng_subtr) # Generates random samples with exact same sizes as subtribes
            out_dict.setdefault("subtribes_random", {}).update(rnd_dict)
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
