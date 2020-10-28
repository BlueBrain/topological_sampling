"""
NEUTRINO - NEUral TRIbe and Network Observer
Copyright (C) 2020 Blue Brain Project / EPFL

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import json
from os import path

from toposample import Config

valid_stages = ["manifold_analysis", "classifier", "topological_featurization"]

outputs_to_use = {
    "manifold_analysis": "components",
    "classifier": "classifier_{0}_results",
    "topological_featurization": "features"
}

expected_filenames = {
    "manifold_analysis": "all_results.h5",
    "classifier": "all_results_{0}.h5",
    "topological_featurization": "all_results.h5"
}


def compress_results(json_fn, other_path, stage_name, type_of_classification):
    import h5py
    fn_out = path.join(other_path, expected_filenames[stage_name].format(type_of_classification))
    assert not path.isfile(fn_out), "File {0} already exists. To regenerate it, delete it first!".format(fn_out)
    with open(json_fn, "r") as fid:
        j_in = json.load(fid)

    print("Copying linked content in {0} to {1}...".format(json_fn, fn_out))
    with h5py.File(fn_out, "w") as h5_out:
        for sampling in j_in.keys():
            print("\tfor sampling = {0}".format(sampling))
            for specifier in j_in[sampling].keys():
                for index in j_in[sampling][specifier].keys():
                    info_dict = j_in[sampling][specifier][index]
                    if "data_fn" in info_dict:
                        grp_out_name = "{0}/{1}/{2}".format(sampling, specifier, index)
                        grp_out = h5_out.require_group(grp_out_name)
                        h5_in = h5py.File(info_dict["data_fn"], "r")
                        for k in h5_in.keys():
                            h5_in.copy(k, grp_out)
                        info_dict["data_fn"] = path.join(fn_out, grp_out_name)
    return j_in


def main(path_to_cfg, stage_name, type_of_classification=None, path_to_output=None):
    import shutil
    assert stage_name in valid_stages
    if stage_name == "classifier":
        assert type_of_classification is not None, "To repair classifier results, specify which classifier!"
    cfg = Config(path_to_cfg)
    stage = cfg.stage(stage_name)
    path_to_other = stage["other"]
    path_to_json = stage["outputs"][outputs_to_use[stage_name].format(type_of_classification)]

    modified_json = compress_results(path_to_json, path_to_other, stage_name, type_of_classification)
    if path_to_output is None:
        shutil.copy(path_to_json, path_to_json + ".BAK")
        path_to_output = path_to_json
    with open(path_to_output, "w") as fid:
        json.dump(modified_json, fid, indent=2)


if __name__ == "__main__":
    import sys
    import getopt

    opts, args = getopt.getopt(sys.argv[1:], "o:")
    opts = dict(opts)
    if len(args) < 2:
        print("""
        The table of contents file will be placed where the pipeline expects it, according to specifications in the 
        common config, unless overridden by using the -o option
        """.format())
        sys.exit(2)
    main(*args, path_to_output=opts.get("-o", None))
