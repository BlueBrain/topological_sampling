"""
toposampling - Topology-assisted sampling and analysis of activity data
Copyright (C) 2020 Blue Brain Project / EPFL & University of Aberdeen

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import json
from os import path
import shutil

from toposample import Config
from toposample.data.read_data_json import H5File

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
    fn_tmp = fn_out + ".TMP" # Write to .TMP file first and rename later, since json file may contain references to existing output file
    if path.isfile(fn_out):
        fn_bak = fn_out + ".BAK"
        print("WARNING: File {0} already exists. Moving to {1}.".format(fn_out, path.split(fn_bak)[1]))
        assert not path.isfile(fn_bak), "File {0} already exists. To regenerate it, delete it first!".format(fn_bak) # Don't overwrite an existing backup
        shutil.copy(fn_out, fn_bak) # Create backup, since file will be overwritten at the end
    with open(json_fn, "r") as fid:
        j_in = json.load(fid)
    
    print("Copying linked content in {0} to {1}...".format(json_fn, fn_out))
    with h5py.File(fn_tmp, "w") as h5_out:
        for sampling in j_in.keys():
            print("\tfor sampling = {0}".format(sampling))
            for specifier in j_in[sampling].keys():
                for index in j_in[sampling][specifier].keys():
                    info_dict = j_in[sampling][specifier][index]
                    if "data_fn" in info_dict:
                        grp_out_name = "{0}/{1}/{2}".format(sampling, specifier, index)
                        grp_out = h5_out.require_group(grp_out_name)
                        data_fn = info_dict["data_fn"]
                        if not path.isabs(data_fn): # In case of relative paths, interprete them relative to json file
                            data_fn = path.join(path.split(json_fn)[0], data_fn)
                        with H5File(data_fn) as h5_in: # Use H5File supporting path continuation to a group within the file
                            for k in h5_in.keys():
                                h5_in.copy(k, grp_out)
                        info_dict["data_fn"] = path.join(fn_out, grp_out_name)
    shutil.move(fn_tmp, fn_out) # Renaming output file, potentially overwriting existing file
    return j_in


def main(path_to_cfg, stage_name, type_of_classification=None, path_to_output=None):
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
        {0} [-o out_fn] path/to/common-config.json stage_name [type-of-classification]
        Merges the various data payload files referenced in the table-of-contents file of a pipeline stage result into
        one single h5 file. The merged h5 file will be placed at the root of the pipeline stages "other" directory,
        by default that is working_dir/data/other/$pipeline-stage-name.
        
        The table of contents file will be updated to point at groups within the merged file, unless overridden by
        using the -o option, in which case the updated file will be placed at out_fn.
        
        Input [type-of-classification] is required if stage_name == "classifier" and must be either "components"
        or "features".
        """.format(__name__))
        sys.exit(2)
    main(*args, path_to_output=opts.get("-o", None))
