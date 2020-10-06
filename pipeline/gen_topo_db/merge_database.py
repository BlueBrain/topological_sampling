"""
Topological sampling
Copyright (C) 2020 Blue Brain Project / EPFL

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pandas
import os


from toposample import Config


def find_files(stage):
    file_list = []
    other_root = stage["other"]
    base_fn = os.path.split(stage["outputs"]["database"])[1]
    for param in ["tribe", "neuron_info"] + stage['config']['parameters']:
        expected_fn = os.path.join(other_root, base_fn + "." + param.lower().replace(" ", "_"))
        if os.path.isfile(expected_fn):
            file_list.append(expected_fn)
        else:
            print("Warning: expected column {0}, but file {1} was not found".format(param, expected_fn))
    return file_list


def main(cfg_fn, delete_inputs=False):
    cfg = Config(cfg_fn)
    stage = cfg.stage("gen_topo_db")
    files = find_files(stage)
    if len(files) == 0:
        return
    DB_raw = [pandas.read_pickle(_fn) for _fn in files]
    DB = pandas.concat(DB_raw, axis=1)
    fn_out = stage["outputs"]["database"]
    if not os.path.isdir(os.path.split(fn_out)[0]):
        os.makedirs(os.path.split(fn_out)[0])
    pandas.to_pickle(DB, fn_out)
    if delete_inputs:
        for fn in files:
            os.remove(fn)


if __name__ == "__main__":
    import sys
    import getopt

    opts, args = getopt.getopt(sys.argv[1:], "D")
    opts = dict(opts)

    main(args[0], delete_inputs=("-D" in opts))
