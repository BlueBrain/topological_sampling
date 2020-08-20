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
    pandas.to_pickle(DB, stage["outputs"]["database"])
    if delete_inputs:
        for fn in files:
            os.remove(fn)


if __name__ == "__main__":
    import sys
    import getopt

    opts, args = getopt.getopt(sys.argv[1:], "D")
    opts = dict(opts)

    main(args[0], delete_inputs=("-D" in opts))
