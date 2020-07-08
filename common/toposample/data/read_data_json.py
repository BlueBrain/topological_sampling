import json

from .data_structures import ConditionCollection, ResultsWithConditions


class TopoData(object):
    """
    TOPODATA:
    A class holding results of our spiking data analysis that were formatted into a .json file.

    Instantiate with the path to the results file:
        res = TopoData("working_dir/data/analyzed_data/extracted_components.json")

    A results .json file can contain any number of results for the various sampling conditions. Each such result
    has a name. The names can be queried using the .keys() function and accessed using dict-type indexing:
        res.keys()
            ["data_fn", "idv_label"]
        data = res["data_fn"]

    This result then holds the requested data for all sampling conditions. It can be accessed using the .get function
    specifying the "sampling", "specifier" and "index" conditions:

        data_point = data.get(sampling="random", specifier="Euler characteristic", index=0)
            ["working_dir/data/other/manifold_analysis/random/Euler characteristic/0/results.h5"]

    Specifications for the .get function do not have to be complete. In that case you get multiple results:

        data_points = data.get(sampling="random", specifier="Euler characteristic")
            ["working_dir/data/other/manifold_analysis/random/Euler characteristic/0/results.h5",
            ...,
            "working_dir/data/other/manifold_analysis/random/Euler characteristic/20/results.h5"]

    In some cases (such as the examples above), the result is merely a path pointing to the file holding the actual
    results. In that case you can use the 'follow_link_functions' argument when instantiating the TopoData object
    to specify the function to be used to read the actual data from the file.
    The argument for 'follow_link_function' is a dict where the key(s) are the names of the results you want resolved
    in this way and the values(s) are tuples where the first item is a function to be called with the path to the file
    as the only argument that returns the data contained in the file and the second item is a bool that specifies
    whether this data should be cached or read from the file on every call.

        from toposample import data

        functions_dict = {"data_fn": [data.read_h5_dataset("transformed"), False]} # returns the dset "transformed"
        res = data.TopoData(fn, follow_link_functions=functions_dict)
        data = res["data_fn"]
        data.get(sampling="random", specifier="Euler characteristic", index=0)
            [array([[ 1.94484386, -1.78500038, -1.92773744, ...,  0.40306363, 3.86585029, -0.99272454],
                    ...,
                   [-0.49383982, -0.02171111, -0.01068425, ...,  0.09241486, -0.09268441,  0.06436797]])]
    """
    def __init__(self, fn, follow_link_functions={}):
        with open(fn, "r") as fid:
            self._raw = json.load(fid)
        self.data = self.parse_raw(follow_link_functions)

    def parse_raw(self, follow_link_functions):
        out_dict = {}
        for sampling, samp_lvl in self._raw.items():
            for specifier, spec_lvl in samp_lvl.items():
                for index, idx_lvl in spec_lvl.items():
                    for k, v in idx_lvl.items():
                        out_dict.setdefault(k, []).append(ResultsWithConditions(v, *follow_link_functions.get(k, []),
                                                                                sampling=sampling,
                                                                                specifier=specifier,
                                                                                index=index)
                                                          )
        out_dict = dict([(k, ConditionCollection(v)) for k, v in out_dict.items()])
        return out_dict

    def __getitem__(self, spec_key):
        return self.data[spec_key]

    def keys(self):
        return self.data.keys()

    @staticmethod
    def condition_collection_to_dict(data):
        final_dict = {}
        for sampling in data.labels_of("sampling"):
            smpl_lvl = final_dict.setdefault(sampling, {})
            smpl_fltr = data.filter(sampling=sampling)
            for specifier in smpl_fltr.labels_of("specifier"):
                spec_lvl = smpl_lvl.setdefault(specifier, {})
                for index in data.labels_of("index"):
                    data_point = data.get2(sampling=sampling, specifier=specifier, index=index)
                    if data_point != []:
                        spec_lvl[index] = data_point
        return final_dict


def read_h5_dataset(dset_name):
    import h5py
    import numpy

    def read_func(fn):
        with h5py.File(fn, "r") as h5:
            return numpy.array(h5[dset_name])
    return read_func