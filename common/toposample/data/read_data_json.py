import json

from .data_structures import ConditionCollection, ResultsWithConditions


class TopoData(object):

    def __init__(self, fn):
        with open(fn, "r") as fid:
            self._raw = json.load(fid)
        self.data = self.parse_raw()

    def parse_raw(self):
        out_dict = {}
        for sampling, samp_lvl in self._raw.items():
            for specifier, spec_lvl in samp_lvl.items():
                for index, idx_lvl in spec_lvl.items():
                    for k, v in idx_lvl.items():
                        out_dict.setdefault(k, []).append(ResultsWithConditions(v,
                                                                                sampling=sampling,
                                                                                specifier=specifier,
                                                                                index=index)
                                                          )
        out_dict = dict([(k, ConditionCollection(v)) for k, v in out_dict.items()])
        return out_dict

    def __getitem__(self, spec_key):
        return self.data[spec_key]

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
