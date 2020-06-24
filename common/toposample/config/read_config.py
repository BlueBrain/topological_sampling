import json
import os


class Config(object):
    structure = {
        "gen_topo_db": {
            "inputs": {
                "adjacency_matrix": "inputs",
                "neuron_info": "inputs"
            },
            "outputs": {
                "database": "analyzed"
            }
        },
        "sample_tribes": {
            "inputs": {
                "database": "analyzed"
            },
            "outputs": {
                "tribes": "analyzed"
            }
        },
        "struc_tribe_analysis": {
            "inputs": {
                "database": "analyzed",
                "tribes": "analyzed"
            },
            "outputs": {
                "struc_parameters": "analyzed"
            }
        }
    }

    def __init__(self, fn):
        self._self_path = os.path.split(os.path.abspath(fn))[0]
        with open(fn, 'r') as fid:
            self._raw = json.load(fid)

    def parse(self):
        self._cfg = {}
        for k, v in self._raw['paths'].items():
            full_path = os.path.abspath(os.path.join(self._self_path, v["dir"]))
            self._cfg[k] = {'_dir': full_path}
            for lbl, fn in v.get('files', {}).items():
                self._cfg[k][lbl] = os.path.join(full_path, fn)

    def stage(self, stage_name):
        if stage_name not in self.__class__.structure:
            raise Exception()
        struc = self.__class__.structure[stage_name]
        inputs = dict([(k, self._cfg[v][k]) for k, v in struc['inputs'].items()])
        outputs = dict([(k, self._cfg[v][k]) for k, v in struc['outputs'].items()])
        config = self._cfg["config"][stage_name]
        other = os.path.join(self._cfg["other"]["_dir"], stage_name)
        return {
            'inputs': inputs,
            'outputs': outputs,
            'config': config,
            'other': other
        }


