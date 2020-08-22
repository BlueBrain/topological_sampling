import json
import os


# noinspection SpellCheckingInspection
class Config(object):

    def __init__(self, fn):
        self._self_path = os.path.split(os.path.abspath(fn))[0]
        with open(fn, 'r') as fid:
            self._raw = json.load(fid)
        self._structure = self._raw["structure"]
        self._cfg = {}
        self.parse()

    def parse(self):
        for k, v in self._raw['paths'].items():
            full_path = os.path.abspath(os.path.join(self._self_path, v["dir"]))
            self._cfg[k] = {'_dir': full_path}
            for lbl, fn in v.get('files', {}).items():
                self._cfg[k][lbl] = os.path.join(full_path, fn)

    def stage(self, stage_name):
        if stage_name not in self._structure:
            raise Exception()
        struc = self._structure[stage_name]
        inputs = dict([(k, self._cfg[v][k]) for k, v in struc['inputs'].items()])
        outputs = dict([(k, self._cfg[v][k]) for k, v in struc['outputs'].items()])
        config_fn = self._cfg["config"][stage_name]
        if os.path.splitext(config_fn)[1] == ".json":
            with open(config_fn, "r") as fid:
                config = json.load(fid)
        else:
            config = None
        other = os.path.join(self._cfg["other"]["_dir"], stage_name)
        return {
            'inputs': inputs,
            'outputs': outputs,
            'config_fn': config_fn,
            'config': config,
            'other': other
        }


