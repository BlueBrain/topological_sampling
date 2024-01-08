#!/usr/bin/env python
"""
toposampling - Topology-assisted sampling and analysis of activity data
Copyright (C) 2020 Blue Brain Project / EPFL & University of Aberdeen

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy
import os
import json
import h5py

from toposample import config
from toposample import TopoData
from toposample.data.data_structures import ConditionCollection


def read_input(input_config):
    tribes = TopoData(input_config["tribes"])
    tribal_chiefs = tribes["chief"]
    tribal_gids = tribes["gids"]
    if "center_offset" in tribes.data:
        tribal_chiefs.merge(tribes["center_offset"])
    if "parent" in tribes.data:
        tribal_chiefs.merge(tribes["parent"])
    spikes = numpy.load(input_config["raw_spikes"])
    stims = numpy.load(input_config["stimuli"])
    return spikes, stims, tribal_chiefs, tribal_gids


def merge_tribes(tribal_chiefs, tribal_gids, merge_specs, merged_label="merged"):
    tgt_cond = merge_specs["target_condition"]
    tgt_vals = merge_specs["target_values"]
    pool_cond = merge_specs["pool_condition"]

    def _tf_and_merge(lst_vals, lst_conds, func):
        tf_out = ConditionCollection([])
        for _v, _c in zip(lst_vals, lst_conds):
            if _v[0] in tgt_vals:
                _c = _c.pool([pool_cond], func=func)
                if pool_cond not in _c.conditions():
                    _c.add_label(pool_cond, merged_label)
                if tgt_cond not in _c.conditions():
                    _c.add_label(tgt_cond, _v[0])
                tf_out.merge(_c)
            else:
                tf_out.merge(_c)
        return tf_out

    vals, conds = tribal_gids.split(tgt_cond)
    tribal_gids = _tf_and_merge(vals, conds, lambda _x: numpy.unique(numpy.hstack(_x)))

    vals, conds = tribal_chiefs.split(tgt_cond)
    tribal_chiefs = _tf_and_merge(vals, conds, lambda _x: merged_label)
    return tribal_chiefs, tribal_gids


def spikes_to_y_vec(spikes, gids, t_bin_width, t_stim_start):
    t_max = numpy.ceil(numpy.max(spikes[:, 0]) / t_bin_width) * t_bin_width
    spikes = spikes[numpy.in1d(spikes[:, 1], gids)]
    gid_bins = numpy.hstack([sorted(gids), numpy.max(gids) + 1])
    t_bins = numpy.arange(t_stim_start, t_max + t_bin_width, t_bin_width)
    out = numpy.histogram2d(spikes[:, 1], spikes[:, 0], bins=(gid_bins, t_bins))[0]
    return out


# noinspection PyPep8Naming
def factor_analysis(y_mat, num_components):
    from sklearn.decomposition import FactorAnalysis
    F = FactorAnalysis(num_components)
    transformed = F.fit_transform(y_mat.transpose())  # shape: time x components
    components = F.components_
    mn = F.mean_
    noise_variance = F.noise_variance_
    return transformed, components, mn, noise_variance


def split_transformed_into_t_wins(transformed, stim_train):
    u_stims = numpy.unique(stim_train)
    tf_splt = numpy.split(transformed, len(stim_train), axis=0)  # [trials] x time x components
    per_stim_splt = [numpy.dstack([_splt for _splt, s in zip(tf_splt, stim_train)
                                   if s == i]) for i in numpy.unique(u_stims)]  # [stimulus] x time x component x trial
    # per_stim_splt = [_splt.transpose([1, 0, 2]) for _splt in per_stim_splt]  # [stimulus] x time x component x trial
    return per_stim_splt


def output_filename(out_root, conds):
    return os.path.join(out_root, conds.get("sampling", "UNSPECIFIED"),
                        conds.get("specifier", "UNSPECIFIED"),
                        conds.get("index", "UNSPECIFIED"), "results.h5")


def write_results_file(transformed, tf_split, components, mn, noise_variance, chief_spec, out_fn):
    assert not os.path.exists(out_fn)
    if not os.path.exists(os.path.split(out_fn)[0]):
        os.makedirs(os.path.split(out_fn)[0])
    with h5py.File(out_fn, 'w') as h5:
        h5.create_dataset("transformed", data=transformed)
        h5.create_dataset("components", data=components)
        h5.create_dataset("mean", data=mn)
        h5.create_dataset("noise_variance", data=noise_variance)
        h5.attrs["idv_label"] = chief_spec
        grp = h5.require_group("per_stimulus")
        for i, res in enumerate(tf_split):
            grp.create_dataset("stim{0}".format(i), data=res)
    return out_fn


def transform_all(spikes, stims, tribal_chiefs, tribal_gids, stage_config, out_root):
    result_lookup = {}
    for res in tribal_gids.contents:
        out_fn = output_filename(out_root, res.cond)
        if os.path.exists(out_fn):
            print("{0} exists. Skipping...".format(out_fn))
            continue
        print("Transforming for {0}".format(res.cond))
        gids = res.res
        y_vec = spikes_to_y_vec(spikes, gids, stage_config["t_bin_width"], stage_config["t_stim_start"])
        transformed, components, mn, noise_variance = factor_analysis(y_vec, stage_config["n_components"])
        tf_split = split_transformed_into_t_wins(transformed, stims)
        chief = tribal_chiefs.get2(**res.cond)
        out_fn = write_results_file(transformed, tf_split, components, mn,
                                    noise_variance, chief, out_fn)
        spec_lvl = result_lookup.setdefault(res.cond["sampling"], {}).setdefault(res.cond["specifier"], {})
        spec_lvl[res.cond["index"]] = {"data_fn": out_fn, "idv_label": chief}
    return result_lookup


def write_output(data, output_config):
    fn_out = output_config["components"]
    if os.path.exists(fn_out):
        with open(fn_out, "r") as fid:
            existing = json.load(fid)
        for k, v in data.items():
            existing.setdefault(k, {}).update(v)
        with open(fn_out, "w") as fid:
            json.dump(existing, fid, indent=2)
    else:
        with open(fn_out, 'w') as fid:
            json.dump(data, fid, indent=2)


def main(path_to_config, **kwargs):
    # Read the meta-config file
    cfg = config.Config(path_to_config)
    # Get configuration related to the current pipeline stage
    stage = cfg.stage("manifold_analysis")
    spikes, stims, tribal_chiefs, tribal_gids = read_input(stage["inputs"])
    if len(kwargs) > 0:
        tribal_gids = tribal_gids.filter(**kwargs)
        tribal_chiefs = tribal_chiefs.filter(**kwargs)
    if "merge_samples" in stage["config"]:
        tribal_chiefs, tribal_gids = merge_tribes(tribal_chiefs, tribal_gids, stage["config"]["merge_samples"])
    res_lookup = transform_all(spikes, stims, tribal_chiefs, tribal_gids, stage["config"], stage["other"])
    write_output(res_lookup, stage["outputs"])


def parse_filter_arguments(*args):
    fltr_dict = {}
    for arg in args:
        if "=" in arg:
            splt = arg.split("=")
            fltr_dict[splt[0]] = splt[1]
    return fltr_dict


if __name__ == "__main__":
    import sys
    main(sys.argv[1], **parse_filter_arguments(*sys.argv[2:]))
