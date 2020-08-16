import json
from os import path, listdir

from toposample import Config, TopoData
from toposample.data.data_structures import ConditionCollection


additional_fields = {
    "manifold_analysis": {
        "idv_label": ["parent", "center_offset", "chief"]
    },
    "classifier": {
        "idv_label": ["parent", "center_offset", "chief"]
    },
    "topological_featurization": {

    }
}

outputs_to_use = {
    "manifold_analysis": "components",
    "classifier": "classifier_{0}_results",
    "topological_featurization": "features"
}

expected_filenames = {
    "manifold_analysis": "results.h5",
    "classifier": "results_{0}.h5",
    "topological_featurization": "results.h5"
}


def get_relevant_stage(cfg, path_to_results):
    stage_name = path.split(path_to_results)[1]
    return cfg.stage(stage_name), stage_name


def assemble_additional_data(specs, tribes):
    out_dict = {}
    for k, lst_fields in specs.items():
        out_collection = ConditionCollection(tribes[lst_fields[0]].contents)
        for field in lst_fields[1:]:
            out_collection.merge(tribes[field])
        out_dict[k] = out_collection
    return out_dict


def assemble_results_at(path_to_results, additional_data, filename, field_name="data_fn"):
    subdir_sampling = [_x for _x in listdir(path_to_results) if path.isdir(path.join(path_to_results, _x))]
    out_dict = {}
    for sampling in subdir_sampling:
        path_plus_sampling = path.join(path_to_results, sampling)
        subdir_specifier = [_x for _x in listdir(path_plus_sampling)
                            if path.isdir(path.join(path_plus_sampling, _x))]
        for specifier in subdir_specifier:
            path_plus_spec = path.join(path_plus_sampling, specifier)
            subdir_index = [_x for _x in listdir(path_plus_spec)
                            if path.isdir(path.join(path_plus_spec, _x))]
            for index in subdir_index:
                path_plus_index = path.join(path_plus_spec, index)
                fn = path.abspath(path.join(path_plus_index, filename))
                if path.isfile(fn):
                    this_result = dict([(field_name, fn)])
                    for additional_field_name, additional_field_struc in additional_data.items():
                        this_result[additional_field_name] = additional_field_struc.get2(sampling=sampling,
                                                                                         specifier=specifier,
                                                                                         index=index)
                    out_dict.setdefault(sampling, {}).setdefault(specifier, {})[index] = this_result
    return out_dict


def main(path_to_cfg, stage_name, type_of_classification=None, path_to_output=None):
    if stage_name == "classifier":
        assert type_of_classification is not None, "To repair classifier results, specify which classifier!"
    cfg = Config(path_to_cfg)
    stage = cfg.stage(stage_name)
    path_to_results = stage["other"]
    tribes = TopoData(list(cfg.stage("sample_tribes")["outputs"].values())[0])
    additional_data = assemble_additional_data(additional_fields[stage_name], tribes)

    if path_to_output is None:
        path_to_output = stage["outputs"][outputs_to_use[stage_name].format(type_of_classification)]

    res = assemble_results_at(path_to_results, additional_data,
                              expected_filenames[stage_name].format(type_of_classification))
    with open(path_to_output, "w") as fid:
        json.dump(res, fid, indent=2)


if __name__ == "__main__":
    import sys
    import getopt
    opts, args = getopt.getopt(sys.argv[1:], "o:")
    opts = dict(opts)
    if len(args) < 2:
        print("""
        {0} -- re-generates the .json file that serves as the table of contents for results of the later pipeline
        stages (manifold_analysis, topological_featurization and classifier). For use in case that file gets corrupted
        or lost, but the .h5 files holding the actual data still exist.
        
        Use:
        {0} (-o path/to/output/file) path/to/common_config.json pipeline_stage (type_of_classification)
        
        where pipeline_stage is one of {1}.
        type_of_classification is required for 'classifier' and is one of (manifold, features).
        
        The table of contents file will be placed where the pipeline expects it, according to specifications in the 
        common config, unless overridden by using the -o option
        """.format(__file__, list(additional_fields.keys())))
        sys.exit(2)
    main(*args, path_to_output=opts.get("-o", None))
