import os
import numpy
import h5py
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from toposample import config, data


def write_results_file(scores, y_truth, y_predicted, out_root, conds):
    out_fn = os.path.join(out_root, conds.get("sampling", "UNSPECIFIED"),
                          conds.get("specifier", "UNSPECIFIED"),
                          conds.get("index", "UNSPECIFIED"), "results.h5")
    assert not os.path.exists(out_fn)
    if not os.path.exists(os.path.split(out_fn)[0]):
        os.makedirs(os.path.split(out_fn)[0])
    with h5py.File(out_fn, 'w') as h5:
        h5.create_dataset("scores", data=scores)
        h5.create_dataset("y_truth", data=y_truth)
        h5.create_dataset("y_pred", data=y_predicted)
    return out_fn


def execute_classifier(features, stage_cfg):
    X = StandardScaler().fit_transform(features[:, :-1])
    y = features[:, -1]

    # dividing X, y into train and test data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=stage_cfg["train_size"], random_state=0)

    clf = SVC(cache_size=500, gamma='scale')
    # cv_scores = cross_val_score(clf, X, y, cv=stage_cfg["num_cv"])
    test_scores = []
    test_truth = []
    test_predictions = []

    for seed in stage_cfg["random_seeds"]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=stage_cfg["train_size"], random_state=seed)
        fitted = clf.fit(X_train, y_train)
        prediction = fitted.predict(X_test)
        test_truth.append(y_test)
        test_predictions.append(prediction)
        test_scores.append((prediction == y_test).mean())

    return test_scores, test_truth, test_predictions


def execute_classifier_all(input_struc, label_struc, stage_cfg, out_root):
    result_lookup = {}
    for res in input_struc.contents:
        print("Evaluating for {0}".format(res.cond))
        scores, y_truth, y_pred = execute_classifier(res.res, stage_cfg)
        out_fn = write_results_file(scores, y_truth, y_pred, out_root, res.cond)
        spec_lvl = result_lookup.setdefault(res.cond["sampling"], {}).setdefault(res.cond["specifier"], {})
        chief = label_struc.get2(**res.cond)
        spec_lvl[res.cond["index"]] = {"data_fn": out_fn, "idv_label": chief}
    return result_lookup


def read_input(input_config):
    features = numpy.load(input_config["features"])
    return features


def read_reshape_stack(fn):
    import h5py
    all_X = []
    with h5py.File(fn, "r") as h5:
        grp = h5["per_stimulus"]
        for k in grp.keys():
            stim_id = int(k[4:])
            per_stim = numpy.array(grp[k]) # time x components x trials
            X = per_stim.reshape((-1, per_stim.shape[-1])) # time-components x trials
            X = numpy.vstack((X, stim_id * numpy.ones(X.shape[1]))).transpose() # trials x time-components+y
            all_X.append(X)
    return numpy.vstack(all_X)


def read_input_manifold(input_config):
    functions_dict = {"data_fn": [read_reshape_stack, False]}
    res = data.TopoData(input_config["components"],
                        follow_link_functions=functions_dict)
    return res


def write_output_for_manifold(res_dict, output_config):
    fn_out = output_config["classifier_manifold_results"]
    if os.path.exists(fn_out):
        with open(fn_out, "r") as fid:
            existing = json.load(fid)
        for k, v in res_dict.items():
            existing.setdefault(k, {}).update(v)
        with open(fn_out, "w") as fid:
            json.dump(existing, fid, indent=2)
    else:
        with open(fn_out, 'w') as fid:
            json.dump(res_dict, fid, indent=2)


def write_output(data, output_config):
    numpy.save(output_config["classifier_test_scores"], data[0])
    numpy.save(output_config["classifier_cv_scores"], data[1])


def main_manifold(path_to_config, **kwargs):
    cfg = config.Config(path_to_config)
    stage = cfg.stage("classifier")
    manifold_data = read_input_manifold(stage["inputs"])
    res_dict = execute_classifier_all(manifold_data["data_fn"].filter(**kwargs),
                                      manifold_data["idv_label"].filter(**kwargs),
                                      stage["config"],
                                      stage["other"])
    write_output_for_manifold(res_dict,
                              stage["outputs"])


def main(path_to_config):
    cfg = config.Config(path_to_config)
    stage = cfg.stage("classifier")
    features = read_input(stage["inputs"])
    scores = execute_classifier(features, stage)
    write_output(scores, stage["outputs"])


def parse_filter_arguments(*args):
    fltr_dict = {}
    for arg in args:
        if "=" in arg:
            splt = arg.split("=")
            fltr_dict[splt[0]] = splt[1]
    return fltr_dict


if __name__ == "__main__":
    import sys
    main_manifold(sys.argv[1], **parse_filter_arguments(*sys.argv[2:]))
