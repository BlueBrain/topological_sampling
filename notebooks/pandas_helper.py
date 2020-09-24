import pandas
import numpy


def sanitize_param_name(param):
        param_label = param.replace(" ", "_")
        return "".join([_x for _x in param_label if _x not in ")(-+"])


def column_names_to_values(dframe_in, columns=None, name_col="Column", name_value="Value"):
    if columns is None:
        columns = dframe_in.columns
    out_dict = {}
    for _col in columns:
        out_dict.setdefault(name_col, []).extend([_col] * len(dframe_in))
        out_dict.setdefault(name_value, []).extend(dframe_in[_col])
    return pandas.DataFrame.from_dict(out_dict)


def assemble_result_dataframe(acc_data_struc, param_data_struc,
                              cond_to_iterate="specifier",
                              labels_to_iterate=None,
                              parameters_to_add=None,
                              label_accuracy="Accuracy",
                              sanitize=True,
                              **kwargs):
    
    cols = {}
    if labels_to_iterate is None:
        labels_to_iterate = acc_data_struc.filter(**kwargs).labels_of(cond_to_iterate)
    if parameters_to_add is None:
        parameters_to_add = param_data_struc.data.keys()
    if sanitize:
        param_labels = dict([(param, sanitize_param_name(param)) for param in parameters_to_add])
    else:
        param_labels = dict([(param, param) for param in parameters_to_add])
    kw_cp = kwargs.copy()
    
    for index, label, acc in zip(*acc_data_struc.filter(**kwargs).get_x_y(["index", cond_to_iterate])):
        if label not in labels_to_iterate:
            continue
        cols.setdefault(cond_to_iterate, []).append(label)
        cols.setdefault(label_accuracy, []).append(acc)
        kw_cp.update({cond_to_iterate: label, "index": index})
        for param in parameters_to_add:
            if param not in param_data_struc.data:
                print("{0} not calculated for the samples. Skipping...".format(spec))
                continue
            new_value = param_data_struc[param].get2(**kw_cp)
            if hasattr(new_value, '__len__'):  # returns empty list if no value is found -> use NaN instead
                cols.setdefault(param_labels[param], []).append(numpy.NaN)
            else:
                cols.setdefault(param_labels[param], []).append(new_value)
    return pandas.DataFrame(cols)

