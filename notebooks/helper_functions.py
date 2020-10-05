import numpy
from pandas_helper import sanitize_param_name
from statsmodels.formula.api import ols
from scipy.stats import pearsonr


def percentile_of(data_base, data_sampled):
    y = numpy.arange(0, 100)
    x = numpy.percentile(data_base, y)
    return numpy.interp(data_sampled, x, y)


class Percentile(object):
    def __init__(self, data_base):
        self._data = numpy.array(data_base)
        self._data = self._data[~numpy.isnan(self._data)]
    def __call__(self, data_sampled):
        return percentile_of(self._data, data_sampled)


def normalize(data, x=None):
    if x is None:
        x = data
    mn = numpy.nanmin(data)
    mx = numpy.nanmax(data)
    return (numpy.array(x) - mn) / (mx - mn)


class Normalizer(object):
    def __init__(self, data_base):
        self._data = data_base
    def __call__(self, data_sampled):
        return normalize(self._data, x=data_sampled)

def perform_anovas(dframe, result_param, static_params, static_categories, iterated_params=None):
    typ = len(static_params) + len(static_categories) + 1
    assert typ in [1, 2, 3]
    from statsmodels.formula.api import ols
    import statsmodels.api as sm
    
    if iterated_params is None:
        iterated_params = [_x for _x in dframe.columns if ((_x not in static_params)
                          and (_x not in static_categories) and (_x != result_param))]
    
    model_pat = '{0} ~ '.format(result_param)
    i = 1
    pat_strs = []
    for c in static_categories:
        model_pat = model_pat + 'C({0}) + '.format(c)
    for c in static_params:
        model_pat = model_pat + '{0} + '.format(c)
    model_pat = model_pat + '{0}'
    ret = {"columns": [], "p-values": [], "static-p-values": {}}
    for col in iterated_params:
        model = ols(model_pat.format(col), data=dframe).fit()
        anova_table = sm.stats.anova_lm(model, typ=typ)
        pvals = anova_table["PR(>F)"]
        ret
        for c in static_categories:
            ret["static-p-values"].setdefault(c, []).append(pvals['C({0})'.format(c)])
        for c in static_params:
            ret["static-p-values"].setdefault(c, []).append(pvals['{0}'.format(c)])
        ret["columns"].append(col)
        ret["p-values"].append(pvals[col])
    return ret

    
def find_redundant_entries(corr_mat, labels, threshold=0.95):
    decider_mat = corr_mat > threshold
    decider_mat[numpy.eye(decider_mat.shape[0]) == 1] = False
    labels = labels.copy()
    
    if numpy.any(decider_mat):
        red_sum = numpy.sum(decider_mat, axis=0)
        max_num = numpy.max(red_sum)
        idxx = numpy.nonzero(red_sum == max_num)[0]
        to_remove = idxx[0]
        print("{0} is redundant {1} times\n\t...removing".format(labels[to_remove], red_sum[to_remove]))
        if len(idxx) > 1:
            print("\t\t -- although {0} others were equally redundant".format(len(idxx) - 1))
        idxx = list(range(len(labels)))
        idxx.pop(to_remove)
        labels.pop(to_remove)
        return find_redundant_entries(corr_mat[numpy.ix_(idxx, idxx)], labels, threshold=threshold)
    return corr_mat, labels


def variance_explained(y_truth, y_pred):
    var_before = numpy.var(y_truth)
    var_after = numpy.var(y_truth - y_pred)
    return 1.0 - var_after / var_before


def analyze_linear_fit(dframe, columns_x, column_category="specifier", column_y="Accuracy", also_fit_category=True):
    """
    Analyzes the linear dependence of one parameter on another, also (potentially) taking into consideration that the 
    samples come from different categories. 
    Input: dframe: pandas.DataFrame holding the data
    columns_x: list of columns holding the independent variables
    column_category: column holding the category values that were sampled
    column_y: column holding the dependent variable
    also_fit_category (default: True): Should category also be part of the fit?
    
    Returns: For each dependent variable a dict with entries:
         model: the ols model used
         corr: pearson correlation of independent and depdent variable
         pvalue: pvalue of the ols fit of the independent variable
         slope: slope of the ols fit of the independent variable
         var_expl: additional variance explained of the ols model over a model that only used the category values
    """
    if also_fit_category:
        str_model = "{0} ~ C({1}) + {2}"
    else:
        str_model = "{0} ~ {2}"
    model_baseline = ols("{0} ~ C({1})".format(column_y, column_category), data=dframe).fit()
    var_expl_baseline = variance_explained(dframe[column_y], model_baseline.predict(dframe))
    res_out = {}
    for col in columns_x:
        try:
            model = ols(str_model.format(column_y, column_category, sanitize_param_name(col)),
                        data=dframe).fit()
            correlation = pearsonr(dframe[sanitize_param_name(col)], dframe[column_y])
            var_expl_data = variance_explained(dframe[column_y], model.predict(dframe))
            res_out[col] = {"model": model,
                            "corr": correlation,
                            "var_expl": var_expl_data - var_expl_baseline,
                            "pvalue": model.pvalues[sanitize_param_name(col)],
                            "slope": model.params[sanitize_param_name(col)]
                           }
        except: # If values in column are degenerate the fit fails. In that case create "insignificant" results
            res_out[col] = {
                "model": None,
                "corr": 0.0,
                "var_expl": 0.0,
                "pvalue": 1.0,
                "slope": 0.0
            }
    return res_out

