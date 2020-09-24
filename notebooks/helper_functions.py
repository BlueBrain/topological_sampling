import numpy


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

