import pyflagser


def compute(adj_sumatrix, parameter="euler", index=None):
    res = pyflagser.flagser_unweighted(adj_sumatrix, directed=True)[parameter]
    if index is not None:
        res = res[index]
    return res
