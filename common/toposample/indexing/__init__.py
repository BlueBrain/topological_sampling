import numpy


class GidConverter(object):
    def __init__(self, info):
        self._index = info.index

    def index(self, gid):
        return self._index.get_loc(gid)

    def gid(self, idx):
        return self._index[idx]

    def indices(self, gids):
        return numpy.array([self.index(_g) for _g in gids])

    def gids(self, idxx):
        return self._index[idxx]


def submatrix(gids, M, info):
    """
    submatrix: Gets the submatrix of a neuron sample from the whole adjacency matrix
    :param gids: list; specifies the gids of the sample
    :param M: scipy.sparse.csc; the adjacency matrix of the entire circuit
    :param info: pandas.DataFrame; basic information on all neurons in the circuit
    :return: numpy.array; the adjacency matrix of the neuron sample
    """
    idxb = numpy.in1d(info.index.values, gids)
    return numpy.array(M[:, idxb][idxb].todense())