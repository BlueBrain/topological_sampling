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