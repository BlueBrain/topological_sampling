"""
NEUTRINO - NEUral TRIbe and Network Observer
Copyright (C) 2020 Blue Brain Project / EPFL

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np


def make_compute_degree(axis):
    def compute(tribes, adj_matrix, conv, precision):
        return np.array(adj_matrix.sum(axis=axis)).flatten().tolist()

    return compute
