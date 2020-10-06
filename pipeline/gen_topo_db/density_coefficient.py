"""
Topological sampling
Copyright (C) 2020 Blue Brain Project / EPFL

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import progressbar
from pyflagsercontain import flagser_count


def compute(tribes, adj_matrix, conv, precision):

    # Density coefficients
    dcs = []
    graph_size = adj_matrix.shape[0]
    simplexcontainment = flagser_count(adj_matrix)
    pbar = progressbar.ProgressBar()

    # This assumes tribes are in the order of adjacency matrix indexing
    for i in pbar(range(len(tribes))):
        counts = simplexcontainment[i]

        dc_list = []

        for k in range(2, len(counts)):
            parameter_at_k = k * counts[k] / ((k+1) * (graph_size-k) * counts[k-1]) if counts[k-1] != 0 else 0
            dc_list.append(parameter_at_k)

        dcs.append(dc_list)

    return dcs
