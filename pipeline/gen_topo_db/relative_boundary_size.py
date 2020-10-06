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


def compute(tribes, adj_matrix, conv, precision):
    import networkx as nx

    rel_boundaries = []
    pbar = progressbar.ProgressBar()
    G_full = nx.from_scipy_sparse_matrix(adj_matrix)

    for tribe in pbar(tribes):
        # Convert from GIDs to local indexing
        tribe_local_indices = conv.indices(tribe)

        adj_submat = adj_matrix[np.ix_(tribe_local_indices, tribe_local_indices)]
        G = nx.from_scipy_sparse_matrix(adj_submat, create_using=nx.DiGraph)
        edges_in_tribe = G.number_of_edges()
        if edges_in_tribe == 0:
            rel_boundaries.append(0)
        else:
            # Get the edge boundary of a tribe within the full circuit
            boundary = nx.algorithms.boundary.edge_boundary(G_full, tribe_local_indices)

            rel_boundaries.append(np.round(len(list(boundary)) / edges_in_tribe, precision))

    return rel_boundaries
