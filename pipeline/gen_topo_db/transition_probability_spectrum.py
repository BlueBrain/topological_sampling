"""
toposampling - Topology-assisted sampling and analysis of activity data
Copyright (C) 2020 Blue Brain Project / EPFL & University of Aberdeen

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import progressbar
import scipy.linalg


def compute(tribes, adj_matrix, conv, precision):
    import networkx as nx

    spectra = []
    pbar = progressbar.ProgressBar()

    for tribe in pbar(tribes):
        tribe_ids = conv.indices(tribe)
        adj_submat = adj_matrix[np.ix_(tribe_ids, tribe_ids)]
        G = nx.from_scipy_sparse_matrix(adj_submat, create_using=nx.DiGraph)

        # Find the largest connected component of the graph
        largest = max(nx.strongly_connected_components(G), key=len)

        if len(largest) <= 2:  # Needs at least a certain size...
            spectra.append([])
        else:
            # Adjacency matrix of the tribe's strong component
            tribe_strong_adj_submat = nx.to_numpy_array(G.subgraph(largest), dtype='int8')

            # Make a diagonal matrix of inverses of outdegrees in the tribe
            diag_outdegree_inverses = np.diagflat(np.power(np.sum(tribe_strong_adj_submat, axis=1).astype(float), -1))

            # The transition probability matrix
            tr_prob = diag_outdegree_inverses @ tribe_strong_adj_submat          

            # Find the eigenvalues
            eig = scipy.linalg.eig(tr_prob)[0]

            # Order the non-zero eigenvalues and round to desired precision
            spectrum = np.unique(np.round(eig[np.nonzero(eig)], precision))
            spectra.append(spectrum)
            
    return spectra
