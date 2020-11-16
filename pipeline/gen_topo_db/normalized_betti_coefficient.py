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
import pyflagser


def compute(tribes, adj_matrix, conv, precision):

    # Normalized Betti coefficients
    nbcs = []
    pbar = progressbar.ProgressBar()

    for tribe in pbar(tribes):

        # Convert from GIDs to local indexing
        tribe_local_indices = conv.indices(tribe)

        adj_submat = adj_matrix[np.ix_(tribe_local_indices, tribe_local_indices)]
        bettinumbers = pyflagser.flagser_unweighted(adj_submat, directed=True)['betti']
        cellcounts = pyflagser.flagser_unweighted(adj_submat, directed=True)['cell_count']

        parameter = sum(list(map(lambda x: (x+1)*bettinumbers[x]/cellcounts[x]
                                 if cellcounts[x] != 0 else 0,
                                 range(min(len(bettinumbers), len(cellcounts))))))
        nbcs.append(np.round(parameter,precision))

    return nbcs
