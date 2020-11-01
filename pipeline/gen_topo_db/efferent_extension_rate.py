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
import progressbar


def compute(tribes, adj_matrix, conv, precision):

    extensions = []
    pbar = progressbar.ProgressBar()

    for tribe in pbar(tribes):
        if len(tribe) == 1:
            return 0
        
        tribe_ids = conv.indices(tribe)
        extension_submat = adj_matrix[tribe_ids]

        # Get all the neighbours of all the vertices in the tribe
        all_neighbours = np.nonzero(np.sum(extension_submat, axis=0))[1]

        # Remove tribe itself to get the extension rate
        extension_rate = len(all_neighbours) - len(tribe)

        # Something is funky if this trips
        #assert extension_rate > 0

        extensions.append(extension_rate)
        
    return extensions
