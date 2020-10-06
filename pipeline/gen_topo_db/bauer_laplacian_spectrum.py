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
import scipy.linalg


def compute(tribes, adj_matrix, conv, precision):

    spectra = []
    #  In csc format we can get the in-degree easily as the diff of the .indptr property
    assert adj_matrix.getformat() == 'csc'
    pbar = progressbar.ProgressBar()

    for tribe in pbar(tribes):
        tribe_ids = conv.indices(tribe)
        adj_submat = adj_matrix[np.ix_(tribe_ids, tribe_ids)]

        # Construct Bauer Laplacian matrix from vertices that are not sources, i.e. all those whose indegree is not zero
        not_source_vertices = np.nonzero(np.diff(adj_submat.indptr))[0]  # Because this is csc format
        tribe_nosources = adj_submat[np.ix_(not_source_vertices, not_source_vertices)]
        size_tribe_nosources = tribe_nosources.shape[0]
        matrix_D_inv = np.diagflat(np.power((size_tribe_nosources -
                                             np.diff(tribe_nosources.indptr)).astype(float),
                                            -1))
        matrix_W = np.transpose(tribe_nosources) 
        matrix_bauer_laplacian = np.subtract(np.eye(size_tribe_nosources, dtype=int),
                                             matrix_D_inv @ matrix_W)

        # Find the eigenvalues
        eig = scipy.linalg.eig(matrix_bauer_laplacian)[0]

        # Order the non-zero eigenvalues and round to desired precision
        spectrum = np.unique(np.round(eig[np.nonzero(eig)], precision))
        spectra.append(spectrum)
        
    return spectra
