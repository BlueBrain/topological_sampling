import numpy as np
import progressbar
import scipy.linalg


def compute(tribes, adj_matrix, conv, precision):

    spectra = []
    pbar = progressbar.ProgressBar()

    for tribe in pbar(tribes):
        tribe_ids = conv.indices(tribe)
        adj_submat = adj_matrix[np.ix_(tribe_ids, tribe_ids)]

        # Construct Bauer Laplacian matrix from vertices that are not sources, i.e. all those whose indegree is not zero 
        not_source_vertices = np.nonzero(np.any(adj_submat, axis=0))[0]
        tribe_nosources = np.array(adj_submat[np.ix_(not_source_vertices, not_source_vertices)].todense())
        size_tribe_nosources = tribe_nosources.shape[0]
        matrix_D_inv = np.diagflat(np.power((size_tribe_nosources -
                                             np.count_nonzero(tribe_nosources, axis=0)).astype(float),
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
