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
        not_source_vertices = np.nonzero(np.array([reduce((lambda x,y: x or y),column) for column in np.transpose(adj_submat)]))[0]
        tribe_nosources = adj_submat[np.ix_(not_source_vertices,not_source_vertices)]
        size_tribe_nosources = len(tribe_nosources)
        matrix_D_inv = np.diagflat(np.power(np.array([size_tribe_nosources - np.count_nonzero(column) for column in np.transpose(tribe_nosources)]).astype(float),-1))
        matrix_W = np.transpose(tribe_nosources) 
        matrix_bauer_laplacian = np.subtract(np.eye(size_tribe_nosources,dtype=int), matrix_D_inv @ matrix_W)

        # Find the eigenvalues
        eig = scipy.linalg.eig(matrix_bauer_laplacian)[0]

        # Order the non-zero eigenvalues and round to desired precision
        spectrum = np.unique(np.round(eig[np.nonzero(eig)], precision))
        spectra.append(spectrum)
        
    return spectra
