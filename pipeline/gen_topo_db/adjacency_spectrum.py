import numpy as np
import progressbar
import scipy.linalg


def compute(tribes, adj_matrix, conv, precision):

    spectra = []
    pbar = progressbar.ProgressBar()

    for tribe in pbar(tribes):
        tribe_ids = conv.indices(tribe)
        adj_submat = adj_matrix[np.ix_(tribe_ids, tribe_ids)]

        # Find the eigenvalues
        eig = scipy.linalg.eig(adj_submat.todense())[0]

        # Order the non-zero eigenvalues and round to desired precision
        spectrum = np.unique(np.round(eig[np.nonzero(eig)], precision))
        spectra.append(spectrum)
        
    return spectra
