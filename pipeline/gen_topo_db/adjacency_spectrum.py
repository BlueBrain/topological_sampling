import numpy as np
import progressbar
from numpy import linalg as LA

from toposample.indexing import GidConverter

def compute(tribes, adj_matrix, precision):

    spectra = []
    conv = GidConverter(tribes)
    pbar = progressbar.ProgressBar()

    for tribe in pbar(tribes):
        tribe_ids = conv.indices(tribe)
        adj_submat = adj_matrix[np.ix_(tribe_ids, tribe_ids)]

        # Find the eigenvalues
        eig = LA.eigvals(adj_submat)

        # Order the non-zero eigenvalues and round to desired precision
        spectrum = np.round(np.unique(eig[np.nonzero(eig)]), precision)
        spectra.append(spectrum)
        
    return spectra
