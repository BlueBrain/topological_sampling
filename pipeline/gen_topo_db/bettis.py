import numpy as np
import progressbar

import pyflagser


def compute(tribes, adj_matrix, conv, precision):

    bettis = []
    pbar = progressbar.ProgressBar()

    for tribe in pbar(tribes):
        # Convert from GIDs to local indexing
        tribe_local_indices = conv.indices(tribe)

        adj_submat = adj_matrix[np.ix_(tribe_local_indices, tribe_local_indices)]
        bettis.append(pyflagser.flagser_unweighted(adj_submat, directed=True)['betti'])

    return bettis
