import numpy as np
import progressbar

import pyflagser

from toposample.indexing import GidConverter


def compute(tribes, adj_matrix, precision):

    ec = []
    pbar = progressbar.ProgressBar()
    conv = GidConverter(tribes)

    for tribe in pbar(tribes):
        # Convert from GIDs to local indexing
        tribe_local_indices = conv.indices(tribe)

        adj_submat = adj_matrix[np.ix_(tribe_local_indices, tribe_local_indices)]
        ec.append(pyflagser.flagser_unweighted(adj_submat, directed=True)['euler'])

    return ec
