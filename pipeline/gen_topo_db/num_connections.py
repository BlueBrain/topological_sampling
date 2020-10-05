import numpy as np
import progressbar


def compute(tribes, adj_matrix, conv, precision):

    N = []
    pbar = progressbar.ProgressBar()

    for tribe in pbar(tribes):
        # Convert from GIDs to local indexing
        tribe_local_indices = conv.indices(tribe)

        adj_submat = adj_matrix[np.ix_(tribe_local_indices, tribe_local_indices)]
        N.append(adj_submat.sum())
    return N
