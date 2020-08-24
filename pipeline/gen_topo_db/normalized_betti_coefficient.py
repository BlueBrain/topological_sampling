import numpy as np
import progressbar
from toposample.indexing import GidConverter
import pyflagser


def compute(tribes, adj_matrix, precision):

    # Normalized Betti coefficients
    nbcs = []
    pbar = progressbar.ProgressBar()
    conv = GidConverter(tribes)

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
